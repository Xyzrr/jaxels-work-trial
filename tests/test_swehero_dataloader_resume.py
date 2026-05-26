"""Regression tests for TorchTitan SWE-HERO dataloader resume behavior.

The production trainer feeds tokenized SWE traces to a causal language model:
``input_ids`` are the token IDs the model reads, and ``labels`` are the target
token IDs used to compute loss. TorchTitan also runs with data parallelism, so
each GPU rank must see a deterministic slice of the JSONL bucket and must resume
from the exact same slice after checkpoint restore.

These tests are deliberately small, but they protect training correctness. A
resume bug here would not look like a Python exception in a long run; it would
silently change which examples the model trains on after a restart.
"""

import json
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "torchtitan"))

try:
    import torch
    from torchtitan.experiments.swehero.dataloader import SweHeroDataLoader
except ModuleNotFoundError:
    torch = None
    SweHeroDataLoader = None


@pytest.mark.skipif(torch is None, reason="torch/torchdata runtime is not available")
class TestSweHeroDataLoaderResume:
    def _write_bucket(self, path: Path, count: int) -> None:
        """Write a tiny tokenized bucket in the same shape as training data."""

        with path.open("w") as handle:
            for index in range(count):
                token = index + 1
                handle.write(
                    json.dumps(
                        {
                            "input_ids": [token, token + 100],
                            "labels": [token, token + 100],
                            "bucket": 8,
                            "source_id": f"example-{index}",
                        }
                    )
                    + "\n"
                )

    def _build_loader(
        self,
        bucket_path: Path,
        *,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        shuffle: bool = True,
        allow_empty_rank_reuse: bool = False,
    ) -> "SweHeroDataLoader":
        assert SweHeroDataLoader is not None
        # The real bucket files already contain token IDs and loss labels after
        # preprocessing. This test bypasses tokenization so it can focus only on
        # data-parallel partitioning and checkpoint resume state.
        config = SweHeroDataLoader.Config(
            dataset_path=str(bucket_path),
            pad_token_id=0,
            seed=123,
            shuffle=shuffle,
            infinite=True,
            allow_empty_rank_reuse=allow_empty_rank_reuse,
            pin_memory=False,
        )
        return SweHeroDataLoader(
            config,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=None,
            seq_len=8,
            local_batch_size=2,
        )

    def test_loader_streams_rank_offsets_without_full_jsonl_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            bucket_path = Path(tmp) / "bucket_8.jsonl"
            self._write_bucket(bucket_path, count=6)

            # Large training buckets should be streamed by byte offset. A full
            # read would scale with the entire dataset before training starts and
            # would make huge SWE trace buckets painful to resume.
            with patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("full JSONL read is not allowed"),
            ):
                loader = self._build_loader(
                    bucket_path,
                    dp_world_size=2,
                    dp_rank=1,
                    shuffle=False,
                )
                inputs, labels = next(iter(loader))

        # With data parallelism, rank 1 of 2 receives records whose global index
        # is 1, 3, 5, ... . The first token in each synthetic row makes that
        # partition visible as token values 2 and 4 in the first batch.
        assert inputs["input"][:, 0].tolist() == [2, 4]
        assert labels[:, 0].tolist() == [2, 4]

    def test_empty_rank_reuses_tiny_bucket_for_smoke_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            bucket_path = Path(tmp) / "bucket_8.jsonl"
            self._write_bucket(bucket_path, count=1)

            loader = self._build_loader(
                bucket_path,
                dp_world_size=4,
                dp_rank=3,
                shuffle=False,
                allow_empty_rank_reuse=True,
            )
            inputs, labels = next(iter(loader))

        # Smoke buckets can be smaller than the number of GPU ranks. Explicit
        # reuse lets every rank exercise the training loop, but this is only safe
        # for smoke tests because it repeats data across ranks.
        assert inputs["input"][0, 0].item() == 1
        assert labels[0, 0].item() == 1

    def test_empty_rank_reuse_is_rejected_unless_explicitly_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            bucket_path = Path(tmp) / "bucket_8.jsonl"
            self._write_bucket(bucket_path, count=1)

            with pytest.raises(ValueError, match="Refusing to reuse data"):
                self._build_loader(
                    bucket_path,
                    dp_world_size=4,
                    dp_rank=3,
                    shuffle=False,
                )

    def test_state_dict_restores_next_batch_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            bucket_path = Path(tmp) / "bucket_8.jsonl"
            self._write_bucket(bucket_path, count=7)

            loader = self._build_loader(bucket_path)
            iterator = iter(loader)
            for _ in range(3):
                next(iterator)

            # Training checkpoints save dataloader state alongside model and
            # optimizer state. The next batches after restore must match exactly
            # or a resumed run trains on a different curriculum than the
            # uninterrupted run.
            state = loader.state_dict()
            expected_batches = [next(iterator) for _ in range(6)]

            resumed_loader = self._build_loader(bucket_path)
            resumed_loader.load_state_dict(state)
            resumed_iterator = iter(resumed_loader)
            resumed_batches = [next(resumed_iterator) for _ in range(6)]

        for (expected_inputs, expected_labels), (
            resumed_inputs,
            resumed_labels,
        ) in zip(expected_batches, resumed_batches):
            assert torch.equal(expected_inputs["input"], resumed_inputs["input"])
            # Positions tell the transformer where each token sits in the
            # sequence. Matching token IDs is not enough if positions drift.
            assert torch.equal(
                expected_inputs["positions"], resumed_inputs["positions"]
            )
            assert torch.equal(expected_labels, resumed_labels)

    def test_legacy_state_dict_restores_from_batch_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            bucket_path = Path(tmp) / "bucket_8.jsonl"
            self._write_bucket(bucket_path, count=7)

            loader = self._build_loader(bucket_path)
            iterator = iter(loader)
            for _ in range(3):
                next(iterator)

            state = loader.state_dict()
            rank_state = pickle.loads(state["dp_rank_0"])
            # Older checkpoints only recorded how many batches a rank had
            # yielded. The loader reconstructs epoch/offset from that count so
            # older runs can still resume after the state format became richer.
            rank_state["dataset_state"] = None
            state["dp_rank_0"] = pickle.dumps(rank_state)
            expected_batches = [next(iterator) for _ in range(6)]

            resumed_loader = self._build_loader(bucket_path)
            resumed_loader.load_state_dict(state)
            resumed_iterator = iter(resumed_loader)
            resumed_batches = [next(resumed_iterator) for _ in range(6)]

        for (expected_inputs, expected_labels), (
            resumed_inputs,
            resumed_labels,
        ) in zip(expected_batches, resumed_batches):
            assert torch.equal(expected_inputs["input"], resumed_inputs["input"])
            assert torch.equal(expected_labels, resumed_labels)
