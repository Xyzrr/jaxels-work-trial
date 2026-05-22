import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "torchtitan"))

try:
    import torch
    from torchtitan.experiments.swehero.dataloader import SweHeroDataLoader
except ModuleNotFoundError:
    torch = None
    SweHeroDataLoader = None


@unittest.skipIf(torch is None, "torch/torchdata runtime is not available")
class SweHeroDataLoaderResumeTests(unittest.TestCase):
    def _write_bucket(self, path: Path, count: int) -> None:
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

        self.assertEqual(inputs["input"][:, 0].tolist(), [2, 4])
        self.assertEqual(labels[:, 0].tolist(), [2, 4])

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

        self.assertEqual(inputs["input"][0, 0].item(), 1)
        self.assertEqual(labels[0, 0].item(), 1)

    def test_empty_rank_reuse_is_rejected_unless_explicitly_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            bucket_path = Path(tmp) / "bucket_8.jsonl"
            self._write_bucket(bucket_path, count=1)

            with self.assertRaisesRegex(ValueError, "Refusing to reuse data"):
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
            self.assertTrue(
                torch.equal(expected_inputs["input"], resumed_inputs["input"])
            )
            self.assertTrue(
                torch.equal(expected_inputs["positions"], resumed_inputs["positions"])
            )
            self.assertTrue(torch.equal(expected_labels, resumed_labels))

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
            self.assertTrue(
                torch.equal(expected_inputs["input"], resumed_inputs["input"])
            )
            self.assertTrue(torch.equal(expected_labels, resumed_labels))


if __name__ == "__main__":
    unittest.main()
