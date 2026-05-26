# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torchdata.stateful_dataloader.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX


def _jsonl_offsets_for_rank(
    dataset_path: Path,
    *,
    dp_rank: int | None,
    dp_world_size: int,
) -> tuple[list[int], int]:
    offsets: list[int] = []
    record_index = 0
    with dataset_path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            if dp_rank is None or record_index % dp_world_size == dp_rank:
                offsets.append(offset)
            record_index += 1
    return offsets, record_index


class _SweHeroJsonlDataset(IterableDataset, Stateful):
    def __init__(
        self,
        *,
        dataset_path: Path,
        offsets: list[int],
        dp_rank: int,
        dp_world_size: int,
        seed: int,
        shuffle: bool,
        infinite: bool,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.offsets = offsets
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.seed = seed
        self.shuffle = shuffle
        self.infinite = infinite
        self._epoch = 0
        self._offset = 0

    def _offsets_for_iterator(self) -> list[int]:
        offsets = self.offsets
        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            offsets = offsets[worker.id :: worker.num_workers] or offsets
        return offsets

    def __iter__(self) -> Iterator[dict[str, Any]]:
        offsets = self._offsets_for_iterator()
        epoch = self._epoch
        offset = self._offset
        with self.dataset_path.open("rb") as handle:
            while True:
                order = list(range(len(offsets)))
                if self.shuffle:
                    random.Random(self.seed + self.dp_rank + epoch * 1_000_003).shuffle(
                        order
                    )
                while offset < len(order):
                    index = order[offset]
                    offset += 1
                    self._epoch = epoch
                    self._offset = offset
                    handle.seek(offsets[index])
                    yield json.loads(handle.readline())
                epoch += 1
                offset = 0
                self._epoch = epoch
                self._offset = offset
                if not self.infinite:
                    break

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "offset": self._offset,
            "dp_rank": self.dp_rank,
            "dp_world_size": self.dp_world_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "infinite": self.infinite,
            "record_count": len(self._offsets_for_iterator()),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        expected = {
            "dp_rank": self.dp_rank,
            "dp_world_size": self.dp_world_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "infinite": self.infinite,
            "record_count": len(self._offsets_for_iterator()),
        }
        mismatches = {
            key: {"expected": expected[key], "actual": state_dict.get(key)}
            for key in expected
            if state_dict.get(key) != expected[key]
        }
        if mismatches:
            raise ValueError(
                "SWE-HERO dataloader checkpoint does not match this dataloader: "
                f"{mismatches}"
            )
        self._epoch = int(state_dict["epoch"])
        self._offset = int(state_dict["offset"])


class SweHeroDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset_path: str | None = None
        pad_token_id: int = 0
        seed: int = 17
        shuffle: bool = True
        infinite: bool = True
        allow_empty_rank_reuse: bool = False

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer,
        seq_len: int,
        local_batch_size: int,
    ) -> None:
        if config.dataset_path is None:
            raise ValueError("SweHeroDataLoader requires dataset_path")
        dataset_path = Path(config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"SWE-HERO bucket file not found: {dataset_path}")

        offsets, total_records = _jsonl_offsets_for_rank(
            dataset_path,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
        if total_records == 0:
            raise ValueError(f"SWE-HERO bucket file is empty: {dataset_path}")
        if not offsets:
            if not config.allow_empty_rank_reuse:
                raise ValueError(
                    "SWE-HERO bucket file has no records for "
                    f"dp_rank={dp_rank} with dp_world_size={dp_world_size}: "
                    f"{dataset_path}. Refusing to reuse data on an empty rank."
                )
            # Tiny smoke buckets can have fewer examples than DP ranks. Reuse
            # the bucket on empty ranks so distributed smoke tests do not fail
            # before the full filtered dataset is ready.
            offsets, _total_records = _jsonl_offsets_for_rank(
                dataset_path,
                dp_rank=None,
                dp_world_size=dp_world_size,
            )

        self.seq_len = seq_len
        self.pad_token_id = config.pad_token_id
        dataset = _SweHeroJsonlDataset(
            dataset_path=dataset_path,
            offsets=offsets,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            seed=config.seed,
            shuffle=config.shuffle,
            infinite=config.infinite,
        )
        self._swehero_dataset = dataset
        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=local_batch_size,
            collate_fn=self._collate,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
        )

    def _legacy_dataset_state_from_num_yielded(
        self, num_yielded: int
    ) -> dict[str, Any]:
        offsets = self._swehero_dataset._offsets_for_iterator()
        consumed_records = num_yielded * int(self.batch_size or 1)
        state = self._swehero_dataset.state_dict()
        state["epoch"] = consumed_records // len(offsets)
        state["offset"] = consumed_records % len(offsets)
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict and self._rank_id in state_dict:
            rank_state = pickle.loads(state_dict[self._rank_id])
            if rank_state.get("dataset_state") is None:
                rank_state["dataset_state"] = self._legacy_dataset_state_from_num_yielded(
                    int(rank_state.get("_num_yielded", 0))
                )
                state_dict = dict(state_dict)
                state_dict[self._rank_id] = pickle.dumps(rank_state)
        super().load_state_dict(state_dict)

    def _collate(
        self, records: list[dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        batch_size = len(records)
        input_ids = torch.full(
            (batch_size, self.seq_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, self.seq_len),
            IGNORE_INDEX,
            dtype=torch.long,
        )

        for row, record in enumerate(records):
            ids = record["input_ids"]
            row_labels = record["labels"]
            length = min(len(ids), self.seq_len)
            if len(ids) > self.seq_len:
                raise ValueError(
                    f"record length {len(ids)} exceeds bucket seq_len {self.seq_len}"
                )
            input_ids[row, :length] = torch.tensor(ids[:length], dtype=torch.long)
            labels[row, :length] = torch.tensor(row_labels[:length], dtype=torch.long)

        positions = torch.arange(self.seq_len, dtype=torch.int32).repeat(batch_size, 1)
        return {"input": input_ids, "positions": positions}, labels
