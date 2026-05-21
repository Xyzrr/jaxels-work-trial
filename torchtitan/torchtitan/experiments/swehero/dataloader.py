# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX


class _SweHeroJsonlDataset(IterableDataset):
    def __init__(
        self,
        *,
        records: list[dict[str, Any]],
        dp_rank: int,
        dp_world_size: int,
        seed: int,
        shuffle: bool,
        infinite: bool,
    ) -> None:
        super().__init__()
        self.records = records
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.seed = seed
        self.shuffle = shuffle
        self.infinite = infinite

    def _records_for_iterator(self) -> list[dict[str, Any]]:
        records = self.records[self.dp_rank :: self.dp_world_size]
        if not records:
            # Tiny smoke buckets can have fewer examples than DP ranks. Reuse
            # the bucket on empty ranks so distributed smoke tests do not fail
            # before the full filtered dataset is ready.
            records = self.records

        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            records = records[worker.id :: worker.num_workers] or records
        return records

    def __iter__(self) -> Iterator[dict[str, Any]]:
        records = self._records_for_iterator()
        epoch = 0
        while True:
            order = list(range(len(records)))
            if self.shuffle:
                random.Random(self.seed + self.dp_rank + epoch * 1_000_003).shuffle(order)
            for index in order:
                yield records[index]
            epoch += 1
            if not self.infinite:
                break


class SweHeroDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset_path: str | None = None
        pad_token_id: int = 0
        seed: int = 17
        shuffle: bool = True
        infinite: bool = True

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

        records = [
            json.loads(line)
            for line in dataset_path.read_text().splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError(f"SWE-HERO bucket file is empty: {dataset_path}")

        self.seq_len = seq_len
        self.pad_token_id = config.pad_token_id
        dataset = _SweHeroJsonlDataset(
            records=records,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            seed=config.seed,
            shuffle=config.shuffle,
            infinite=config.infinite,
        )
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
