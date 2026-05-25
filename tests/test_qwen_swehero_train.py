import ast
import contextlib
import hashlib
import io
import json
import os
import shlex
import signal
import sys
import tempfile
import types
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import qwen_swehero_train as train


class FakeTokenizer:
    bos_id = None
    eos_id = None
    pad_id = 0

    def encode(self, text, **kwargs):
        return [ord(ch) for ch in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _BatchBackend:
    def __init__(self, owner):
        self.owner = owner

    def encode_batch(self, texts):
        self.owner.batch_calls += 1
        return [
            types.SimpleNamespace(ids=self.owner._encode_ids(text)) for text in texts
        ]


class BatchFakeTokenizer(FakeTokenizer):
    def __init__(self):
        self.batch_calls = 0
        self.encode_calls = 0
        self.tokenizer = _BatchBackend(self)

    def _encode_ids(self, text):
        return [ord(ch) for ch in text]

    def encode(self, text, **kwargs):
        self.encode_calls += 1
        return self._encode_ids(text)


class TestQwenSweHeroTorchTitanLauncher:
    @staticmethod
    def _legacy_encode_swehero_example(
        tokenizer,
        example: dict[str, object],
        *,
        max_length: int,
        min_trainable_tokens: int,
        include_model_patch: bool = False,
    ) -> dict[str, object] | None:
        token_ids: list[int] = []
        labels: list[int] = []

        bos_id = getattr(tokenizer, "bos_id", getattr(tokenizer, "bos_token_id", None))
        if bos_id is not None:
            token_ids.append(int(bos_id))
            labels.append(train.IGNORE_INDEX)

        for text, is_trainable in train.qwen_openhands_segments(
            example, include_model_patch=include_model_patch
        ):
            ids = train._tokenize_text(tokenizer, text)
            token_ids.extend(ids)
            labels.extend(ids if is_trainable else [train.IGNORE_INDEX] * len(ids))

        eos_id = getattr(tokenizer, "eos_id", getattr(tokenizer, "eos_token_id", None))
        if eos_id is not None:
            token_ids.append(int(eos_id))
            labels.append(
                int(eos_id)
                if labels and labels[-1] != train.IGNORE_INDEX
                else train.IGNORE_INDEX
            )

        if len(token_ids) > max_length + 1:
            raise train.LongExampleError(
                token_count=len(token_ids),
                max_length=max_length,
            )
        if len(token_ids) < 2:
            return None

        shifted_input_ids = token_ids[:-1]
        shifted_labels = labels[1:]
        trainable_tokens = sum(label != train.IGNORE_INDEX for label in shifted_labels)
        if trainable_tokens < min_trainable_tokens:
            return None

        return {
            "input_ids": shifted_input_ids,
            "labels": shifted_labels,
            "length": len(shifted_input_ids),
            "trainable_tokens": trainable_tokens,
        }

    def _expected_materialization_from_legacy_encoder(
        self,
        *,
        args,
        tokenizer,
        examples,
    ) -> dict[str, object]:
        buckets = train.parse_bucket_list(args.buckets)
        bucket_lines: dict[str, list[str]] = {str(bucket): [] for bucket in buckets}
        bucket_counts: Counter[int] = Counter()
        bucket_source_ids: dict[int, list[str]] = {bucket: [] for bucket in buckets}
        bucket_lengths: dict[int, list[int]] = {bucket: [] for bucket in buckets}
        bucket_trainable_tokens: dict[int, list[int]] = {
            bucket: [] for bucket in buckets
        }
        bucket_length_histograms: dict[int, Counter[int]] = {
            bucket: Counter() for bucket in buckets
        }
        length_histogram: Counter[int] = Counter()
        skipped: Counter[str] = Counter()
        skipped_source_ids_by_reason: dict[str, list[str]] = {}
        streamed_source_ids: list[str] = []
        included_source_ids: list[str] = []
        long_examples_sample: list[dict[str, object]] = []
        streamed_examples = 0
        usable_examples = 0

        for example in examples:
            streamed_examples += 1
            source_id = train._example_id(example, streamed_examples)
            streamed_source_ids.append(source_id)
            try:
                encoded = self._legacy_encode_swehero_example(
                    tokenizer,
                    example,
                    max_length=args.max_length,
                    min_trainable_tokens=args.min_trainable_tokens,
                    include_model_patch=args.include_model_patch,
                )
            except train.LongExampleError as exc:
                reason = "too_long_for_max_length"
                skipped[reason] += 1
                skipped_source_ids_by_reason.setdefault(reason, []).append(source_id)
                if len(long_examples_sample) < 20:
                    long_examples_sample.append(
                        {
                            "source_id": source_id,
                            "token_count": exc.token_count,
                            "shifted_input_length": exc.shifted_input_length,
                            "max_length": exc.max_length,
                        }
                    )
                if args.long_example_policy == "error":
                    raise
                encoded = None

            if encoded is None:
                if skipped_source_ids_by_reason.get("too_long_for_max_length", [])[
                    -1:
                ] != [source_id]:
                    reason = "not_enough_trainable_tokens"
                    skipped[reason] += 1
                    skipped_source_ids_by_reason.setdefault(reason, []).append(
                        source_id
                    )
            else:
                try:
                    bucket = train.choose_bucket(int(encoded["length"]), buckets)
                except ValueError:
                    reason = "too_long_for_largest_bucket"
                    skipped[reason] += 1
                    skipped_source_ids_by_reason.setdefault(reason, []).append(
                        source_id
                    )
                else:
                    record = {**encoded, "bucket": bucket, "source_id": source_id}
                    line = json.dumps(record) + "\n"
                    bucket_lines[str(bucket)].append(line)
                    bucket_counts[bucket] += 1
                    included_source_ids.append(source_id)
                    bucket_source_ids[bucket].append(source_id)
                    bucket_lengths[bucket].append(int(encoded["length"]))
                    bucket_trainable_tokens[bucket].append(
                        int(encoded["trainable_tokens"])
                    )
                    rounded_length = ((int(encoded["length"]) + 1023) // 1024) * 1024
                    length_histogram[rounded_length] += 1
                    bucket_length_histograms[bucket][rounded_length] += 1
                    usable_examples += 1

            if args.num_examples > 0 and usable_examples >= args.num_examples:
                break
            if (
                args.max_streamed_examples > 0
                and streamed_examples >= args.max_streamed_examples
            ):
                break

        bucket_file_integrity = {}
        for bucket in buckets:
            payload = "".join(bucket_lines[str(bucket)]).encode()
            bucket_file_integrity[str(bucket)] = {
                "bytes": len(payload),
                "records": len(bucket_lines[str(bucket)]),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

        return {
            "bucket_lines": bucket_lines,
            "bucket_counts": {str(bucket): bucket_counts[bucket] for bucket in buckets},
            "bucket_file_integrity": bucket_file_integrity,
            "length_histogram_rounded_to_1024": {
                str(length): count for length, count in sorted(length_histogram.items())
            },
            "num_usable_examples": usable_examples,
            "streamed_examples_scanned": streamed_examples,
            "skipped": dict(skipped),
            "streamed_source_ids": streamed_source_ids,
            "included_source_ids": included_source_ids,
            "skipped_source_ids_by_reason": skipped_source_ids_by_reason,
            "bucket_source_ids": bucket_source_ids,
            "bucket_lengths": bucket_lengths,
            "bucket_trainable_tokens": bucket_trainable_tokens,
            "bucket_length_histograms": bucket_length_histograms,
            "long_examples_sample": long_examples_sample,
        }

    def _git_state(
        self,
        *,
        dirty: bool = False,
        available: bool = True,
        status_short: str = "",
    ) -> dict:
        return {
            "schema_version": train.GIT_STATE_SCHEMA_VERSION,
            "repo_root": str(train.CANONICAL_WORKSPACE_ROOT),
            "available": available,
            "top_level": str(train.CANONICAL_WORKSPACE_ROOT) if available else None,
            "branch": "main" if available else None,
            "commit": "a" * 40 if available else None,
            "status_short": status_short if available else None,
            "dirty": dirty if available else None,
        }

    def _resume_test_setup(self, tmp: str):
        out_dir = Path(tmp) / "run"
        data_dir = out_dir / "data"
        data_dir.mkdir(parents=True)
        args = train.parse_args(
            [
                "--out-dir",
                str(out_dir),
                "--dataset-path",
                str(Path(tmp) / "dataset"),
                "--hf-assets-path",
                str(Path(tmp) / "hf" / "Qwen2.5-Coder-7B-Instruct"),
                "--buckets",
                "32768,65536",
                "--bucket-cp",
                "32768:2,65536:4",
                "--max-length",
                "65536",
                "--num-examples",
                "34",
                "--max-streamed-examples",
                "100",
            ]
        )
        args.buckets = ",".join(str(b) for b in train.parse_bucket_list(args.buckets))
        bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)
        args.bucket_cp = train._format_bucket_cp_map(bucket_cp)
        bucket_files = {
            32768: data_dir / "bucket_32768.jsonl",
            65536: data_dir / "bucket_65536.jsonl",
        }
        for path in bucket_files.values():
            path.write_text("")
        manifest = {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "dataset_id": args.dataset_id,
            "dataset_path": str(args.dataset_path),
            "dataset_artifact": {
                "path": str(args.dataset_path),
                "metadata_json_sha256": "metadata-sha",
                "selection_manifest_sha256": "selection-sha",
                "data_files": [],
                "total_data_bytes": 0,
            },
            "source_dataset_id": args.source_dataset_id,
            "source_dataset_revision": {
                "requested_revision": args.source_dataset_revision,
                "resolved_sha": "source-sha",
            },
            "model_assets": {
                "schema_version": train.MODEL_ASSET_PROVENANCE_SCHEMA_VERSION,
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "hf_assets_path": str(args.hf_assets_path),
                "hf_assets_realpath": str(args.hf_assets_path),
                "file_count": 1,
                "total_bytes": 10,
                "files": [
                    {
                        "path": "config.json",
                        "kind": "model_config",
                        "bytes": 10,
                        "sha256": "config-sha",
                    }
                ],
                "config": {
                    "path": "config.json",
                    "sha256": "config-sha",
                    "summary": {"model_type": "qwen2"},
                    "json_error": None,
                },
                "generation_config": {
                    "path": None,
                    "sha256": None,
                    "summary": {},
                    "json_error": None,
                },
                "safetensors": {
                    "index_path": None,
                    "index_sha256": None,
                    "metadata": {},
                    "weight_map_entries": 0,
                    "shard_files": [],
                    "unindexed_safetensors_files": [],
                    "index_error": None,
                },
                "tokenizer": {
                    "hf_assets_path": str(args.hf_assets_path),
                    "tokenizer_json_sha256": "tokenizer-sha",
                    "tokenizer_config_sha256": "tokenizer-config-sha",
                },
            },
            "tokenizer": {
                "hf_assets_path": str(args.hf_assets_path),
                "tokenizer_json_sha256": "tokenizer-sha",
                "tokenizer_config_sha256": "tokenizer-config-sha",
                "chat_template_sha256": "chat-template-sha",
                "bos_id": None,
                "eos_id": 151645,
                "pad_id": 151643,
                "trace_serializer": "test serializer",
            },
            "pad_token_id": 151643,
            "max_length": args.max_length,
            "buckets": [32768, 65536],
            "bucket_files": {
                str(bucket): str(path) for bucket, path in bucket_files.items()
            },
            "bucket_counts": {"32768": 33, "65536": 1},
            "num_usable_examples": 34,
            "streamed_examples_scanned": 34,
            "skipped": {},
            "include_model_patch": args.include_model_patch,
        }
        (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        plan = train.build_bucket_plan(
            bucket_counts={32768: 33, 65536: 1},
            bucket_files=bucket_files,
            bucket_cp=bucket_cp,
            epochs=args.num_train_epochs,
            global_batch_size=args.global_batch_size,
            warmup_ratio=args.warmup_ratio,
        )
        return args, manifest, plan

    def _materialize_with_fake_runtime(self, args, examples=(), *, synthetic=False):
        fake_tokenizer_module = types.ModuleType("torchtitan.components.tokenizer")

        class FakeHuggingFaceTokenizer(FakeTokenizer):
            def __init__(self, tokenizer_path):
                self.tokenizer_path = tokenizer_path

        fake_tokenizer_module.HuggingFaceTokenizer = FakeHuggingFaceTokenizer

        with (
            patch.dict(
                sys.modules,
                {
                    "torchtitan": types.ModuleType("torchtitan"),
                    "torchtitan.components": types.ModuleType("torchtitan.components"),
                    "torchtitan.components.tokenizer": fake_tokenizer_module,
                },
            ),
            patch.object(train, "load_training_dataset", return_value=iter(examples)),
            patch.object(
                train,
                "_dataset_artifact_metadata",
                return_value={"path": str(args.dataset_path), "data_files": []},
            ),
            patch.object(
                train,
                "_dataset_revision_info",
                return_value={"requested_revision": args.source_dataset_revision},
            ),
            patch.object(
                train,
                "_tokenizer_metadata",
                return_value={
                    "hf_assets_path": str(args.hf_assets_path),
                    "pad_id": 0,
                },
            ),
            patch.object(
                train,
                "_model_asset_provenance",
                return_value={
                    "schema_version": train.MODEL_ASSET_PROVENANCE_SCHEMA_VERSION,
                    "model_id": args.model_id,
                    "model_revision": args.model_revision,
                    "hf_assets_path": str(args.hf_assets_path),
                    "hf_assets_realpath": str(args.hf_assets_path),
                    "file_count": 1,
                    "total_bytes": 10,
                    "files": [
                        {
                            "path": "config.json",
                            "kind": "model_config",
                            "bytes": 10,
                            "sha256": "config-sha",
                        }
                    ],
                    "config": {
                        "path": "config.json",
                        "sha256": "config-sha",
                        "summary": {"model_type": "qwen2"},
                        "json_error": None,
                    },
                    "generation_config": {
                        "path": None,
                        "sha256": None,
                        "summary": {},
                        "json_error": None,
                    },
                    "safetensors": {
                        "index_path": None,
                        "index_sha256": None,
                        "metadata": {},
                        "weight_map_entries": 0,
                        "shard_files": [],
                        "unindexed_safetensors_files": [],
                        "index_error": None,
                    },
                    "tokenizer": {
                        "hf_assets_path": str(args.hf_assets_path),
                        "pad_id": 0,
                    },
                },
            ),
            patch.object(train, "_package_versions", return_value={}),
            patch.object(train, "_run_git", return_value=None),
        ):
            if synthetic:
                return train.materialize_synthetic_smoke_buckets(args)
            return train.materialize_training_buckets(args)

    def _real_swehero_dataset_path(self) -> Path:
        dataset_path = Path(
            os.environ.get(
                "SWEHERO_TEST_DATASET_PATH",
                Path(__file__).resolve().parents[1]
                / "datasets"
                / "swe-hero-openhands-trajectories-5b2ed21-one-rollout",
            )
        )
        if not (dataset_path / "data").is_dir():
            pytest.skip(f"SWE-HERO test dataset is not available: {dataset_path}")
        return dataset_path

    def _real_qwen_hf_assets_path(self) -> Path:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            Path(value)
            for value in [
                os.environ.get("SWEHERO_TEST_HF_ASSETS_PATH"),
                str(
                    repo_root
                    / "tmp"
                    / "hf"
                    / "Qwen2.5-Coder-7B-Instruct-tokenizer-only"
                ),
                "/workspace/assets/hf/Qwen2.5-Coder-7B-Instruct",
            ]
            if value
        ]
        for candidate in candidates:
            if (candidate / "tokenizer.json").is_file() and (
                candidate / "tokenizer_config.json"
            ).is_file():
                return candidate
        pytest.skip(
            "Qwen tokenizer assets are not available; set SWEHERO_TEST_HF_ASSETS_PATH"
        )

    def _mini_qwen_tokenizer_class(self):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            pytest.skip(f"tokenizers is not available: {exc}")

        class MiniHuggingFaceTokenizer:
            def __init__(self, tokenizer_path: str):
                self.tokenizer_path = tokenizer_path
                root = Path(tokenizer_path)
                self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
                config = json.loads((root / "tokenizer_config.json").read_text())

                def token_content(key: str) -> str | None:
                    value = config.get(key)
                    if isinstance(value, Mapping):
                        value = value.get("content")
                    return value if isinstance(value, str) else None

                bos_token = token_content("bos_token")
                eos_token = token_content("eos_token")
                pad_token = token_content("pad_token")
                self.bos_id = (
                    self.tokenizer.token_to_id(bos_token) if bos_token else None
                )
                self.eos_id = (
                    self.tokenizer.token_to_id(eos_token) if eos_token else None
                )
                self.pad_id = (
                    self.tokenizer.token_to_id(pad_token) if pad_token else None
                )

            def encode(self, text, add_bos=False, add_eos=False):
                ids = list(self.tokenizer.encode(text).ids)
                if add_bos and self.bos_id is not None:
                    ids.insert(0, self.bos_id)
                if add_eos and self.eos_id is not None:
                    ids.append(self.eos_id)
                return ids

            def decode(self, ids):
                return self.tokenizer.decode(list(ids))

            def token_to_id(self, token):
                return self.tokenizer.token_to_id(token)

        return MiniHuggingFaceTokenizer

    def _patch_torchtitan_tokenizer_module(self, tokenizer_class):
        fake_tokenizer_module = types.ModuleType("torchtitan.components.tokenizer")
        fake_tokenizer_module.HuggingFaceTokenizer = tokenizer_class
        return patch.dict(
            sys.modules,
            {
                "torchtitan": types.ModuleType("torchtitan"),
                "torchtitan.components": types.ModuleType("torchtitan.components"),
                "torchtitan.components.tokenizer": fake_tokenizer_module,
            },
        )

    def _require_real_materialization_dependencies(self):
        try:
            import pyarrow  # noqa: F401

            import datasets  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"real SWE-HERO materialization deps unavailable: {exc}")

    def _write_preflight_hf_assets(self, hf_assets: Path) -> None:
        hf_assets.mkdir(parents=True, exist_ok=True)
        (hf_assets / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
        )
        (hf_assets / "tokenizer.json").write_text('{"version":"1.0"}')
        (hf_assets / "tokenizer_config.json").write_text("{}")
        (hf_assets / "model-00001-of-00001.safetensors").write_bytes(b"shard")
        (hf_assets / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 5},
                    "weight_map": {
                        "model.embed_tokens.weight": (
                            "model-00001-of-00001.safetensors"
                        )
                    },
                }
            )
        )

    def _model_assets_manifest(self, args) -> dict:
        tokenizer_metadata = {
            "hf_assets_path": str(args.hf_assets_path),
            "tokenizer_json_sha256": train._hash_file(
                args.hf_assets_path / "tokenizer.json"
            ),
            "tokenizer_config_sha256": train._hash_file(
                args.hf_assets_path / "tokenizer_config.json"
            ),
        }
        return {
            "model_assets": train._model_asset_provenance(
                model_id=args.model_id,
                model_revision=args.model_revision,
                hf_assets_path=args.hf_assets_path,
                tokenizer_metadata=tokenizer_metadata,
            )
        }

    def _write_dcp_checkpoint(self, out_dir: Path, step: int) -> Path:
        step_dir = train._checkpoint_dir(out_dir) / f"step-{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / ".metadata").write_bytes(b"metadata")
        (step_dir / "__0_0.distcp").write_bytes(b"dcp-payload")
        return step_dir

    def _write_first_step_checkpoint_validation_report(self, out_dir: Path) -> dict:
        checkpoint = train._validate_dcp_checkpoint_step(
            self._write_dcp_checkpoint(out_dir, step=1)
        )
        report = {
            "schema_version": train.FIRST_STEP_CHECKPOINT_VALIDATION_SCHEMA_VERSION,
            "created_at_unix": 1.0,
            "step": 1,
            "checkpoint": checkpoint,
        }
        train._first_step_checkpoint_validation_path(out_dir).write_text(
            json.dumps(report, indent=2)
        )
        return report

    def _write_final_export(
        self,
        out_dir: Path,
        step: int,
        *,
        legacy: bool = False,
    ) -> Path:
        root = (
            train._checkpoint_dir(out_dir)
            if legacy
            else train._final_model_export_dir(out_dir)
        )
        step_dir = root / f"step-{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "model-00001-of-00002.safetensors").write_bytes(b"shard-1")
        (step_dir / "model-00002-of-00002.safetensors").write_bytes(b"shard-2")
        (step_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": len(b"shard-1") + len(b"shard-2")},
                    "weight_map": {
                        "lm_head.weight": "model-00002-of-00002.safetensors",
                        "model.embed_tokens.weight": (
                            "model-00001-of-00002.safetensors"
                        ),
                    },
                }
            )
        )
        return step_dir

    def _validate_launch_args(self, extra_args: list[str]):
        args = train.parse_args(
            [
                "--buckets",
                "256",
                "--bucket-cp",
                "256:1",
                "--max-length",
                "256",
                *extra_args,
            ]
        )
        args.buckets = ",".join(str(b) for b in train.parse_bucket_list(args.buckets))
        buckets = train.parse_bucket_list(args.buckets)
        bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)
        args.bucket_cp = train._format_bucket_cp_map(bucket_cp)
        train.validate_launch_inputs(args, buckets=buckets, bucket_cp=bucket_cp)
        train.validate_bucket_config(
            buckets=buckets,
            bucket_cp=bucket_cp,
            nproc_per_node=args.nproc_per_node,
            attention_backend=args.attention_backend,
        )
        return args

    def _validate_default_production_launch_args(
        self,
        extra_args: list[str] | None = None,
    ):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                train,
                "_detected_workspace_root",
                return_value=train.CANONICAL_WORKSPACE_ROOT,
            ),
            patch.object(
                train,
                "git_state_for_workspace",
                return_value=self._git_state(),
            ),
        ):
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf" / "Qwen2.5-Coder-7B-Instruct"),
                    "--production-mode",
                    "--enable-wandb",
                    *(extra_args or []),
                ]
            )
            args.buckets = ",".join(
                str(b) for b in train.parse_bucket_list(args.buckets)
            )
            buckets = train.parse_bucket_list(args.buckets)
            bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)
            args.bucket_cp = train._format_bucket_cp_map(bucket_cp)
            train.validate_launch_inputs(args, buckets=buckets, bucket_cp=bucket_cp)
            train.validate_bucket_config(
                buckets=buckets,
                bucket_cp=bucket_cp,
                nproc_per_node=args.nproc_per_node,
                attention_backend=args.attention_backend,
            )
            return args

    def test_defaults_track_paper_hyperparameters_and_target_pod(self):
        args = train.parse_args([])
        buckets = train.parse_bucket_list(args.buckets)
        bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)
        train.validate_launch_inputs(args, buckets=buckets, bucket_cp=bucket_cp)
        train.validate_bucket_config(
            buckets=buckets,
            bucket_cp=bucket_cp,
            nproc_per_node=args.nproc_per_node,
            attention_backend=args.attention_backend,
        )

        assert args.model_id == train.MODEL_ID
        assert args.model_revision == train.MODEL_REVISION
        assert args.dataset_id == train.DATASET_ID
        assert args.dataset_path == train.DEFAULT_DATASET_PATH
        assert args.source_dataset_id == train.SOURCE_DATASET_ID
        assert args.source_dataset_revision == train.SOURCE_DATASET_REVISION
        assert args.num_examples == 0
        assert args.max_streamed_examples == 0
        assert args.build_dataset_if_missing
        assert args.max_length == train.PAPER_CONTEXT_LENGTH
        assert args.num_train_epochs == 3.0
        assert args.global_batch_size == 32
        assert args.learning_rate == 1e-5
        assert args.min_learning_rate == 1e-8
        assert args.warmup_ratio == 0.1
        assert args.nproc_per_node == 8
        assert args.nnodes == 1
        assert args.node_rank == 0
        assert args.rdzv_backend == "c10d"
        assert args.rdzv_endpoint == "localhost:0"
        assert args.rdzv_id == ""
        assert args.enable_fp8
        assert args.attention_backend == "sdpa"
        assert args.optimizer_impl == "foreach"
        assert args.training_dtype == "float32"
        assert args.mixed_precision_param_dtype == "bfloat16"
        assert args.mixed_precision_reduce_dtype == "bfloat16"
        assert args.fsdp_reshard_after_forward == "never"
        assert not (args.production_mode)
        assert not (args.production_acceptance_smoke)
        assert not (args.detect_anomaly)
        assert args.min_free_disk_gb == train.DEFAULT_MIN_FREE_DISK_GB
        assert args.min_free_gpu_memory_gb == train.DEFAULT_MIN_FREE_GPU_MEMORY_GB
        assert args.min_free_cpu_memory_gb == train.DEFAULT_MIN_FREE_CPU_MEMORY_GB
        assert args.min_write_throughput_mb_s == train.DEFAULT_MIN_WRITE_THROUGHPUT_MB_S
        assert args.write_throughput_probe_mb == train.DEFAULT_WRITE_THROUGHPUT_PROBE_MB
        assert args.cuda_device_max_connections == "1"
        assert args.torch_nccl_async_error_handling == "1"
        assert args.bucket_curriculum == train.DEFAULT_BUCKET_CURRICULUM
        assert not (args.enable_profiler)
        assert args.profiler_freq == 10
        assert args.profiler_active == 1
        assert args.profiler_warmup == 3
        assert not (args.enable_memory_snapshot)
        assert train.parse_bucket_list(args.buckets) == train.DEFAULT_BUCKETS
        assert train.DEFAULT_BUCKETS == (32768, 65536, 131072)
        assert 8192 not in train.DEFAULT_BUCKETS
        assert 16384 not in train.DEFAULT_BUCKETS
        assert train.choose_bucket(8192, train.DEFAULT_BUCKETS) == 32768
        assert train.choose_bucket(16384, train.DEFAULT_BUCKETS) == 32768
        assert args.validate_first_step_checkpoint
        assert args.workspace_root == train._detected_workspace_root()

    def test_default_training_recipe_is_loaded_from_preset(self):
        default_args = train.parse_args([])
        explicit_args = train.parse_args([f"@{train.DEFAULT_TRAINING_PRESET}"])

        for field in (
            "model_id",
            "model_revision",
            "dataset_id",
            "source_dataset_revision",
            "max_length",
            "buckets",
            "bucket_cp",
            "num_train_epochs",
            "global_batch_size",
            "learning_rate",
            "min_learning_rate",
            "warmup_ratio",
            "enable_fp8",
            "compile",
            "checkpoint_interval",
            "checkpoint_async_mode",
            "nproc_per_node",
        ):
            assert getattr(default_args, field) == getattr(explicit_args, field)

    def test_production_mode_accepts_full_default_training_recipe(self):
        args = self._validate_default_production_launch_args()

        assert args.production_mode
        assert args.num_examples == 0
        assert args.max_streamed_examples == 0
        assert args.max_steps == 0
        assert args.max_length == train.PAPER_CONTEXT_LENGTH
        assert args.validate_first_step_checkpoint
        assert args.workspace_root == train.CANONICAL_WORKSPACE_ROOT
        assert args.enable_wandb
        assert not (args.production_acceptance_smoke)

    def test_production_acceptance_smoke_allows_bounded_real_subset(self):
        args = self._validate_default_production_launch_args(
            [
                "--production-acceptance-smoke",
                "--num-examples",
                "8",
                "--max-streamed-examples",
                "64",
                "--max-length",
                "32768",
                "--long-example-policy",
                "skip",
                "--buckets",
                "32768",
                "--bucket-cp",
                "32768:2",
                "--bucket-curriculum",
                "single-bucket",
                "--num-train-epochs",
                "1",
                "--max-steps",
                "1",
                "--global-batch-size",
                "8",
                "--no-compile",
                "--no-enable-fp8",
            ]
        )

        assert args.production_mode
        assert args.production_acceptance_smoke
        assert args.num_examples == 8
        assert args.max_steps == 1
        assert args.bucket_curriculum == "single-bucket"

    def test_production_acceptance_smoke_requires_production_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf" / "Qwen2.5-Coder-7B-Instruct"),
                    "--production-acceptance-smoke",
                    "--num-examples",
                    "8",
                    "--max-streamed-examples",
                    "64",
                ]
            )
            with pytest.raises(ValueError, match="--production-mode"):
                train.validate_launch_inputs(
                    args,
                    buckets=train.parse_bucket_list(args.buckets),
                    bucket_cp=train.parse_bucket_cp_map(args.bucket_cp),
                )

    def test_production_mode_requires_clean_git_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf" / "Qwen2.5-Coder-7B-Instruct"),
                    "--production-mode",
                ]
            )
            buckets = train.parse_bucket_list(args.buckets)
            bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)

            with (
                patch.object(
                    train,
                    "_detected_workspace_root",
                    return_value=train.CANONICAL_WORKSPACE_ROOT,
                ),
                patch.object(
                    train,
                    "git_state_for_workspace",
                    return_value=self._git_state(
                        dirty=True,
                        status_short=" M scripts/qwen_swehero_train.py",
                    ),
                ),
                pytest.raises(ValueError, match="clean Git worktree"),
            ):
                train.validate_launch_inputs(
                    args,
                    buckets=buckets,
                    bucket_cp=bucket_cp,
                )

    def test_production_mode_requires_durable_metrics_backend(self):
        cases = [
            ([], "--enable-wandb"),
            (["--enable-wandb", "--wandb-mode", "offline"], "not durable"),
            (["--enable-wandb", "--wandb-mode", "disabled"], "not durable"),
        ]
        for extra_args, message in cases:
            with (
                contextlib.nullcontext(),
                tempfile.TemporaryDirectory() as tmp,
            ):
                args = train.parse_args(
                    [
                        "--out-dir",
                        str(Path(tmp) / "run"),
                        "--dataset-path",
                        str(Path(tmp) / "dataset"),
                        "--hf-assets-path",
                        str(Path(tmp) / "hf" / "Qwen2.5-Coder-7B-Instruct"),
                        "--production-mode",
                        *extra_args,
                    ]
                )
                buckets = train.parse_bucket_list(args.buckets)
                bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)

                with (
                    patch.object(
                        train,
                        "_detected_workspace_root",
                        return_value=train.CANONICAL_WORKSPACE_ROOT,
                    ),
                    patch.object(
                        train,
                        "git_state_for_workspace",
                        return_value=self._git_state(),
                    ),
                    pytest.raises(ValueError, match=message),
                ):
                    train.validate_launch_inputs(
                        args,
                        buckets=buckets,
                        bucket_cp=bucket_cp,
                    )

    def test_production_mode_requires_available_git_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf" / "Qwen2.5-Coder-7B-Instruct"),
                    "--production-mode",
                ]
            )
            buckets = train.parse_bucket_list(args.buckets)
            bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)

            with (
                patch.object(
                    train,
                    "_detected_workspace_root",
                    return_value=train.CANONICAL_WORKSPACE_ROOT,
                ),
                patch.object(
                    train,
                    "git_state_for_workspace",
                    return_value=self._git_state(available=False),
                ),
                pytest.raises(ValueError, match="requires Git metadata"),
            ):
                train.validate_launch_inputs(
                    args,
                    buckets=buckets,
                    bucket_cp=bucket_cp,
                )

    def test_production_mode_requires_canonical_workspace_root(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                train,
                "_detected_workspace_root",
                return_value=Path(tmp) / "repo",
            ),
        ):
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf" / "Qwen2.5-Coder-7B-Instruct"),
                    "--production-mode",
                ]
            )
            buckets = train.parse_bucket_list(args.buckets)
            bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)
            with pytest.raises(ValueError, match="canonical workspace root"):
                train.validate_launch_inputs(args, buckets=buckets, bucket_cp=bucket_cp)

    def test_detected_workspace_root_prefers_canonical_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            physical_root = Path(tmp) / "home" / "jaxels-work-trial"
            script_path = physical_root / "scripts" / "qwen_swehero_train.py"
            script_path.parent.mkdir(parents=True)
            script_path.write_text("# launcher\n")
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            canonical_root = workspace_dir / "jaxels-work-trial"
            canonical_root.symlink_to(physical_root, target_is_directory=True)

            with (
                patch.object(train, "__file__", str(script_path)),
                patch.object(train, "CANONICAL_WORKSPACE_ROOT", canonical_root),
            ):
                detected = train._detected_workspace_root()

        assert detected == canonical_root

    def test_launch_lock_rejects_duplicate_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(["--out-dir", str(Path(tmp) / "run")])
            lock_path = train._launch_lock_path(args.out_dir)

            with train.launch_lock(args):
                assert lock_path.is_file()
                payload = json.loads(lock_path.read_text())
                assert payload["schema_version"] == train.LAUNCH_LOCK_SCHEMA_VERSION
                assert payload["out_dir"] == str(args.out_dir)
                with pytest.raises(RuntimeError, match="Launch lock already exists"):
                    with train.launch_lock(args):
                        pass

            assert not (lock_path.exists())

    def test_launch_lock_rejects_duplicate_overwrite_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            args = train.parse_args(["--out-dir", str(out_dir)])
            overwrite_args = train.parse_args(
                ["--out-dir", str(out_dir), "--overwrite-output"]
            )
            lock_path = train._launch_lock_path(args.out_dir)

            with train.launch_lock(args):
                with pytest.raises(RuntimeError, match="Launch lock already exists"):
                    with train.launch_lock(overwrite_args):
                        pass
                assert lock_path.is_file()

            assert not (lock_path.exists())

    def test_launch_lock_rejects_malformed_existing_lock_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(["--out-dir", str(Path(tmp) / "run")])
            lock_path = train._launch_lock_path(args.out_dir)
            lock_path.write_text("not json")

            with pytest.raises(RuntimeError, match="existing lock could not be parsed"):
                with train.launch_lock(args):
                    pass

            assert lock_path.read_text() == "not json"

    def test_launch_lock_error_reports_existing_lock_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(["--out-dir", str(Path(tmp) / "run")])
            lock_path = train._launch_lock_path(args.out_dir)
            lock_path.write_text(
                json.dumps(
                    {
                        "pid": 12345,
                        "hostname": "worker-a",
                        "created_at_unix": 100.5,
                    }
                )
            )

            with pytest.raises(
                RuntimeError,
                match="pid=12345.*hostname='worker-a'.*created_at_unix=100.5",
            ):
                with train.launch_lock(args):
                    pass

    def test_launch_lock_is_sidecar_to_survive_overwrite_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            args = train.parse_args(["--out-dir", str(out_dir), "--overwrite-output"])

            lock_path = train._launch_lock_path(args.out_dir)

        assert lock_path == out_dir.with_name("run.launch.lock")
        assert lock_path.parent != out_dir

    def test_production_mode_rejects_smoke_and_subset_controls(self):
        cases = [
            (["--dry-run"], "--dry-run"),
            (["--prepare-data-only"], "--prepare-data-only"),
            (["--skip-data-prep"], "--skip-data-prep"),
            (["--smoke-synthetic-buckets"], "--smoke-synthetic-buckets"),
            (["--num-examples", "1"], "--num-examples=0"),
            (["--max-streamed-examples", "10"], "--max-streamed-examples=0"),
            (["--max-steps", "1"], "--max-steps=0"),
            (
                ["--no-validate-first-step-checkpoint"],
                "--validate-first-step-checkpoint=True",
            ),
        ]
        for extra_args, message in cases:
            with contextlib.nullcontext():
                with pytest.raises(ValueError, match=message):
                    self._validate_default_production_launch_args(extra_args)

    def test_production_mode_rejects_non_production_training_recipe(self):
        cases = [
            (["--max-length", "32768"], "--max-length=131072"),
            (["--buckets", "131072", "--bucket-cp", "131072:8"], "--buckets"),
            (
                [
                    "--bucket-cp",
                    "32768:1,65536:4,131072:8",
                ],
                "--bucket-cp",
            ),
            (["--bucket-curriculum", "long-to-short"], "--bucket-curriculum"),
            (["--num-train-epochs", "1"], "--num-train-epochs=3.0"),
            (["--global-batch-size", "8"], "--global-batch-size=32"),
            (["--local-batch-size", "2"], "--local-batch-size=1"),
            (["--learning-rate", "2e-5"], "--learning-rate=1e-05"),
            (["--min-learning-rate", "0"], "--min-learning-rate=1e-08"),
            (["--warmup-ratio", "0"], "--warmup-ratio=0.1"),
            (["--weight-decay", "0.1"], "--weight-decay=0.0"),
            (["--long-example-policy", "skip"], "--long-example-policy='error'"),
            (["--include-model-patch"], "--include-model-patch=False"),
            (["--min-trainable-tokens", "2"], "--min-trainable-tokens=1"),
            (["--source-dataset-revision", "main"], "--source-dataset-revision"),
        ]
        for extra_args, message in cases:
            with contextlib.nullcontext():
                with pytest.raises(ValueError, match=message):
                    self._validate_default_production_launch_args(extra_args)

    def test_launch_env_file_is_secret_only_not_experiment_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "NUM_EXAMPLES=7",
                        "MAX_STREAMED_EXAMPLES=11",
                        "export SWEHERO_BUCKETS=1024",
                        "SWEHERO_BUCKET_CP=1024:1",
                        "ENABLE_FP8=0 # disable for test",
                        "WANDB_RUN_NAME='dotenv-run'",
                        "HF_TOKEN='secret-token'",
                    ]
                )
            )
            argv = ["--env-file", str(env_file)]

            with patch.dict(os.environ, {}, clear=True):
                loaded_env_file = train.load_launch_env_file(argv)
                args = train.parse_args(argv, env_file_default=loaded_env_file)
                hf_token = os.environ["HF_TOKEN"]

        assert args.env_file == str(env_file)
        assert args.num_examples == train.DEFAULT_NUM_EXAMPLES
        assert args.max_streamed_examples == train.DEFAULT_MAX_STREAMED_EXAMPLES
        assert train.parse_bucket_list(args.buckets) == train.DEFAULT_BUCKETS
        assert args.enable_fp8
        assert args.wandb_run_name == "qwen25-coder7b-swehero-tt"
        assert hf_token == "secret-token"

    def test_cli_flags_override_launch_env_file_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("HF_TOKEN=secret\n")
            argv = [
                "--env-file",
                str(env_file),
                "--num-examples",
                "3",
                "--enable-fp8",
            ]

            with patch.dict(os.environ, {}, clear=True):
                loaded_env_file = train.load_launch_env_file(argv)
                args = train.parse_args(argv, env_file_default=loaded_env_file)

        assert args.num_examples == 3
        assert args.enable_fp8

    def test_launch_argfile_supports_comments_quoting_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "configured run"
            argfile = Path(tmp) / "launch.args"
            argfile.write_text(
                "\n".join(
                    [
                        "# reviewed production launch flags",
                        f"--out-dir {shlex.quote(str(out_dir))}",
                        "--buckets 1024",
                        "--bucket-cp 1024:1",
                        "--max-length 1024",
                        "--num-examples 4",
                        "--no-enable-fp8",
                    ]
                )
            )

            args = train.parse_args([f"@{argfile}", "--num-examples", "7"])

        assert args.out_dir == out_dir
        assert args.buckets == "1024"
        assert args.bucket_cp == "1024:1"
        assert args.max_length == 1024
        assert args.num_examples == 7
        assert not (args.enable_fp8)

    def test_process_env_does_not_override_preset_or_cli_settings(self):
        with patch.dict(
            os.environ,
            {
                "NUM_EXAMPLES": "13",
                "ENABLE_FP8": "0",
                "SWEHERO_BUCKETS": "1024",
            },
            clear=True,
        ):
            args = train.parse_args(["--num-examples", "3"])

        assert args.num_examples == 3
        assert train.parse_bucket_list(args.buckets) == train.DEFAULT_BUCKETS
        assert args.enable_fp8

    def test_argfile_values_are_validated_by_argparse_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            argfile = Path(tmp) / "bad.args"
            argfile.write_text("--num-examples abc\n")
            with pytest.raises(SystemExit):
                train.parse_args([f"@{argfile}"])

    def test_launch_input_validation_rejects_bad_numeric_values(self):
        cases = [
            (
                ["--model-id", "Qwen/Qwen2.5-Coder-14B-Instruct"],
                "--model-id must be",
            ),
            (["--model-revision", "main"], "--model-revision must be an exact"),
            (["--model-revision", "0" * 40], "--model-revision must be the pinned"),
            (
                ["--source-dataset-rows-per-shard", "0"],
                "--source-dataset-rows-per-shard",
            ),
            (
                ["--source-dataset-build-batch-size", "0"],
                "--source-dataset-build-batch-size",
            ),
            (["--num-examples", "-1"], "--num-examples"),
            (["--max-streamed-examples", "-1"], "--max-streamed-examples"),
            (["--shuffle-buffer", "-1"], "--shuffle-buffer"),
            (["--seed", "-1"], "--seed"),
            (["--max-length", "0"], "--max-length"),
            (["--min-trainable-tokens", "0"], "--min-trainable-tokens"),
            (["--num-train-epochs", "0"], "--num-train-epochs"),
            (["--max-steps", "-1"], "--max-steps"),
            (["--global-batch-size", "0"], "--global-batch-size"),
            (["--local-batch-size", "0"], "--local-batch-size"),
            (["--learning-rate", "0"], "--learning-rate"),
            (["--min-learning-rate", "-1"], "--min-learning-rate"),
            (["--warmup-ratio", "-0.1"], "--warmup-ratio"),
            (["--warmup-ratio", "1.1"], "--warmup-ratio"),
            (["--weight-decay", "-0.1"], "--weight-decay"),
            (["--max-grad-norm", "0"], "--max-grad-norm"),
            (["--cuda-device-max-connections", "0"], "--cuda-device-max-connections"),
            (
                ["--torch-nccl-async-error-handling", ""],
                "--torch-nccl-async-error-handling",
            ),
            (["--chunked-ce-chunks", "0"], "--chunked-ce-chunks"),
            (["--checkpoint-interval", "0"], "--checkpoint-interval"),
            (["--metrics-log-freq", "0"], "--metrics-log-freq"),
            (["--min-free-disk-gb", "-1"], "--min-free-disk-gb"),
            (["--min-free-gpu-memory-gb", "-1"], "--min-free-gpu-memory-gb"),
            (["--min-free-cpu-memory-gb", "-1"], "--min-free-cpu-memory-gb"),
            (
                ["--min-write-throughput-mb-s", "-1"],
                "--min-write-throughput-mb-s",
            ),
            (["--write-throughput-probe-mb", "0"], "--write-throughput-probe-mb"),
            (["--profiler-freq", "0"], "--profiler-freq"),
            (["--profiler-active", "0"], "--profiler-active"),
            (["--profiler-warmup", "-1"], "--profiler-warmup"),
            (["--profiler-repeat", "0"], "--profiler-repeat"),
            (["--profiler-skip-first", "-1"], "--profiler-skip-first"),
            (
                ["--profiler-skip-first-wait", "-1"],
                "--profiler-skip-first-wait",
            ),
            (
                [
                    "--profiler-freq",
                    "3",
                    "--profiler-active",
                    "1",
                    "--profiler-warmup",
                    "3",
                ],
                "--profiler-freq must be greater",
            ),
            (["--nproc-per-node", "0"], "--nproc-per-node"),
            (["--nnodes", "0"], "--nnodes"),
            (["--node-rank", "-1"], "--node-rank"),
            (["--node-rank", "1"], "--node-rank must be 0"),
            (
                [
                    "--nnodes",
                    "2",
                    "--node-rank",
                    "2",
                    "--rdzv-endpoint",
                    "train-master:29400",
                    "--rdzv-id",
                    "run",
                ],
                "--node-rank must be less than --nnodes",
            ),
            (["--nnodes", "2"], "--rdzv-id is required"),
            (
                ["--nnodes", "2", "--rdzv-id", "run"],
                "--rdzv-endpoint must be a stable host:port",
            ),
            (
                ["--smoke-synthetic-examples-per-bucket", "0"],
                "--smoke-synthetic-examples-per-bucket",
            ),
            (
                ["--smoke-synthetic-examples-per-bucket", "2"],
                "--smoke-synthetic-examples-per-bucket only applies",
            ),
            (
                ["--smoke-synthetic-buckets", "--num-examples", "1"],
                "--smoke-synthetic-buckets cannot be combined with --num-examples",
            ),
            (
                ["--smoke-synthetic-buckets", "--max-streamed-examples", "1"],
                "--smoke-synthetic-buckets cannot be combined with --max-streamed-examples",
            ),
            (["--learning-rate", "nan"], "--learning-rate must be finite"),
            (["--min-learning-rate", "inf"], "--min-learning-rate must be finite"),
            (
                ["--learning-rate", "1e-5", "--min-learning-rate", "1e-4"],
                "--min-learning-rate cannot exceed --learning-rate",
            ),
            (["--global-batch-size", "10"], "--global-batch-size must be divisible"),
            (["--log-rank", "rank0"], "--log-rank contains a non-integer rank"),
            (
                ["--torchrun-log-rank-filter", "-1"],
                "--torchrun-log-rank-filter ranks must be non-negative",
            ),
        ]
        for cli_args, message in cases:
            with contextlib.nullcontext():
                with pytest.raises(ValueError, match=message):
                    self._validate_launch_args(cli_args)

    def test_launch_input_validation_rejects_context_and_bucket_mismatches(self):
        with pytest.raises(ValueError, match="paper context"):
            self._validate_launch_args(
                [
                    "--buckets",
                    str(train.PAPER_CONTEXT_LENGTH * 2),
                    "--bucket-cp",
                    f"{train.PAPER_CONTEXT_LENGTH * 2}:1",
                    "--max-length",
                    str(train.PAPER_CONTEXT_LENGTH + 1),
                ]
            )
        with pytest.raises(ValueError, match="largest bucket"):
            self._validate_launch_args(["--max-length", "512"])
        with pytest.raises(ValueError, match="not present in --buckets"):
            self._validate_launch_args(["--bucket-cp", "256:1,512:1"])
        with pytest.raises(ValueError, match="single-bucket"):
            self._validate_launch_args(
                [
                    "--buckets",
                    "128,256",
                    "--bucket-cp",
                    "128:1,256:1",
                    "--bucket-curriculum",
                    "single-bucket",
                ]
            )

    def test_bucket_parsers_reject_malformed_or_ambiguous_values(self):
        with pytest.raises(ValueError, match="invalid bucket size"):
            train.parse_bucket_list("256,abc")
        with pytest.raises(ValueError, match="duplicate bucket"):
            train.parse_bucket_list("256,256")
        with pytest.raises(ValueError, match="invalid bucket CP map entry"):
            train.parse_bucket_cp_map("256:not-a-cp")
        with pytest.raises(ValueError, match="duplicate bucket"):
            train.parse_bucket_cp_map("256:1,256:2")

    def test_wandb_identity_generates_run_id_and_env_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            args = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--enable-wandb",
                    "--wandb-project",
                    "proj",
                    "--wandb-entity",
                    "team",
                    "--wandb-run-name",
                    "run-name",
                    "--wandb-run-group",
                    "group-a",
                    "--wandb-run-job-type",
                    "train",
                    "--wandb-run-tags",
                    "direct-to-hero,smoke",
                    "--wandb-run-notes",
                    "notes",
                    "--wandb-mode",
                    "offline",
                ]
            )
            out_dir.mkdir(parents=True)
            identity = train.resolve_wandb_identity(args, resume_state=None)
            stage = train.BucketStage(
                bucket=256,
                cp_degree=1,
                example_count=1,
                steps=1,
                cumulative_steps=1,
                bucket_file=out_dir / "data" / "bucket_256.jsonl",
            )
            env = train.build_stage_env(
                args,
                stage=stage,
                total_steps=1,
                warmup_steps=0,
                pad_token_id=0,
            )
            persisted = json.loads(
                (out_dir / train.WANDB_IDENTITY_FILENAME).read_text()
            )

        assert identity is not None
        assert identity["run_id"].startswith("swehero-")
        assert len(identity["run_id"]) <= 64
        assert identity == persisted
        assert identity["resume"] == "allow"
        assert env["SWEHERO_ENABLE_WANDB"] == "1"
        assert env["WANDB_PROJECT"] == "proj"
        assert env["WANDB_TEAM"] == "team"
        assert env["WANDB_ENTITY"] == "team"
        assert env["WANDB_RUN_NAME"] == "run-name"
        assert env["WANDB_NAME"] == "run-name"
        assert env["WANDB_RUN_ID"] == identity["run_id"]
        assert env["WANDB_RESUME"] == "allow"
        assert env["WANDB_RUN_GROUP"] == "group-a"
        assert env["WANDB_RUN_JOB_TYPE"] == "train"
        assert env["WANDB_JOB_TYPE"] == "train"
        assert env["WANDB_RUN_TAGS"] == "direct-to-hero,smoke"
        assert env["WANDB_TAGS"] == "direct-to-hero,smoke"
        assert env["WANDB_RUN_NOTES"] == "notes"
        assert env["WANDB_NOTES"] == "notes"
        assert env["WANDB_MODE"] == "offline"
        assert "WANDB_API_KEY" not in identity["env"]

    def test_wandb_identity_reuses_run_id_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            args = train.parse_args(["--out-dir", str(out_dir), "--enable-wandb"])
            out_dir.mkdir(parents=True)
            train.resolve_wandb_identity(args, resume_state=None)
            run_id = args.wandb_run_id

            resume_args = train.parse_args(
                ["--out-dir", str(out_dir), "--enable-wandb", "--resume"]
            )
            resume_state = train.ResumeCheckpointState(
                checkpoint_dir=train._checkpoint_dir(out_dir),
                final_export_dir=train._final_model_export_dir(out_dir),
                latest_resumable_step=1,
                latest_model_export_step=None,
                latest_any_step=1,
            )
            identity = train.resolve_wandb_identity(
                resume_args,
                resume_state=resume_state,
            )

        assert resume_args.wandb_run_id == run_id
        assert identity["run_id"] == run_id
        assert identity["resume"] == "allow"

    def test_wandb_identity_rejects_changed_existing_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            args = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--enable-wandb",
                    "--wandb-run-id",
                    "original-run",
                ]
            )
            out_dir.mkdir(parents=True)
            train.resolve_wandb_identity(args, resume_state=None)
            changed = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--enable-wandb",
                    "--wandb-run-id",
                    "different-run",
                ]
            )

            with pytest.raises(RuntimeError, match="W&B identity"):
                train.resolve_wandb_identity(changed, resume_state=None)

    def test_wandb_resume_controls_reject_conflicts_and_bad_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            out_dir.mkdir(parents=True)
            conflicting = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--enable-wandb",
                    "--wandb-resume",
                    "allow",
                    "--wandb-resume-from",
                    "abc123?_step=10",
                ]
            )
            bad_run_id = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--enable-wandb",
                    "--wandb-run-id",
                    "bad/id",
                ]
            )

            with pytest.raises(ValueError, match="cannot be combined"):
                train.resolve_wandb_identity(conflicting, resume_state=None)
            with pytest.raises(ValueError, match="forbids"):
                train.resolve_wandb_identity(bad_run_id, resume_state=None)

    def test_explicit_launch_env_file_must_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.env"
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(FileNotFoundError, match="Requested env file"):
                    train.load_launch_env_file(["--env-file", str(missing)])

    def test_default_launch_env_file_is_optional(self):
        with patch.dict(os.environ, {}, clear=True):
            loaded_env_file = train.load_launch_env_file([])

        assert loaded_env_file == train.DEFAULT_ENV_FILE

    def test_expected_qwen_yarn_rope_config_tracks_128k_extension(self):
        rope = train.expected_qwen_yarn_rope_config()

        assert rope["rope_type"] == "yarn"
        assert rope["max_position_embeddings"] == train.PAPER_CONTEXT_LENGTH
        assert (
            rope["original_max_position_embeddings"] == train.QWEN_NATIVE_CONTEXT_LENGTH
        )
        assert rope["factor"] == 4.0
        assert rope["rope_theta"] == 1_000_000.0
        assert rope["beta_fast"] == 32.0
        assert rope["beta_slow"] == 1.0

    def test_torchtitan_qwen_registry_uses_standard_yarn_beta_names(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (
            repo_root / "torchtitan/torchtitan/models/qwen2_5/__init__.py"
        ).read_text()

        assert "max_seq_len=QWEN25_CODER_7B_CONTEXT" in source
        assert "theta=1_000_000.0" in source
        assert 'scaling="yarn"' in source
        assert "rope_factor=QWEN25_CODER_7B_CONTEXT / QWEN25_NATIVE_CONTEXT" in source
        assert "beta_fast=32.0" in source
        assert "beta_slow=1.0" in source
        assert "original_seq_len=QWEN25_NATIVE_CONTEXT" in source

    def test_model_asset_provenance_records_complete_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf_assets = Path(tmp) / "hf"
            hf_assets.mkdir()
            config = {
                "_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 3584,
                "max_position_embeddings": 32768,
                "model_type": "qwen2",
                "num_hidden_layers": 28,
                "rope_scaling": None,
                "torch_dtype": "bfloat16",
                "vocab_size": 152064,
            }
            (hf_assets / "config.json").write_text(json.dumps(config))
            (hf_assets / "generation_config.json").write_text(
                json.dumps({"eos_token_id": 151645, "pad_token_id": 151643})
            )
            (hf_assets / "tokenizer.json").write_text('{"tokenizer": true}')
            (hf_assets / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": "template"})
            )
            (hf_assets / "model-00001-of-00002.safetensors").write_bytes(b"shard-1")
            (hf_assets / "model-00002-of-00002.safetensors").write_bytes(b"shard-2")
            (hf_assets / "orphan.safetensors").write_bytes(b"orphan")
            (hf_assets / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 13},
                        "weight_map": {
                            "lm_head.weight": "model-00002-of-00002.safetensors",
                            "model.embed_tokens.weight": (
                                "model-00001-of-00002.safetensors"
                            ),
                        },
                    }
                )
            )
            tokenizer_metadata = {
                "hf_assets_path": str(hf_assets),
                "tokenizer_json_sha256": train._hash_file(hf_assets / "tokenizer.json"),
                "tokenizer_config_sha256": train._hash_file(
                    hf_assets / "tokenizer_config.json"
                ),
            }

            provenance = train._model_asset_provenance(
                model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                model_revision=train.MODEL_REVISION,
                hf_assets_path=hf_assets,
                tokenizer_metadata=tokenizer_metadata,
            )

        files = {record["path"]: record for record in provenance["files"]}
        assert (
            provenance["schema_version"] == train.MODEL_ASSET_PROVENANCE_SCHEMA_VERSION
        )
        assert provenance["model_revision"] == train.MODEL_REVISION
        assert provenance["file_count"] == len(files)
        assert provenance["total_bytes"] == sum(
            record["bytes"] for record in files.values()
        )
        assert files["config.json"]["kind"] == "model_config"
        assert files["model-00001-of-00002.safetensors"][
            "sha256"
        ] == train._sha256_text("shard-1")
        assert provenance["config"]["summary"]["model_type"] == "qwen2"
        assert provenance["config"]["summary"]["hidden_size"] == 3584
        assert provenance["generation_config"]["summary"]["pad_token_id"] == 151643
        assert provenance["safetensors"]["weight_map_entries"] == 2
        assert [
            record["path"] for record in provenance["safetensors"]["shard_files"]
        ] == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        assert provenance["safetensors"]["unindexed_safetensors_files"] == [
            "orphan.safetensors"
        ]
        assert provenance["tokenizer"] == tokenizer_metadata

    def test_hf_asset_preflight_requires_indexed_weight_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf_assets = Path(tmp) / "hf"
            self._write_preflight_hf_assets(hf_assets)
            shard = hf_assets / "model-00001-of-00001.safetensors"
            shard.unlink()
            args = train.parse_args(["--hf-assets-path", str(hf_assets)])

            with pytest.raises(RuntimeError, match="safetensors shard"):
                train.validate_hf_asset_preflight(args)

            shard.write_bytes(b"shard")
            summary = train.validate_hf_asset_preflight(args)

        assert summary["config_model_type"] == "qwen2"
        assert summary["safetensors"]["shard_count"] == 1
        assert summary["safetensors"]["weight_map_entries"] == 1

    def test_hf_asset_preflight_rejects_manifest_asset_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf_assets = Path(tmp) / "hf"
            self._write_preflight_hf_assets(hf_assets)
            args = train.parse_args(["--hf-assets-path", str(hf_assets)])
            manifest = self._model_assets_manifest(args)

            summary = train.validate_hf_asset_preflight(args, manifest)
            mismatched_manifest = json.loads(json.dumps(manifest))
            mismatched_manifest["model_assets"]["model_id"] = "other/model"
            with pytest.raises(RuntimeError, match="model_assets.model_id"):
                train.validate_hf_asset_preflight(args, mismatched_manifest)

            mismatched_manifest = json.loads(json.dumps(manifest))
            mismatched_manifest["model_assets"]["model_revision"] = "0" * 40
            with pytest.raises(RuntimeError, match="model_assets.model_revision"):
                train.validate_hf_asset_preflight(args, mismatched_manifest)

            (hf_assets / "model-00001-of-00001.safetensors").write_bytes(b"drift")
            with pytest.raises(RuntimeError, match="sha256"):
                train.validate_hf_asset_preflight(args, manifest)

            (hf_assets / "model-00001-of-00001.safetensors").write_bytes(
                b"changed-length"
            )
            with pytest.raises(RuntimeError, match="byte size"):
                train.validate_hf_asset_preflight(args, manifest)

        assert summary["manifest_model_assets"]["file_count"] == 5
        assert summary["manifest_model_assets"]["sha256_verified_files"] == 5

    def test_cuda_launch_summary_requires_visible_device_per_rank(self):
        class FakeCuda:
            def __init__(self, count):
                self.count = count

            def is_available(self):
                return self.count > 0

            def device_count(self):
                return self.count

            def get_device_name(self, index):
                return f"Fake GPU {index}"

            def get_device_capability(self, index):
                return (9, 0)

        fake_torch = types.SimpleNamespace(cuda=FakeCuda(1))

        with pytest.raises(RuntimeError, match="visible CUDA device"):
            train._cuda_launch_summary(fake_torch, nproc_per_node=2)

        summary = train._cuda_launch_summary(fake_torch, nproc_per_node=1)
        assert summary["available"]
        assert summary["device_count"] == 1
        assert summary["devices"][0]["capability"] == [9, 0]

    def test_nvidia_smi_metadata_parses_driver_and_cuda_version(self):
        def fake_command(command, *, timeout_seconds=5.0):
            if (
                "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free"
                in command
            ):
                return {
                    "command": command,
                    "available": True,
                    "returncode": 0,
                    "stdout": (
                        "0, NVIDIA H100 80GB HBM3, GPU-test, 570.195.03, 81559, 80000\n"
                    ),
                    "stderr": "",
                }
            return {
                "command": command,
                "available": True,
                "returncode": 0,
                "stdout": (
                    "| NVIDIA-SMI 570.195.03    Driver Version: 570.195.03"
                    "    CUDA Version: 12.8 |\n"
                ),
                "stderr": "",
            }

        with patch.object(train, "_run_metadata_command", side_effect=fake_command):
            metadata = train._nvidia_smi_metadata()

        assert metadata["cuda_version_from_banner"] == "12.8"
        assert metadata["gpus"][0]["index"] == "0"
        assert metadata["gpus"][0]["name"] == "NVIDIA H100 80GB HBM3"
        assert metadata["gpus"][0]["uuid"] == "GPU-test"
        assert metadata["gpus"][0]["driver_version"] == "570.195.03"
        assert metadata["gpus"][0]["memory_total_mib"] == "81559"
        assert metadata["gpus"][0]["memory_free_mib"] == "80000"

    def test_resource_preflights_reject_low_disk_cpu_gpu_and_write_throughput(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            with (
                patch.object(
                    train.shutil,
                    "disk_usage",
                    return_value=types.SimpleNamespace(
                        total=100,
                        used=90,
                        free=10,
                    ),
                ),
                pytest.raises(RuntimeError, match="Disk preflight failed"),
            ):
                train._disk_space_preflight(out_dir, min_free_gb=1)

            with (
                patch.object(train, "_available_cpu_memory_bytes", return_value=10),
                pytest.raises(RuntimeError, match="CPU memory preflight failed"),
            ):
                train._cpu_memory_preflight(min_free_gb=1)

            with (
                patch.object(
                    train,
                    "_nvidia_smi_metadata",
                    return_value={
                        "gpus": [
                            {
                                "index": "0",
                                "name": "H100",
                                "memory_free_mib": "1024",
                            }
                        ]
                    },
                ),
                pytest.raises(RuntimeError, match="GPU memory preflight failed"),
            ):
                train._gpu_memory_preflight(
                    min_free_gb=2,
                    required_gpus=1,
                )

            with pytest.raises(RuntimeError, match="Write-throughput preflight failed"):
                train._write_throughput_preflight(
                    out_dir,
                    min_mb_s=1e12,
                    probe_mb=1,
                )

    def test_write_runtime_metadata_records_environment_and_lockfiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(["--out-dir", str(Path(tmp) / "run")])
            args.out_dir.mkdir(parents=True)
            lockfile = Path(tmp) / "lock.txt"
            lockfile.write_text("locked\n")
            runtime = {
                "python": sys.executable,
                "torch": "2.x",
                "cuda": {"device_count": 8},
            }

            with (
                patch.object(train.time, "time", return_value=123.0),
                patch.object(
                    train,
                    "_runtime_lockfile_metadata",
                    return_value=[
                        {
                            "kind": "test_lock",
                            **train._file_metadata(lockfile),
                        }
                    ],
                ),
                patch.object(
                    train,
                    "_nvidia_smi_metadata",
                    return_value={
                        "gpus": [
                            {
                                "index": "0",
                                "driver_version": "570.195.03",
                            }
                        ],
                        "cuda_version_from_banner": "12.8",
                    },
                ),
                patch.object(
                    train,
                    "git_state_for_workspace",
                    return_value=self._git_state(),
                ),
                patch.dict(
                    os.environ,
                    {
                        "NCCL_DEBUG": "INFO",
                        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                        "UNRELATED": "ignored",
                    },
                    clear=True,
                ),
            ):
                metadata = train.write_runtime_metadata(args, runtime)

            persisted = json.loads(
                (args.out_dir / train.RUNTIME_METADATA_FILENAME).read_text()
            )

        assert metadata == persisted
        assert metadata["schema_version"] == train.RUNTIME_METADATA_SCHEMA_VERSION
        assert metadata["created_at_unix"] == 123.0
        assert metadata["runtime"] == runtime
        assert metadata["lockfiles"][0]["sha256"] == train._sha256_text("locked\n")
        assert metadata["hardware"]["nvidia_smi"]["cuda_version_from_banner"] == "12.8"
        assert metadata["environment"] == (
            {
                "NCCL_DEBUG": "INFO",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            }
        )
        assert metadata["workspace"]["configured_root"] == str(
            train._configured_workspace_root(args)
        )
        assert metadata["workspace"]["canonical_root"] == str(
            train.CANONICAL_WORKSPACE_ROOT
        )
        assert "cwd" in metadata["workspace"]
        assert metadata["git"] == self._git_state()

    def test_runtime_lockfile_metadata_uses_invoked_venv_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            venv_root = Path(tmp) / "venv"
            (repo_root / "requirements").mkdir(parents=True)
            (repo_root / "torchtitan" / ".ci" / "docker").mkdir(parents=True)
            (venv_root / "bin").mkdir(parents=True)
            (repo_root / "requirements" / "torchtitan-pod-cu128.lock").write_text(
                "lock\n"
            )
            (repo_root / "requirements" / "torchtitan-pod-cu128.txt").write_text(
                "requirements\n"
            )
            (
                repo_root / "torchtitan" / ".ci" / "docker" / "requirements.txt"
            ).write_text("torchtitan\n")
            runtime_json = venv_root / "torchtitan-swehero-runtime.json"
            runtime_json.write_text('{"runtime": true}\n')

            metadata = train._runtime_lockfile_metadata(
                repo_root=repo_root,
                python_executable=str(venv_root / "bin" / "python"),
            )

        by_kind = {record["kind"]: record for record in metadata}
        assert by_kind["venv_runtime_metadata"]["path"] == str(runtime_json)
        assert by_kind["venv_runtime_metadata"]["exists"]
        assert by_kind["venv_runtime_metadata"]["sha256"] == train._sha256_text(
            '{"runtime": true}\n'
        )

    def test_dataset_artifact_metadata_records_selection_and_shard_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            data_dir = dataset / "data"
            data_dir.mkdir(parents=True)
            metadata = {"rows": 2, "source_revision": "abc123"}
            (dataset / "metadata.json").write_text(json.dumps(metadata))
            (dataset / "selection_manifest.jsonl").write_text(
                '{"instance_id":"one"}\n{"instance_id":"two"}\n'
            )
            shard = data_dir / "train-00000-of-00001.parquet"
            shard.write_bytes(b"parquet bytes")

            artifact = train._dataset_artifact_metadata(dataset)

        assert artifact["path"] == str(dataset)
        assert artifact["metadata"] == metadata
        assert artifact["metadata_json"]["sha256"] == artifact["metadata_json_sha256"]
        assert (
            artifact["selection_manifest"]["sha256"]
            == artifact["selection_manifest_sha256"]
        )
        assert artifact["data_file_count"] == 1
        assert (
            artifact["data_files"][0]["relative_path"]
            == shard.relative_to(dataset).as_posix()
        )
        assert artifact["data_files"][0]["bytes"] == len(b"parquet bytes")
        assert artifact["data_files"][0]["sha256"] == train._sha256_text(
            "parquet bytes"
        )
        assert artifact["total_data_bytes"] == len(b"parquet bytes")

    def test_cos_sin_yarn_uses_huggingface_correction_range_order(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "torchtitan/torchtitan/models/common/rope.py").read_text()

        fast_index = source.index("cfg.beta_fast * 2 * math.pi")
        slow_index = source.index("cfg.beta_slow * 2 * math.pi")
        assert fast_index < slow_index

    def test_choose_bucket_ceilings(self):
        buckets = (8, 16, 32)

        assert train.choose_bucket(1, buckets) == 8
        assert train.choose_bucket(8, buckets) == 8
        assert train.choose_bucket(9, buckets) == 16
        assert train.choose_bucket(17, buckets) == 32
        with pytest.raises(ValueError):
            train.choose_bucket(33, buckets)

    def test_bucket_plan_uses_epochs_and_cumulative_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            bucket_files = {
                32768: Path(tmp) / "bucket_32768.jsonl",
                65536: Path(tmp) / "bucket_65536.jsonl",
            }
            plan = train.build_bucket_plan(
                bucket_counts={32768: 33, 65536: 1},
                bucket_files=bucket_files,
                bucket_cp={32768: 2, 65536: 4},
                epochs=3.0,
                global_batch_size=32,
                warmup_ratio=0.1,
            )

        assert plan.total_steps == 5
        assert plan.warmup_steps == 1
        assert [stage.bucket for stage in plan.stages] == [32768, 65536]
        assert [stage.steps for stage in plan.stages] == [4, 1]
        assert [stage.cumulative_steps for stage in plan.stages] == [4, 5]
        assert [stage.cp_degree for stage in plan.stages] == [2, 4]

    def test_bucket_plan_uses_explicit_curriculum_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            bucket_files = {
                32768: Path(tmp) / "bucket_32768.jsonl",
                65536: Path(tmp) / "bucket_65536.jsonl",
            }
            plan = train.build_bucket_plan(
                bucket_counts={32768: 33, 65536: 1},
                bucket_files=bucket_files,
                bucket_cp={32768: 2, 65536: 4},
                epochs=3.0,
                global_batch_size=32,
                warmup_ratio=0.1,
                bucket_curriculum="long-to-short",
            )

        assert [stage.bucket for stage in plan.stages] == [65536, 32768]
        assert [stage.steps for stage in plan.stages] == [1, 4]
        assert [stage.cumulative_steps for stage in plan.stages] == [1, 5]
        assert [stage.cp_degree for stage in plan.stages] == [4, 2]

    def test_single_bucket_curriculum_requires_one_non_empty_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            bucket_files = {
                32768: Path(tmp) / "bucket_32768.jsonl",
                65536: Path(tmp) / "bucket_65536.jsonl",
            }

            with pytest.raises(ValueError, match="single-bucket curriculum"):
                train.build_bucket_plan(
                    bucket_counts={32768: 33, 65536: 1},
                    bucket_files=bucket_files,
                    bucket_cp={32768: 2, 65536: 4},
                    epochs=3.0,
                    global_batch_size=32,
                    warmup_ratio=0.1,
                    bucket_curriculum="single-bucket",
                )

    def test_resume_requires_existing_artifact_then_full_dcp_for_incomplete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(["--out-dir", str(Path(tmp) / "run"), "--resume"])
            with pytest.raises(FileNotFoundError):
                train.validate_resume_request(args)

            args, _manifest, plan = self._resume_test_setup(tmp)
            export = train._final_model_export_dir(args.out_dir) / "step-3"
            export.mkdir(parents=True)
            (export / "model.safetensors.index.json").write_text("{}")
            args.resume = True

            resume_state = train.validate_resume_request(args)
            with pytest.raises(RuntimeError, match="full DCP checkpoint"):
                train.validate_resume_progress(plan, resume_state)

    def test_resume_rejects_destructive_refresh_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            (out_dir / "torchtitan" / "checkpoint" / "step-1").mkdir(parents=True)
            args = train.parse_args(
                ["--out-dir", str(out_dir), "--resume", "--overwrite-output"]
            )

            with pytest.raises(ValueError, match="overwrite-output"):
                train.validate_resume_request(args)

    def test_launch_input_validation_rejects_overlapping_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp) / "shared"
            args = train.parse_args(
                [
                    "--out-dir",
                    str(shared),
                    "--dataset-path",
                    str(shared),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                ]
            )
            buckets = train.parse_bucket_list(args.buckets)
            bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)

            with pytest.raises(ValueError, match="overlaps"):
                train.validate_launch_inputs(
                    args,
                    buckets=buckets,
                    bucket_cp=bucket_cp,
                )

    def test_launch_input_validation_rejects_dangerous_output_overwrite_path(self):
        repo_root = Path(__file__).resolve().parents[1]
        args = train.parse_args(
            [
                "--out-dir",
                str(repo_root),
                "--dataset-path",
                str(repo_root / "datasets" / train.TRAINING_DATASET_NAME),
                "--hf-assets-path",
                str(repo_root / "assets" / "hf"),
                "--overwrite-output",
            ]
        )
        buckets = train.parse_bucket_list(args.buckets)
        bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)

        with pytest.raises(ValueError, match="dangerous.*--out-dir"):
            train.validate_launch_inputs(args, buckets=buckets, bucket_cp=bucket_cp)

    def test_launch_input_validation_rejects_dangerous_dataset_rebuild_path(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(repo_root),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--rebuild-source-dataset",
                ]
            )
            buckets = train.parse_bucket_list(args.buckets)
            bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)

            with pytest.raises(ValueError, match="dangerous.*--dataset-path"):
                train.validate_launch_inputs(
                    args,
                    buckets=buckets,
                    bucket_cp=bucket_cp,
                )

    def test_resume_ignores_final_model_export_when_deciding_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint_root = train._checkpoint_dir(args.out_dir)
            final_export_root = train._final_model_export_dir(args.out_dir)
            full = checkpoint_root / "step-5"
            export = final_export_root / "step-5"
            full.mkdir(parents=True)
            export.mkdir(parents=True)
            (full / ".metadata").write_text("{}")
            (export / "model.safetensors.index.json").write_text("{}")
            args.resume = True

            resume_state = train.validate_resume_request(args)
            train.validate_resume_progress(plan, resume_state)

        assert resume_state.latest_resumable_step == 5
        assert resume_state.latest_model_export_step == 5
        assert resume_state.latest_any_step == 5
        assert train.stages_to_run_for_resume(plan, resume_state) == ()

    def test_resume_rejects_completed_final_export_without_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            export = train._final_model_export_dir(args.out_dir) / "step-5"
            export.mkdir(parents=True)
            (export / "model.safetensors.index.json").write_text("{}")
            args.resume = True

            resume_state = train.validate_resume_request(args)

            with pytest.raises(RuntimeError, match="final.*full DCP"):
                train.validate_resume_progress(plan, resume_state)

    def test_resume_rejects_nonfinal_model_export_newer_than_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint_root = train._checkpoint_dir(args.out_dir)
            final_export_root = train._final_model_export_dir(args.out_dir)
            full = checkpoint_root / "step-2"
            export = final_export_root / "step-3"
            full.mkdir(parents=True)
            export.mkdir(parents=True)
            (full / ".metadata").write_text("{}")
            (export / "model.safetensors.index.json").write_text("{}")
            resume_state = train.ResumeCheckpointState(
                checkpoint_dir=checkpoint_root,
                final_export_dir=final_export_root,
                latest_resumable_step=2,
                latest_model_export_step=3,
                latest_any_step=3,
            )

            with pytest.raises(RuntimeError, match="non-resumable export"):
                train.validate_resume_progress(plan, resume_state)

    def test_resume_rejects_incomplete_export_without_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            final_export_root = train._final_model_export_dir(args.out_dir)
            export = final_export_root / "step-3"
            export.mkdir(parents=True)
            (export / "model.safetensors.index.json").write_text("{}")
            resume_state = train.ResumeCheckpointState(
                checkpoint_dir=train._checkpoint_dir(args.out_dir),
                final_export_dir=final_export_root,
                latest_resumable_step=None,
                latest_model_export_step=3,
                latest_any_step=3,
            )

            with pytest.raises(RuntimeError, match="full DCP"):
                train.validate_resume_progress(plan, resume_state)

    def test_final_artifact_validation_writes_report_for_completed_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_dcp_checkpoint(args.out_dir, step=5)
            self._write_final_export(args.out_dir, step=5)

            report = train.validate_final_artifacts(args, plan)
            persisted = json.loads(
                (args.out_dir / train.FINAL_ARTIFACT_VALIDATION_FILENAME).read_text()
            )

        assert (
            report["schema_version"] == train.FINAL_ARTIFACT_VALIDATION_SCHEMA_VERSION
        )
        assert report == persisted
        assert report["plan_total_steps"] == 5
        assert report["final_export"]["layout"] == "final_export"
        assert report["final_export"]["shard_count"] == 2
        assert report["final_export"]["weight_map_entries"] == 2
        assert report["final_export"]["index_metadata_total_size"] == 14
        assert report["final_export"]["shards"][0]["sha256"] == train._sha256_text(
            "shard-1"
        )
        assert report["resumable_checkpoints"]["steps"] == [5]
        assert report["resumable_checkpoints"]["latest_step"] == 5
        assert (
            report["resumable_checkpoints"]["checkpoints"][0]["payload_file_count"] == 1
        )
        assert (
            (
                report["resumable_checkpoints"]["checkpoints"][0]["payload_files"][0][
                    "rank"
                ]
            )
            == 0
        )

    def test_final_artifact_validation_requires_final_resumable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_dcp_checkpoint(args.out_dir, step=4)
            self._write_final_export(args.out_dir, step=5)

            with pytest.raises(RuntimeError, match="Final resumable DCP"):
                train.validate_final_artifacts(args, plan, write_report=False)

    def test_final_artifact_validation_rejects_missing_export_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            export = self._write_final_export(args.out_dir, step=5)
            (export / "model-00002-of-00002.safetensors").unlink()

            with pytest.raises(RuntimeError, match="final model export shard"):
                train.validate_final_artifacts(args, plan, write_report=False)

    def test_final_artifact_validation_rejects_unindexed_export_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_dcp_checkpoint(args.out_dir, step=5)
            export = self._write_final_export(args.out_dir, step=5)
            (export / "orphan.safetensors").write_bytes(b"orphan")

            with pytest.raises(RuntimeError, match="unindexed"):
                train.validate_final_artifacts(args, plan, write_report=False)

    def test_final_artifact_validation_rejects_impossible_export_total_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_dcp_checkpoint(args.out_dir, step=5)
            export = self._write_final_export(args.out_dir, step=5)
            index_path = export / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["metadata"]["total_size"] = 10_000
            index_path.write_text(json.dumps(index))

            with pytest.raises(RuntimeError, match="total_size"):
                train.validate_final_artifacts(args, plan, write_report=False)

    def test_final_artifact_validation_rejects_empty_dcp_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint = self._write_dcp_checkpoint(args.out_dir, step=5)
            (checkpoint / "__0_0.distcp").write_bytes(b"")
            self._write_final_export(args.out_dir, step=5)

            with pytest.raises(RuntimeError, match="DCP checkpoint payload"):
                train.validate_final_artifacts(args, plan, write_report=False)

    def test_final_artifact_validation_rejects_malformed_dcp_payload_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint = self._write_dcp_checkpoint(args.out_dir, step=5)
            (checkpoint / "__0_0.distcp").replace(checkpoint / "rank0.distcp")
            self._write_final_export(args.out_dir, step=5)

            with pytest.raises(RuntimeError, match="payload file name"):
                train.validate_final_artifacts(args, plan, write_report=False)

    def test_first_step_checkpoint_validation_accepts_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, _plan = self._resume_test_setup(tmp)
            written = self._write_first_step_checkpoint_validation_report(args.out_dir)

            report = train.validate_first_step_checkpoint_report(args)

        assert report == written
        assert report["step"] == 1
        assert report["checkpoint"]["payload_file_count"] == 1
        assert report["checkpoint"]["payload_rank_count"] == 1

    def test_first_step_checkpoint_validation_rejects_missing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, _plan = self._resume_test_setup(tmp)

            with pytest.raises(RuntimeError, match="First-step checkpoint"):
                train.validate_first_step_checkpoint_report(args)

    def test_first_step_checkpoint_validation_rejects_wrong_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, _plan = self._resume_test_setup(tmp)
            checkpoint = train._validate_dcp_checkpoint_step(
                self._write_dcp_checkpoint(args.out_dir, step=2)
            )
            report = {
                "schema_version": train.FIRST_STEP_CHECKPOINT_VALIDATION_SCHEMA_VERSION,
                "created_at_unix": 1.0,
                "step": 2,
                "checkpoint": checkpoint,
            }
            train._first_step_checkpoint_validation_path(args.out_dir).write_text(
                json.dumps(report, indent=2)
            )

            with pytest.raises(RuntimeError, match="step 1"):
                train.validate_first_step_checkpoint_report(args)

    def test_final_artifact_validation_rejects_duplicate_legacy_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_final_export(args.out_dir, step=5)
            self._write_final_export(args.out_dir, step=5, legacy=True)

            with pytest.raises(RuntimeError, match="both"):
                train.validate_final_artifacts(args, plan, write_report=False)

    def test_final_artifact_validation_rejects_legacy_export_without_final_checkpoint(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_final_export(args.out_dir, step=5, legacy=True)

            with pytest.raises(RuntimeError, match="Final model export is missing"):
                train.validate_final_artifacts(args, plan, write_report=False)
            with pytest.raises(RuntimeError, match="Final resumable DCP"):
                train.validate_final_artifacts(
                    args,
                    plan,
                    allow_legacy_export=True,
                    write_report=False,
                )

    def test_post_training_eval_hook_records_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_dcp_checkpoint(args.out_dir, step=5)
            self._write_final_export(args.out_dir, step=5)
            final_validation = train.validate_final_artifacts(args, plan)
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=(),
                dataloader_resume_flags={},
            )
            code = (
                "import os, pathlib; "
                "pathlib.Path(os.environ['SWEHERO_OUT_DIR'], "
                "'eval-step.txt').write_text("
                "os.environ['SWEHERO_FINAL_EXPORT_STEP'])"
            )
            args.post_training_eval_command = (
                f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
            )

            eval_status = train.run_post_training_eval(args, plan, final_validation)
            persisted = json.loads(
                (args.out_dir / train.POST_TRAINING_EVAL_STATUS_FILENAME).read_text()
            )
            stage_status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )
            eval_step_text = (args.out_dir / "eval-step.txt").read_text()

        assert eval_status is not None
        assert eval_status == persisted
        assert eval_status["status"] == "succeeded"
        assert eval_status["returncode"] == 0
        assert eval_status["env_overrides"]["SWEHERO_FINAL_EXPORT_STEP"] == "5"
        assert eval_step_text == "5"
        assert stage_status["post_training_eval"]["status"] == "succeeded"
        assert stage_status["summary"]["post_training_eval_status"] == "succeeded"

    def test_post_training_eval_hook_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_dcp_checkpoint(args.out_dir, step=5)
            self._write_final_export(args.out_dir, step=5)
            final_validation = train.validate_final_artifacts(args, plan)
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=(),
                dataloader_resume_flags={},
            )
            code = "import sys; print('eval failed'); sys.exit(3)"
            args.post_training_eval_command = (
                f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
            )

            with pytest.raises(RuntimeError, match="return code 3"):
                train.run_post_training_eval(args, plan, final_validation)

            persisted = json.loads(
                (args.out_dir / train.POST_TRAINING_EVAL_STATUS_FILENAME).read_text()
            )
            stage_status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )

        assert persisted["status"] == "failed"
        assert persisted["returncode"] == 3
        assert "eval failed" in persisted["stdout_tail"]
        assert stage_status["post_training_eval"]["status"] == "failed"
        assert stage_status["summary"]["failure_count"] == 1

    def test_stage_status_records_successful_stage_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            args.validate_first_step_checkpoint = False
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )

            with (
                patch.object(train, "_run_command_with_signal_forwarding") as run_mock,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                train.run_stage_with_status(args, plan.stages[0], plan, manifest)

            status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )

        run_mock.assert_called_once()
        assert "log_paths" in run_mock.call_args.kwargs
        stage = status["stages"][0]
        assert stage["status"] == "succeeded"
        assert stage["attempts"][0]["status"] == "succeeded"
        assert "torchrun_command" in stage["attempts"][0]
        assert stage["attempts"][0]["logs"] == (
            train._stage_attempt_log_paths(
                args.out_dir,
                stage_id=stage["id"],
                attempt_number=1,
            )
        )
        assert status["failures"] == []
        assert status["summary"]["stage_status_counts"]["succeeded"] == 1

    def test_stage_command_persists_stdout_and_stderr_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_paths = {
                "stdout": str(root / "logs" / "stage.stdout.log"),
                "stderr": str(root / "logs" / "stage.stderr.log"),
            }
            command = [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "print('stdout line'); "
                    "print('stderr line', file=sys.stderr)"
                ),
            ]

            train._run_command_with_signal_forwarding(
                command,
                env=os.environ.copy(),
                cwd=root,
                log_paths=log_paths,
            )

            assert Path(log_paths["stdout"]).read_text() == "stdout line\n"
            assert Path(log_paths["stderr"]).read_text() == "stderr line\n"

    def test_stage_status_records_failed_stage_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )
            error = train.subprocess.CalledProcessError(
                42,
                ["torchrun", "-m", "torchtitan.train"],
            )

            with (
                patch.object(
                    train,
                    "_run_command_with_signal_forwarding",
                    side_effect=error,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with pytest.raises(train.subprocess.CalledProcessError):
                    train.run_stage_with_status(args, plan.stages[0], plan, manifest)

            status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )

        stage = status["stages"][0]
        assert stage["status"] == "failed"
        assert stage["attempts"][0]["status"] == "failed"
        assert stage["failure"]["returncode"] == 42
        assert status["failures"][0]["phase"] == "stage"
        assert status["failures"][0]["stage_id"] == stage["id"]
        assert status["summary"]["failure_count"] == 1

    def test_stage_status_records_signal_terminated_stage_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )
            error = train.SignalTerminationError(
                signum=int(signal.SIGTERM),
                command=["torchrun", "-m", "torchtitan.train"],
                returncode=-int(signal.SIGTERM),
            )

            with (
                patch.object(
                    train,
                    "_run_command_with_signal_forwarding",
                    side_effect=error,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with pytest.raises(train.SignalTerminationError):
                    train.run_stage_with_status(args, plan.stages[0], plan, manifest)

            status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )

        stage = status["stages"][0]
        failure = stage["failure"]
        assert stage["status"] == "failed"
        assert failure["terminated_by_signal"]
        assert failure["signum"] == int(signal.SIGTERM)
        assert failure["signal_name"] == "SIGTERM"
        assert failure["returncode"] == -int(signal.SIGTERM)
        assert status["failures"][0] == failure

    def test_stage_command_forwards_sigterm_to_process_group(self):
        command = ["torchrun", "-m", "torchtitan.train"]

        class FakeProcess:
            pid = 12345

            def __init__(self) -> None:
                self.returncode = None

            def poll(self):
                return self.returncode

        fake_process = FakeProcess()
        installed_handlers = {}

        def fake_signal(signum, handler):
            signum = int(signum)
            previous = installed_handlers.get(signum, signal.SIG_DFL)
            installed_handlers[signum] = handler
            return previous

        def fake_sleep(_seconds):
            handler = installed_handlers[int(signal.SIGTERM)]
            handler(int(signal.SIGTERM), None)
            fake_process.returncode = -int(signal.SIGTERM)

        with (
            patch.object(train.subprocess, "Popen", return_value=fake_process) as popen,
            patch.object(train.signal, "signal", side_effect=fake_signal),
            patch.object(train.time, "sleep", side_effect=fake_sleep),
            patch.object(train, "_send_signal_to_process_group") as send_signal,
        ):
            with pytest.raises(train.SignalTerminationError) as raised:
                train._run_command_with_signal_forwarding(
                    command,
                    env={"A": "B"},
                    cwd=Path("/tmp"),
                )

        popen.assert_called_once()
        assert popen.call_args.kwargs["env"] == {"A": "B"}
        assert popen.call_args.kwargs["cwd"] == Path("/tmp")
        assert popen.call_args.kwargs["start_new_session"]
        send_signal.assert_called_with(fake_process.pid, int(signal.SIGTERM))
        assert raised.value.signum == int(signal.SIGTERM)
        assert raised.value.returncode == -int(signal.SIGTERM)

    def test_stage_status_marks_resume_completed_and_pending_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            self._write_dcp_checkpoint(args.out_dir, step=4)
            args.resume = True
            resume_state = train.validate_resume_request(args)
            stages_to_run = train.stages_to_run_for_resume(plan, resume_state)
            dataloader_resume_flags = train.dataloader_resume_flags_by_stage(
                plan,
                resume_state,
            )

            status = train.initialize_stage_status(
                args,
                plan,
                resume_state=resume_state,
                stages_to_run=stages_to_run,
                dataloader_resume_flags=dataloader_resume_flags,
            )

        assert [stage["status"] for stage in status["stages"]] == (
            [
                "completed_before_resume",
                "pending",
            ]
        )
        assert status["launch"]["stages_to_run"] == (
            [
                status["stages"][1]["id"],
            ]
        )

    def test_stage_status_recovers_stale_running_attempt_before_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            args.validate_first_step_checkpoint = False
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )
            status_path = args.out_dir / train.STAGE_STATUS_FILENAME
            stale = json.loads(status_path.read_text())
            stale_stage = stale["stages"][0]
            stale_stage["status"] = "running"
            stale_stage["started_at_unix"] = 100.0
            stale_stage["finished_at_unix"] = None
            stale_stage["duration_seconds"] = None
            stale_stage["attempts"] = [
                {
                    "attempt": 1,
                    "status": "running",
                    "started_at_unix": 100.0,
                    "finished_at_unix": None,
                    "duration_seconds": None,
                    "failure": None,
                }
            ]
            train._write_json_atomic(status_path, stale)

            recovered = train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )
            recovered_stage = recovered["stages"][0]

            with (
                patch.object(train, "_run_command_with_signal_forwarding"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                train.run_stage_with_status(args, plan.stages[0], plan, manifest)

            final_status = json.loads(status_path.read_text())

        assert recovered_stage["status"] == "pending"
        assert recovered_stage["attempts"][0]["status"] == "stale_recovered"
        assert recovered_stage["attempts"][0]["failure"]["phase"] == "stage_recovery"
        assert recovered["failures"][0]["phase"] == "stage_recovery"
        assert final_status["stages"][0]["status"] == "succeeded"
        assert [
            attempt["status"] for attempt in final_status["stages"][0]["attempts"]
        ] == ["stale_recovered", "succeeded"]
        assert final_status["summary"]["running_stage_ids"] == []

    def test_stage_status_recovers_stale_running_validation_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            args.post_training_eval_command = "true"
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )
            status_path = args.out_dir / train.STAGE_STATUS_FILENAME
            stale = json.loads(status_path.read_text())
            for key in (
                "first_step_checkpoint_validation",
                "final_artifact_validation",
                "post_training_eval",
            ):
                stale[key]["status"] = "running"
                stale[key]["started_at_unix"] = 100.0
                stale[key]["finished_at_unix"] = None
                stale[key]["duration_seconds"] = None
            train._write_json_atomic(status_path, stale)

            recovered = train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )

        assert recovered["first_step_checkpoint_validation"]["status"] == "pending"
        assert recovered["final_artifact_validation"]["status"] == "pending"
        assert recovered["post_training_eval"]["status"] == "pending"
        for key in (
            "first_step_checkpoint_validation",
            "final_artifact_validation",
            "post_training_eval",
        ):
            assert recovered[key]["finished_at_unix"] is None
            assert recovered[key]["duration_seconds"] is None
        assert (
            recovered["summary"]["first_step_checkpoint_validation_status"] == "pending"
        )
        assert recovered["summary"]["final_artifact_validation_status"] == "pending"
        assert recovered["summary"]["post_training_eval_status"] == "pending"

    def test_final_validation_status_records_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )
            self._write_dcp_checkpoint(args.out_dir, step=5)
            self._write_final_export(args.out_dir, step=5)

            train.validate_final_artifacts_with_status(args, plan)
            status_path = args.out_dir / train.STAGE_STATUS_FILENAME
            status = json.loads(status_path.read_text())
            report_sha256 = train._hash_file(
                args.out_dir / train.FINAL_ARTIFACT_VALIDATION_FILENAME
            )

        final_status = status["final_artifact_validation"]
        assert final_status["status"] == "succeeded"
        assert final_status["report_sha256"] == report_sha256
        assert final_status["summary"]["final_export"]["shard_count"] == 2
        assert status["summary"]["final_artifact_validation_status"] == "succeeded"

    def test_final_validation_status_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )

            with pytest.raises(RuntimeError, match="Final model export is missing"):
                train.validate_final_artifacts_with_status(args, plan)

            status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )

        final_status = status["final_artifact_validation"]
        assert final_status["status"] == "failed"
        assert final_status["failure"]["phase"] == "final_artifact_validation"
        assert status["failures"][0]["phase"] == "final_artifact_validation"
        assert status["summary"]["failure_count"] == 1

    def test_resume_contract_accepts_same_config_and_skips_completed_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            train._write_resume_contract(args, plan, manifest)
            args.resume = True
            checkpoint_root = args.out_dir / "torchtitan" / "checkpoint"
            latest = checkpoint_root / "step-4"
            latest.mkdir(parents=True)
            (latest / ".metadata").write_text("{}")
            resume_state = train.validate_resume_request(args)

            train.validate_resume_contract(args, plan, manifest)
            stages = train.stages_to_run_for_resume(plan, resume_state)

        assert [stage.bucket for stage in stages] == [65536]

    def test_run_spec_is_written_once_with_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            with patch.dict(os.environ, {"SWEHERO_SECRET": "do-not-record"}):
                written = train.write_or_validate_run_spec(args, plan, manifest)
                written_again = train.write_or_validate_run_spec(args, plan, manifest)

            spec_path = args.out_dir / train.RUN_SPEC_FILENAME
            sha_path = args.out_dir / train.RUN_SPEC_SHA256_FILENAME
            spec_text = spec_path.read_text()
            spec_sha = sha_path.read_text().strip()
            spec = json.loads(spec_text)

        assert written
        assert not (written_again)
        assert spec_sha == train._sha256_text(spec_text)
        assert spec["schema_version"] == train.RUN_SPEC_SCHEMA_VERSION
        assert not (spec["args"]["production_mode"])
        assert spec["args"]["max_length"] == 65536
        assert spec["manifest"] == train._resume_manifest_contract(manifest)
        assert spec["paths"]["resumable_checkpoints"] == str(
            train._checkpoint_dir(args.out_dir)
        )
        assert spec["paths"]["final_model_exports"] == str(
            train._final_model_export_dir(args.out_dir)
        )
        assert spec["paths"]["first_step_checkpoint_validation"] == str(
            train._first_step_checkpoint_validation_path(args.out_dir)
        )
        assert spec["paths"]["workspace_root"] == str(
            train._configured_workspace_root(args)
        )
        assert spec["paths"]["launch_lock"] == str(
            train._launch_lock_path(args.out_dir)
        )
        assert spec["workspace"]["configured_root"] == str(
            train._configured_workspace_root(args)
        )
        assert spec["workspace"]["script_root"] == str(train._detected_workspace_root())
        assert spec["plan"]["total_steps"] == plan.total_steps
        first_env = spec["plan"]["stages"][0]["env_overrides"]
        assert first_env["SWEHERO_WORKSPACE_ROOT"] == str(
            train._configured_workspace_root(args)
        )
        assert first_env["SWEHERO_BUCKET_SEQ_LEN"] == "32768"
        assert first_env["SWEHERO_ALLOW_EMPTY_RANK_REUSE"] == "1"
        assert (
            first_env["SWEHERO_FINAL_EXPORT_FOLDER"] == train.FINAL_MODEL_EXPORT_FOLDER
        )
        assert first_env["SWEHERO_SAVE_FINAL_FULL_CHECKPOINT"] == "1"
        assert first_env["SWEHERO_ENABLE_FIRST_STEP_CHECKPOINT"] == "1"
        assert first_env["SWEHERO_FIRST_STEP_CHECKPOINT_VALIDATION_REPORT"] == str(
            train._first_step_checkpoint_validation_path(args.out_dir)
        )
        assert first_env["SWEHERO_ENABLE_PROFILER"] == "0"
        assert first_env["SWEHERO_PROFILER_FREQ"] == "10"
        assert first_env["SWEHERO_ENABLE_MEMORY_SNAPSHOT"] == "0"
        assert "SWEHERO_SECRET" not in first_env

    def test_production_mode_is_recorded_in_run_spec_and_resume_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            args.production_mode = True
            spec = train.build_run_spec(args, plan, manifest)
            contract = train.build_resume_contract(args, plan, manifest)

        assert spec["args"]["production_mode"]
        assert contract["args"]["production_mode"]
        assert not (spec["args"]["production_acceptance_smoke"])
        assert not (contract["args"]["production_acceptance_smoke"])
        assert spec["paper_alignment"]["run_safety"]["production_mode"]
        assert not (
            spec["paper_alignment"]["run_safety"]["production_acceptance_smoke"]
        )
        assert spec["workspace"]["configured_root"] == str(args.workspace_root)
        assert contract["workspace"]["configured_root"] == str(args.workspace_root)
        assert (
            (
                spec["plan"]["stages"][0]["env_overrides"][
                    "SWEHERO_ALLOW_EMPTY_RANK_REUSE"
                ]
            )
            == "0"
        )
        assert contract["stage_env"][0]["env"]["SWEHERO_ALLOW_EMPTY_RANK_REUSE"] == "0"

    def test_production_acceptance_smoke_records_empty_rank_reuse_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            args.production_mode = True
            args.production_acceptance_smoke = True
            spec = train.build_run_spec(args, plan, manifest)
            contract = train.build_resume_contract(args, plan, manifest)

        assert spec["args"]["production_mode"]
        assert spec["args"]["production_acceptance_smoke"]
        assert spec["paper_alignment"]["run_safety"]["production_acceptance_smoke"]
        assert (
            (
                spec["plan"]["stages"][0]["env_overrides"][
                    "SWEHERO_ALLOW_EMPTY_RANK_REUSE"
                ]
            )
            == "1"
        )
        assert contract["stage_env"][0]["env"]["SWEHERO_ALLOW_EMPTY_RANK_REUSE"] == "1"

    def test_torchtitan_stage_settings_are_recorded_as_launch_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            args.optimizer_impl = "fused"
            args.training_dtype = "bfloat16"
            args.mixed_precision_param_dtype = "float32"
            args.mixed_precision_reduce_dtype = "float32"
            args.fsdp_reshard_after_forward = "always"
            args.detect_anomaly = True
            args.cuda_device_max_connections = "2"
            args.torch_nccl_async_error_handling = "3"
            spec = train.build_run_spec(args, plan, manifest)
            first_env = spec["plan"]["stages"][0]["env_overrides"]

        assert spec["args"]["optimizer_impl"] == "fused"
        assert spec["args"]["training_dtype"] == "bfloat16"
        assert spec["args"]["mixed_precision_param_dtype"] == "float32"
        assert spec["args"]["mixed_precision_reduce_dtype"] == "float32"
        assert spec["args"]["fsdp_reshard_after_forward"] == "always"
        assert spec["args"]["detect_anomaly"]
        assert spec["args"]["cuda_device_max_connections"] == "2"
        assert spec["args"]["torch_nccl_async_error_handling"] == "3"
        assert first_env["SWEHERO_OPTIMIZER_IMPL"] == "fused"
        assert first_env["SWEHERO_TRAINING_DTYPE"] == "bfloat16"
        assert first_env["SWEHERO_MP_PARAM_DTYPE"] == "float32"
        assert first_env["SWEHERO_MP_REDUCE_DTYPE"] == "float32"
        assert first_env["SWEHERO_FSDP_RESHARD_AFTER_FORWARD"] == "always"
        assert first_env["SWEHERO_DETECT_ANOMALY"] == "1"
        assert first_env["CUDA_DEVICE_MAX_CONNECTIONS"] == "2"
        assert first_env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "3"

    def test_first_step_checkpoint_validation_marks_status_after_first_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            stage = plan.stages[0]
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )

            def fake_run_stage(*_args, **_kwargs):
                self._write_first_step_checkpoint_validation_report(args.out_dir)

            with patch.object(train, "run_stage", side_effect=fake_run_stage):
                train.run_stage_with_status(args, stage, plan, manifest)

            status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )

        assert status["stages"][0]["status"] == "succeeded"
        first_validation = status["first_step_checkpoint_validation"]
        assert first_validation["status"] == "succeeded"
        assert first_validation["summary"]["step"] == 1
        assert (
            status["summary"]["first_step_checkpoint_validation_status"] == "succeeded"
        )

    def test_first_step_checkpoint_validation_failure_marks_stage_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            stage = plan.stages[0]
            train.initialize_stage_status(
                args,
                plan,
                resume_state=None,
                stages_to_run=plan.stages,
                dataloader_resume_flags={},
            )

            with (
                patch.object(train, "run_stage", return_value=None),
                pytest.raises(RuntimeError, match="First-step checkpoint"),
            ):
                train.run_stage_with_status(args, stage, plan, manifest)

            status = json.loads(
                (args.out_dir / train.STAGE_STATUS_FILENAME).read_text()
            )

        assert status["stages"][0]["status"] == "failed"
        first_validation = status["first_step_checkpoint_validation"]
        assert first_validation["status"] == "failed"
        assert (
            first_validation["failure"]["phase"] == "first_step_checkpoint_validation"
        )
        assert status["failures"][0]["phase"] == "first_step_checkpoint_validation"

    def test_run_spec_rejects_hidden_torchtitan_env_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            dataset_path = Path(tmp) / "dataset"
            hf_assets_path = Path(tmp) / "hf" / "Qwen2.5-Coder-7B-Instruct"
            base_argv = [
                "--out-dir",
                str(out_dir),
                "--dataset-path",
                str(dataset_path),
                "--hf-assets-path",
                str(hf_assets_path),
                "--buckets",
                "32768,65536",
                "--bucket-cp",
                "32768:2,65536:4",
                "--max-length",
                "65536",
                "--num-examples",
                "34",
                "--max-streamed-examples",
                "100",
            ]
            args, manifest, plan = self._resume_test_setup(tmp)
            args.training_dtype = "bfloat16"
            train.write_or_validate_run_spec(args, plan, manifest)

            with patch.dict(os.environ, {}, clear=True):
                changed = train.parse_args(base_argv)
                changed.buckets = ",".join(
                    str(b) for b in train.parse_bucket_list(changed.buckets)
                )
                bucket_cp = train.parse_bucket_cp_map(changed.bucket_cp)
                changed.bucket_cp = train._format_bucket_cp_map(bucket_cp)

            with pytest.raises(RuntimeError, match="training_dtype"):
                train.write_or_validate_run_spec(changed, plan, manifest)

    def test_stage_env_exposes_profiler_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            args.enable_profiler = True
            args.profiler_trace_folder = "profiles/traces"
            args.profiler_freq = 4
            args.profiler_active = 1
            args.profiler_warmup = 1
            args.profiler_repeat = 2
            args.profiler_skip_first = 1
            args.profiler_skip_first_wait = 1
            args.enable_memory_snapshot = True
            args.memory_snapshot_folder = "profiles/memory"

            env = train.build_stage_env(
                args,
                stage=plan.stages[0],
                total_steps=plan.total_steps,
                warmup_steps=plan.warmup_steps,
                pad_token_id=0,
            )

        assert env["SWEHERO_ENABLE_PROFILER"] == "1"
        assert env["SWEHERO_PROFILER_TRACE_FOLDER"] == "profiles/traces"
        assert env["SWEHERO_PROFILER_FREQ"] == "4"
        assert env["SWEHERO_PROFILER_ACTIVE"] == "1"
        assert env["SWEHERO_PROFILER_WARMUP"] == "1"
        assert env["SWEHERO_PROFILER_REPEAT"] == "2"
        assert env["SWEHERO_PROFILER_SKIP_FIRST"] == "1"
        assert env["SWEHERO_PROFILER_SKIP_FIRST_WAIT"] == "1"
        assert env["SWEHERO_ENABLE_MEMORY_SNAPSHOT"] == "1"
        assert env["SWEHERO_MEMORY_SNAPSHOT_FOLDER"] == "profiles/memory"

    def test_run_spec_rejects_changed_launch_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            train.write_or_validate_run_spec(args, plan, manifest)
            changed = train.parse_args(
                [
                    "--out-dir",
                    str(args.out_dir),
                    "--dataset-path",
                    str(args.dataset_path),
                    "--hf-assets-path",
                    str(args.hf_assets_path),
                    "--buckets",
                    args.buckets,
                    "--bucket-cp",
                    args.bucket_cp,
                    "--max-length",
                    str(args.max_length),
                    "--num-examples",
                    str(args.num_examples),
                    "--max-streamed-examples",
                    str(args.max_streamed_examples),
                    "--learning-rate",
                    "2e-5",
                ]
            )

            with pytest.raises(RuntimeError, match="learning_rate"):
                train.write_or_validate_run_spec(changed, plan, manifest)

    def test_run_spec_rejects_tampered_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            train.write_or_validate_run_spec(args, plan, manifest)
            spec_path = args.out_dir / train.RUN_SPEC_FILENAME
            spec = json.loads(spec_path.read_text())
            spec["args"]["max_length"] = 123
            spec_path.write_text(json.dumps(spec, indent=2))

            with pytest.raises(RuntimeError, match="checksum mismatch"):
                train.write_or_validate_run_spec(args, plan, manifest)

    def test_resume_requires_existing_run_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)

            with pytest.raises(RuntimeError, match="requires an immutable run spec"):
                train.write_or_validate_run_spec(
                    args,
                    plan,
                    manifest,
                    require_existing=True,
                )

    def test_mid_stage_resume_loads_dataloader_state_for_current_stage_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint_root = train._checkpoint_dir(args.out_dir)
            resume_state = train.ResumeCheckpointState(
                checkpoint_dir=checkpoint_root,
                final_export_dir=train._final_model_export_dir(args.out_dir),
                latest_resumable_step=2,
                latest_model_export_step=None,
                latest_any_step=2,
            )

            flags = train.dataloader_resume_flags_by_stage(plan, resume_state)
            stages = train.stages_to_run_for_resume(plan, resume_state)

            first_env = train.build_stage_env(
                args,
                stage=plan.stages[0],
                total_steps=plan.total_steps,
                warmup_steps=plan.warmup_steps,
                pad_token_id=151643,
                load_dataloader_state=flags[plan.stages[0].cumulative_steps],
            )
            second_env = train.build_stage_env(
                args,
                stage=plan.stages[1],
                total_steps=plan.total_steps,
                warmup_steps=plan.warmup_steps,
                pad_token_id=151643,
                load_dataloader_state=flags[plan.stages[1].cumulative_steps],
            )

        assert [stage.bucket for stage in stages] == [32768, 65536]
        assert flags[plan.stages[0].cumulative_steps]
        assert not (flags[plan.stages[1].cumulative_steps])
        assert first_env["SWEHERO_LOAD_DATALOADER_STATE"] == "1"
        assert second_env["SWEHERO_LOAD_DATALOADER_STATE"] == "0"

    def test_stage_env_disables_empty_rank_reuse_in_production_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            stage = train.BucketStage(
                bucket=256,
                cp_degree=1,
                example_count=1,
                steps=1,
                cumulative_steps=1,
                bucket_file=out_dir / "data" / "bucket_256.jsonl",
            )
            smoke_args = train.parse_args(["--out-dir", str(out_dir)])
            production_args = train.parse_args(
                ["--out-dir", str(out_dir), "--production-mode"]
            )

            smoke_env = train.build_stage_env(
                smoke_args,
                stage=stage,
                total_steps=1,
                warmup_steps=0,
                pad_token_id=151643,
            )
            production_env = train.build_stage_env(
                production_args,
                stage=stage,
                total_steps=1,
                warmup_steps=0,
                pad_token_id=151643,
            )

        assert smoke_env["SWEHERO_ALLOW_EMPTY_RANK_REUSE"] == "1"
        assert production_env["SWEHERO_ALLOW_EMPTY_RANK_REUSE"] == "0"

    def test_stage_boundary_resume_does_not_load_previous_bucket_dataloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _manifest, plan = self._resume_test_setup(tmp)
            checkpoint_root = train._checkpoint_dir(args.out_dir)
            resume_state = train.ResumeCheckpointState(
                checkpoint_dir=checkpoint_root,
                final_export_dir=train._final_model_export_dir(args.out_dir),
                latest_resumable_step=4,
                latest_model_export_step=None,
                latest_any_step=4,
            )

            flags = train.dataloader_resume_flags_by_stage(plan, resume_state)
            stages = train.stages_to_run_for_resume(plan, resume_state)

        assert [stage.bucket for stage in stages] == [65536]
        assert not (flags[plan.stages[0].cumulative_steps])
        assert not (flags[plan.stages[1].cumulative_steps])

    def test_swehero_config_can_load_dataloader_state_for_mid_stage_resume(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (
            repo_root / "torchtitan/torchtitan/experiments/swehero/config_registry.py"
        ).read_text()

        assert "SWEHERO_LOAD_DATALOADER_STATE" in source
        assert "SWEHERO_ALLOW_EMPTY_RANK_REUSE" in source
        assert "_checkpoint_exclude_from_loading()" in source
        assert 'exclude_from_loading=["dataloader"]' not in source

    def test_swehero_config_wires_profiler_env_controls(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (
            repo_root / "torchtitan/torchtitan/experiments/swehero/config_registry.py"
        ).read_text()

        assert "Profiler.Config" in source
        assert "SWEHERO_ENABLE_PROFILER" in source
        assert "SWEHERO_PROFILER_FREQ" in source
        assert "SWEHERO_ENABLE_MEMORY_SNAPSHOT" in source
        assert "SWEHERO_MEMORY_SNAPSHOT_FOLDER" in source

    def test_swehero_config_routes_final_export_outside_checkpoint_dir(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (
            repo_root / "torchtitan/torchtitan/experiments/swehero/config_registry.py"
        ).read_text()

        assert "final_model_export_folder=" in source
        assert "SWEHERO_FINAL_EXPORT_FOLDER" in source
        assert "save_last_step_full_checkpoint=" in source
        assert "SWEHERO_SAVE_FINAL_FULL_CHECKPOINT" in source
        assert "enable_first_step_checkpoint=" in source
        assert "SWEHERO_ENABLE_FIRST_STEP_CHECKPOINT" in source
        assert "first_step_checkpoint_validation_report=" in source
        assert "SWEHERO_FIRST_STEP_CHECKPOINT_VALIDATION_REPORT" in source
        assert '"final_export"' in source

    def test_torchtitan_checkpoint_manager_supports_separate_final_exports(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (
            repo_root / "torchtitan/torchtitan/components/checkpoint.py"
        ).read_text()

        assert "final_model_export_folder" in source
        assert "self.final_model_export_folder" in source
        assert "save_last_step_full_checkpoint" in source
        assert "Saving a full resumable checkpoint at last step" in source
        assert "checkpoint.final_model_export_folder must differ" in source
        assert "first_step_checkpoint_validation_report" in source
        assert "_validate_first_step_checkpoint" in source

    def test_resume_contract_rejects_changed_training_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, manifest, plan = self._resume_test_setup(tmp)
            train._write_resume_contract(args, plan, manifest)
            changed = train.parse_args(
                [
                    "--out-dir",
                    str(args.out_dir),
                    "--dataset-path",
                    str(args.dataset_path),
                    "--hf-assets-path",
                    str(args.hf_assets_path),
                    "--buckets",
                    args.buckets,
                    "--bucket-cp",
                    args.bucket_cp,
                    "--max-length",
                    str(args.max_length),
                    "--num-examples",
                    str(args.num_examples),
                    "--max-streamed-examples",
                    str(args.max_streamed_examples),
                    "--learning-rate",
                    "2e-5",
                ]
            )
            changed.buckets = ",".join(
                str(b) for b in train.parse_bucket_list(changed.buckets)
            )
            changed.bucket_cp = train._format_bucket_cp_map(
                train.parse_bucket_cp_map(changed.bucket_cp)
            )

            with pytest.raises(RuntimeError, match="learning_rate"):
                train.validate_resume_contract(changed, plan, manifest)

    def test_varlen_attention_is_rejected_when_any_bucket_uses_cp(self):
        with pytest.raises(ValueError, match="VarlenAttention"):
            train.validate_bucket_config(
                buckets=(32768, 65536),
                bucket_cp={32768: 2, 65536: 4},
                nproc_per_node=8,
                attention_backend="varlen",
            )

    def test_source_dataset_command_builds_pod_local_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "dataset"
            args = train.parse_args(
                [
                    "--dataset-path",
                    str(dataset_path),
                    "--source-dataset-revision",
                    "source-sha",
                    "--source-dataset-rows-per-shard",
                    "123",
                    "--source-dataset-build-batch-size",
                    "17",
                ]
            )
            command = train.build_source_dataset_command(args)

        assert "prepare_swehero_historical_one_rollout.py" in " ".join(command)
        assert "--dataset-id" in command
        assert command[command.index("--dataset-id") + 1] == train.SOURCE_DATASET_ID
        assert "--revision" in command
        assert command[command.index("--revision") + 1] == "source-sha"
        assert "--output-dir" in command
        assert command[command.index("--output-dir") + 1] == str(dataset_path)
        assert "--rows-per-shard" in command
        assert command[command.index("--rows-per-shard") + 1] == "123"
        assert "--batch-size" in command
        assert command[command.index("--batch-size") + 1] == "17"
        assert "--overwrite" not in command

    def test_ensure_training_dataset_does_not_implicitly_overwrite_nonempty_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "dataset"
            dataset_path.mkdir()
            (dataset_path / "notes.txt").write_text("not a parquet dataset")
            args = train.parse_args(
                [
                    "--dataset-path",
                    str(dataset_path),
                    "--build-dataset-if-missing",
                ]
            )

            with (
                patch.object(train.subprocess, "run") as run,
                pytest.raises(FileExistsError, match="rebuild-source-dataset"),
            ):
                train.ensure_training_dataset(args)

            run.assert_not_called()
            assert (dataset_path / "notes.txt").read_text() == "not a parquet dataset"

    def test_hf_asset_download_pins_model_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--hf-assets-path",
                    str(Path(tmp) / "hf" / "Qwen2.5-Coder-7B-Instruct"),
                    "--download-hf-assets",
                ]
            )

            with patch.object(train.subprocess, "run") as run:
                train.download_hf_assets_if_requested(args)

        command = run.call_args.args[0]
        assert "--repo_id" in command
        assert command[command.index("--repo_id") + 1] == train.MODEL_ID
        assert "--revision" in command
        assert command[command.index("--revision") + 1] == train.MODEL_REVISION

    def test_source_dataset_revision_pins_source_revision(self):
        args = train.parse_args(["--source-dataset-revision", "legacy-sha"])

        assert args.source_dataset_revision == "legacy-sha"

    def test_training_dataset_files_expect_hf_style_parquet_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            data_dir = dataset_dir / "data"
            data_dir.mkdir(parents=True)
            later = data_dir / "train-00001-of-00002.parquet"
            earlier = data_dir / "train-00000-of-00002.parquet"
            later.write_bytes(b"")
            earlier.write_bytes(b"")

            files = train._training_dataset_files(dataset_dir)

        assert files == [earlier, later]

    def test_encode_masks_user_system_and_tool_observations(self):
        example = {
            "trajectory": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "please fix it"},
                {
                    "role": "assistant",
                    "content": "RUN_TESTS",
                    "tool_calls": [
                        {"name": "execute_bash", "arguments": {"cmd": "pytest"}}
                    ],
                },
                {"role": "tool", "content": "secret failing output"},
                {"role": "assistant", "content": "DONE"},
            ]
        }

        encoded = train.encode_swehero_example(
            FakeTokenizer(),
            example,
            max_length=4096,
            min_trainable_tokens=1,
        )

        assert encoded is not None
        trainable_text = FakeTokenizer().decode(
            label for label in encoded["labels"] if label != train.IGNORE_INDEX
        )
        assert "RUN_TESTS" in trainable_text
        assert "execute_bash" in trainable_text
        assert "DONE" in trainable_text
        assert "<tool_call>" in trainable_text
        assert "<|im_end|>" in trainable_text
        assert "please fix it" not in trainable_text
        assert "system prompt" not in trainable_text
        assert "secret failing output" not in trainable_text
        assert "<|assistant|>" not in trainable_text
        assert "<|im_start|>assistant" not in trainable_text

    def test_batched_segment_tokenization_matches_legacy_per_segment_encoding(self):
        example = {
            "trajectory": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "please fix it"},
                {
                    "role": "assistant",
                    "content": "RUN_TESTS",
                    "tool_calls": [
                        {"name": "execute_bash", "arguments": {"cmd": "pytest"}}
                    ],
                },
                {"role": "tool", "content": "secret failing output"},
                {"role": "assistant", "content": "DONE"},
            ],
            "model_patch": "diff --git a/file.py b/file.py\n+fixed\n",
        }

        expected = self._legacy_encode_swehero_example(
            FakeTokenizer(),
            example,
            max_length=4096,
            min_trainable_tokens=1,
            include_model_patch=True,
        )
        batch_tokenizer = BatchFakeTokenizer()

        actual = train.encode_swehero_example(
            batch_tokenizer,
            example,
            max_length=4096,
            min_trainable_tokens=1,
            include_model_patch=True,
        )

        assert actual == expected
        assert batch_tokenizer.batch_calls > 0
        assert batch_tokenizer.encode_calls == 0

    def test_encode_rejects_long_examples_instead_of_truncating(self):
        example = {
            "trajectory": [
                {"role": "user", "content": "issue"},
                {"role": "assistant", "content": "x" * 100},
            ],
        }

        with pytest.raises(train.LongExampleError, match="exceeds --max-length"):
            train.encode_swehero_example(
                FakeTokenizer(),
                example,
                max_length=32,
                min_trainable_tokens=1,
            )

    def test_materialization_errors_on_long_examples_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "64",
                    "--max-length",
                    "64",
                    "--max-streamed-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "too-long",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "x" * 1000},
                ],
            }

            with pytest.raises(RuntimeError, match="would have been truncated"):
                self._materialize_with_fake_runtime(args, [example])

    def test_materialization_can_explicitly_skip_long_examples_with_manifest_signal(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                    "--long-example-policy",
                    "skip",
                ]
            )
            examples = [
                {
                    "instance_id": "too-long",
                    "trajectory": [
                        {"role": "user", "content": "issue"},
                        {"role": "assistant", "content": "x" * 1000},
                    ],
                },
                {
                    "instance_id": "short",
                    "trajectory": [
                        {"role": "user", "content": "issue"},
                        {"role": "assistant", "content": "OK"},
                    ],
                },
            ]

            manifest = self._materialize_with_fake_runtime(args, examples)

        assert manifest["long_example_policy"] == "skip"
        assert manifest["skipped"]["too_long_for_max_length"] == 1
        assert manifest["long_examples_sample"][0]["source_id"] == "too-long"
        assert manifest["num_usable_examples"] == 1
        data_provenance = manifest["data_provenance"]
        assert data_provenance["schema_version"] == train.DATA_PROVENANCE_SCHEMA_VERSION
        assert data_provenance["materialization"]["long_example_policy"] == "skip"
        assert data_provenance["streamed"]["source_ids"] == ["too-long", "short"]
        assert data_provenance["included"]["source_ids"] == ["short"]
        assert (
            data_provenance["skipped"]["by_reason"]["too_long_for_max_length"][
                "source_ids"
            ]
        ) == ["too-long"]
        assert data_provenance["buckets"]["256"]["source_ids"]["source_ids"] == [
            "short"
        ]
        assert data_provenance["buckets"]["256"]["record_count"] == 1

    def test_materialization_writes_self_verifying_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }

            manifest = self._materialize_with_fake_runtime(args, [example])
            loaded_manifest = train._load_manifest(args.out_dir)

            assert (
                manifest["materialized_data_schema_version"]
                == train.MATERIALIZED_DATA_SCHEMA_VERSION
            )
            assert manifest == loaded_manifest
            assert manifest["bucket_counts"] == {"256": 1}
            assert manifest["model_revision"] == args.model_revision
            assert manifest["model_assets"]["model_revision"] == args.model_revision
            assert (
                manifest["model_assets"]["schema_version"]
                == train.MODEL_ASSET_PROVENANCE_SCHEMA_VERSION
            )
            assert manifest["model_assets"]["file_count"] == 1
            assert (
                manifest["data_provenance"]["schema_version"]
                == train.DATA_PROVENANCE_SCHEMA_VERSION
            )
            assert manifest["data_provenance"]["included"]["source_ids"] == ["short"]
            assert manifest["bucket_curriculum"] == train.DEFAULT_BUCKET_CURRICULUM
            assert (
                (manifest["data_provenance"]["materialization"]["bucket_curriculum"])
                == train.DEFAULT_BUCKET_CURRICULUM
            )
            assert (
                manifest["data_provenance"]["buckets"]["256"]["integrity"]
                == manifest["bucket_file_integrity"]["256"]
            )
            integrity = manifest["bucket_file_integrity"]["256"]
            bucket_path = Path(manifest["bucket_files"]["256"])
            assert integrity["records"] == 1
            assert integrity == train._bucket_file_stats(bucket_path)

    def test_real_swehero_materialization_matches_preoptimization_goldens(self):
        self._require_real_materialization_dependencies()
        dataset_path = self._real_swehero_dataset_path()
        hf_assets_path = self._real_qwen_hf_assets_path()
        tokenizer_class = self._mini_qwen_tokenizer_class()
        golden_rows = [
            {
                "source_id": "numpy__numpy-9df514382c0b7c8fdaa979f66054285d69afee4d",
                "segment_sha256": "8660b514a8bffff089ba77761fea116fb422f5679bc9259c3a38ee2d9f125a54",
                "length": 28776,
                "bucket": 32768,
                "trainable_tokens": 10674,
                "input_ids_sha256": "78bd4ef5d172dda88f4761a9ee7032a46ada696cc49460507ec9686e9ff9a7f8",
                "labels_sha256": "5c2e5bbb7c7ec48e898052ce4c6ba50fb3bf2ddd4ce9c05d15b7e0036bc7a9d2",
            },
            {
                "source_id": "tornadoweb__tornado-eb61029aa268456890c68ed1565dcaac1fdafb4a",
                "segment_sha256": "b3396e337372a0a989382a88c6da3b7ca7d7a48138425dddccdf9de570f1b7ad",
                "length": 44256,
                "bucket": 65536,
                "trainable_tokens": 14581,
                "input_ids_sha256": "a46a70987ffc81765a8e2ce0ca36f6182676bd487cfd81100b75b117577fcbff",
                "labels_sha256": "bbfe5623f71b2734f3f355d69bb3e91d0a8a87474a507b6b8a59d37a899cdba9",
            },
            {
                "source_id": "tornadoweb__tornado-a48f65a42c3b3f0f2b21bcbce4da12c2d4419915",
                "segment_sha256": "ef9e3225ccabb4596ae28837b9569e73844def8e7cc848668715b839a8045505",
                "length": 42904,
                "bucket": 65536,
                "trainable_tokens": 15378,
                "input_ids_sha256": "8fec300205d2e572d460e8521bd885dab934037759a1056334a80177015aece5",
                "labels_sha256": "1f711e7f6705c03aac1ff17ce10dd77a4b1496e31d57e576a04a8508730c9888",
            },
            {
                "source_id": "pandas-dev__pandas-586f63bc1a2bdc14dd6e3b76dd2713541b10d4f6",
                "segment_sha256": "f40df8dba9a1503f1e8a20d4775a0de1050a46e150a860091a18137365f3ca63",
                "length": 29799,
                "bucket": 32768,
                "trainable_tokens": 9154,
                "input_ids_sha256": "c95cd26df696bda068fa6f06830e792f8df21902ae1044bb8a8b06bf4d2bb97b",
                "labels_sha256": "78a5987dabc41951beb8badf37864c44974372930d32a83a4f2fb23f348d0a6c",
            },
            {
                "source_id": "datalad__datalad-b7077e8b2024e1ac150268cc4dbd73a521ca46d2",
                "segment_sha256": "19408d315561f2f839ef94a99f3fa58fb93501e779a152706066b819cb584445",
                "length": 53219,
                "bucket": 65536,
                "trainable_tokens": 11835,
                "input_ids_sha256": "9a481733315526fdb1ce21a5c695980694de17514bf5134d31d54dc2e5e398fc",
                "labels_sha256": "472503c8de4936e38bbdec11c58dca1ccb8a86066cd315051dd91f903df5d1f5",
            },
            {
                "source_id": "pandas-dev__pandas-04e07d0b9825d6bd9a0640568d929a89fe0e9fa8",
                "segment_sha256": "e8f68c297b85f2ca3fc321f06dff1fb33b5837dd106ca790a6a8ea9b3f69c079",
                "length": 54782,
                "bucket": 65536,
                "trainable_tokens": 17402,
                "input_ids_sha256": "dc66c9e75716c18689ff0660e1af5a0620e8a570f7ff7302180a9379008bf84d",
                "labels_sha256": "a868b1615629b026aa4fe7b1605a7c00a670e51f145b04dd4a0b4ea362fb11d2",
            },
            {
                "source_id": "pandas-dev__pandas-3e4f000ea5fd05c85b2b0f45bc16b1943a0df555",
                "segment_sha256": "69dcc4e6bb7b86035d942f72915d9cffe1ce81caa3cf419a3a64b602fa6c8493",
                "length": 31473,
                "bucket": 32768,
                "trainable_tokens": 11891,
                "input_ids_sha256": "d2a107cc2fe5f681b7b44d0358c57c74e40bb252c59a07ea07ec4d13ce95147b",
                "labels_sha256": "de6be004fd2b4e670bfdd80f2b8901c0f24cb10c659c1e65eed42d306542a8fa",
            },
            {
                "source_id": "nedbat__coveragepy-f3a70c951e838e3cfab706b9a2d0459d783e5a4f",
                "segment_sha256": "ba704faa9ea4a47943f9cca9e1032a47730ba9492c757dbc03ee1ac20019a6b4",
                "length": 128798,
                "bucket": 131072,
                "trainable_tokens": 8437,
                "input_ids_sha256": "83ca86315ee2801aa44a3338ec1548f6e65c516dae5cfe0fa286f06eb66b5249",
                "labels_sha256": "412e9feff072925889821ca7a5219f38a361486b6600014d9f81fe388d7e5afa",
            },
        ]
        golden_manifest = {
            "bucket_counts": {
                "32768": 3,
                "65536": 4,
                "131072": 1,
            },
            "bucket_file_integrity": {
                "32768": {
                    "bytes": 1014056,
                    "records": 3,
                    "sha256": "8db8cb1d7b9640bebd786c162ba2957de4609686d7c2d25247b91c4372951b18",
                },
                "65536": {
                    "bytes": 2211965,
                    "records": 4,
                    "sha256": "345bcb1c943f4f6d3cf3cabf92aae7aec298ef0a08cd7473b8af7769ed144399",
                },
                "131072": {
                    "bytes": 1408019,
                    "records": 1,
                    "sha256": "1e83697c14421db9d3c2db7b6c3735961291def5d9de116f5a77dde5eb9ef09c",
                },
            },
            "length_histogram_rounded_to_1024": {
                "29696": 1,
                "30720": 1,
                "31744": 1,
                "43008": 1,
                "45056": 1,
                "53248": 1,
                "55296": 1,
                "129024": 1,
            },
            "num_usable_examples": 8,
            "streamed_examples_scanned": 8,
            "skipped": {},
            "included_ids_sha256": "9db47e477c9fb78517bda62612b65cbc4535528083f11ba45ab3daca73362051",
            "streamed_ids_sha256": "9db47e477c9fb78517bda62612b65cbc4535528083f11ba45ab3daca73362051",
        }

        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(dataset_path),
                    "--hf-assets-path",
                    str(hf_assets_path),
                    "--num-examples",
                    "8",
                    "--max-streamed-examples",
                    "64",
                    "--buckets",
                    "32768,65536,131072",
                    "--bucket-cp",
                    "32768:2,65536:4,131072:8",
                    "--max-length",
                    "131072",
                    "--long-example-policy",
                    "error",
                ]
            )
            with self._patch_torchtitan_tokenizer_module(tokenizer_class):
                manifest = train.materialize_training_buckets(args)

            tokenizer = tokenizer_class(str(hf_assets_path))
            examples = []
            for index, example in enumerate(train.load_training_dataset(args), start=1):
                source_id = train._example_id(example, index)
                segments = train.qwen_openhands_segments(example)
                encoded = train.encode_swehero_example(
                    tokenizer,
                    example,
                    max_length=args.max_length,
                    min_trainable_tokens=args.min_trainable_tokens,
                    include_model_patch=args.include_model_patch,
                )
                assert encoded is not None
                examples.append((source_id, segments, encoded))
                if len(examples) == len(golden_rows):
                    break

        for golden, (source_id, segments, encoded) in zip(golden_rows, examples):
            assert source_id == golden["source_id"]
            assert (
                hashlib.sha256(
                    json.dumps(segments, ensure_ascii=False).encode()
                ).hexdigest()
            ) == golden["segment_sha256"]
            assert encoded["length"] == golden["length"]
            assert (
                train.choose_bucket(
                    encoded["length"],
                    train.parse_bucket_list(args.buckets),
                )
            ) == golden["bucket"]
            assert encoded["trainable_tokens"] == golden["trainable_tokens"]
            assert (
                hashlib.sha256(json.dumps(encoded["input_ids"]).encode()).hexdigest()
                == golden["input_ids_sha256"]
            )
            assert (
                hashlib.sha256(json.dumps(encoded["labels"]).encode()).hexdigest()
                == golden["labels_sha256"]
            )

        assert manifest["bucket_counts"] == golden_manifest["bucket_counts"]
        assert (
            manifest["bucket_file_integrity"]
            == golden_manifest["bucket_file_integrity"]
        )
        assert (
            manifest["length_histogram_rounded_to_1024"]
            == golden_manifest["length_histogram_rounded_to_1024"]
        )
        assert manifest["num_usable_examples"] == golden_manifest["num_usable_examples"]
        assert (
            manifest["streamed_examples_scanned"]
            == golden_manifest["streamed_examples_scanned"]
        )
        assert manifest["skipped"] == golden_manifest["skipped"]
        assert (
            manifest["data_provenance"]["included"]["sha256"]
            == golden_manifest["included_ids_sha256"]
        )
        assert (
            manifest["data_provenance"]["streamed"]["sha256"]
            == golden_manifest["streamed_ids_sha256"]
        )

    def test_real_swehero_skip_materialization_matches_legacy_encoder(self):
        self._require_real_materialization_dependencies()
        dataset_path = self._real_swehero_dataset_path()
        hf_assets_path = self._real_qwen_hf_assets_path()
        tokenizer_class = self._mini_qwen_tokenizer_class()

        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(dataset_path),
                    "--hf-assets-path",
                    str(hf_assets_path),
                    "--num-examples",
                    "2",
                    "--max-streamed-examples",
                    "8",
                    "--buckets",
                    "30000",
                    "--bucket-cp",
                    "30000:1",
                    "--max-length",
                    "30000",
                    "--long-example-policy",
                    "skip",
                ]
            )
            tokenizer = tokenizer_class(str(hf_assets_path))
            expected = self._expected_materialization_from_legacy_encoder(
                args=args,
                tokenizer=tokenizer,
                examples=train.load_training_dataset(args),
            )
            with self._patch_torchtitan_tokenizer_module(tokenizer_class):
                manifest = train.materialize_training_buckets(args)
            actual_bucket_text = Path(manifest["bucket_files"]["30000"]).read_text()

        assert manifest["bucket_counts"] == expected["bucket_counts"]
        assert manifest["bucket_file_integrity"] == expected["bucket_file_integrity"]
        assert (
            manifest["length_histogram_rounded_to_1024"]
            == expected["length_histogram_rounded_to_1024"]
        )
        assert manifest["num_usable_examples"] == expected["num_usable_examples"]
        assert (
            manifest["streamed_examples_scanned"]
            == expected["streamed_examples_scanned"]
        )
        assert manifest["skipped"] == {"too_long_for_max_length": 2}
        assert manifest["skipped"] == expected["skipped"]
        assert (
            manifest["data_provenance"]["streamed"]["source_ids"]
            == expected["streamed_source_ids"]
        )
        assert (
            manifest["data_provenance"]["included"]["source_ids"]
            == expected["included_source_ids"]
        )
        assert (
            (
                manifest["data_provenance"]["skipped"]["by_reason"][
                    "too_long_for_max_length"
                ]["source_ids"]
            )
            == expected["skipped_source_ids_by_reason"]["too_long_for_max_length"]
        )
        assert actual_bucket_text == "".join(expected["bucket_lines"]["30000"])

    def test_synthetic_smoke_materialization_covers_configured_bucket_cp_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "128,256,512",
                    "--bucket-cp",
                    "128:1,256:2,512:4",
                    "--max-length",
                    "512",
                    "--nproc-per-node",
                    "4",
                    "--global-batch-size",
                    "4",
                    "--num-train-epochs",
                    "4",
                    "--smoke-synthetic-buckets",
                    "--smoke-synthetic-examples-per-bucket",
                    "2",
                ]
            )

            manifest = self._materialize_with_fake_runtime(args, synthetic=True)
            bucket_counts = train._bucket_counts_from_manifest(manifest)
            bucket_files = train._bucket_files_from_manifest(manifest)
            plan = train.build_bucket_plan(
                bucket_counts=bucket_counts,
                bucket_files=bucket_files,
                bucket_cp=train.parse_bucket_cp_map(args.bucket_cp),
                epochs=args.num_train_epochs,
                global_batch_size=args.global_batch_size,
                warmup_ratio=args.warmup_ratio,
            )

            assert manifest["smoke_synthetic_buckets"]
            assert manifest["smoke_synthetic_examples_per_bucket"] == 2
            assert manifest["bucket_curriculum"] == train.DEFAULT_BUCKET_CURRICULUM
            assert manifest["dataset_artifact"]["synthetic_smoke"]
            assert manifest["bucket_counts"] == {"128": 2, "256": 2, "512": 2}
            assert manifest["num_usable_examples"] == 6
            assert manifest["streamed_examples_scanned"] == 6
            assert manifest["data_provenance"]["materialization"][
                "smoke_synthetic_buckets"
            ]
            assert [stage.bucket for stage in plan.stages] == [128, 256, 512]
            assert [stage.cp_degree for stage in plan.stages] == [1, 2, 4]
            for bucket, path in bucket_files.items():
                rows = [
                    json.loads(line)
                    for line in path.read_text().splitlines()
                    if line.strip()
                ]
                assert len(rows) == 2
                assert all(row["bucket"] == bucket for row in rows)
                assert all(row["trainable_tokens"] == 1 for row in rows)

    def test_load_manifest_rejects_missing_model_asset_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }
            manifest = self._materialize_with_fake_runtime(args, [example])
            manifest.pop("model_assets")
            (args.out_dir / "data" / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

            with pytest.raises(RuntimeError, match="model_assets provenance"):
                train._load_manifest(args.out_dir)

    def test_load_manifest_rejects_missing_model_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }
            manifest = self._materialize_with_fake_runtime(args, [example])
            manifest.pop("model_revision")
            (args.out_dir / "data" / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

            with pytest.raises(RuntimeError, match="model_revision"):
                train._load_manifest(args.out_dir)

    def test_load_manifest_rejects_inconsistent_data_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }
            manifest = self._materialize_with_fake_runtime(args, [example])
            manifest["data_provenance"]["included"]["source_ids"] = ["other"]
            (args.out_dir / "data" / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

            with pytest.raises(RuntimeError, match="included.sha256"):
                train._load_manifest(args.out_dir)

    def test_main_writes_run_spec_for_dry_run_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            dataset_path = Path(tmp) / "dataset"
            hf_assets_path = Path(tmp) / "hf"
            args = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--dataset-path",
                    str(dataset_path),
                    "--hf-assets-path",
                    str(hf_assets_path),
                    "--buckets",
                    "256",
                    "--bucket-cp",
                    "256:1",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }
            self._materialize_with_fake_runtime(args, [example])

            with patch.dict(os.environ, {}, clear=True):
                with contextlib.redirect_stdout(io.StringIO()):
                    train.main(
                        [
                            "--out-dir",
                            str(out_dir),
                            "--dataset-path",
                            str(dataset_path),
                            "--hf-assets-path",
                            str(hf_assets_path),
                            "--buckets",
                            "256",
                            "--bucket-cp",
                            "256:1",
                            "--max-length",
                            "256",
                            "--num-examples",
                            "1",
                            "--skip-data-prep",
                            "--dry-run",
                        ]
                    )

            run_spec = json.loads((out_dir / train.RUN_SPEC_FILENAME).read_text())
            launcher_plan = json.loads((out_dir / "launcher_plan.json").read_text())

        assert run_spec["args"]["max_length"] == 256
        assert not (run_spec["args"]["production_mode"])
        assert run_spec["args"]["model_revision"] == train.MODEL_REVISION
        assert (
            run_spec["paper_alignment"]["kept"]["base_model_revision"]
            == train.MODEL_REVISION
        )
        assert run_spec["manifest"]["model_revision"] == train.MODEL_REVISION
        assert run_spec["args"]["bucket_curriculum"] == train.DEFAULT_BUCKET_CURRICULUM
        assert run_spec["plan"]["bucket_curriculum"] == train.DEFAULT_BUCKET_CURRICULUM
        assert run_spec["plan"]["distributed"]["nnodes"] == 1
        assert run_spec["plan"]["distributed"]["world_size"] == 8
        assert launcher_plan["bucket_curriculum"] == train.DEFAULT_BUCKET_CURRICULUM
        assert launcher_plan["distributed"]["nnodes"] == 1
        assert launcher_plan["distributed"]["world_size"] == 8
        assert run_spec["plan"]["total_steps"] == 1
        assert run_spec["manifest"]["data_provenance"]["included"]["source_ids"] == [
            "short"
        ]
        assert launcher_plan["run_spec"] == str(out_dir / train.RUN_SPEC_FILENAME)
        assert launcher_plan["launch_lock"] == str(train._launch_lock_path(out_dir))
        assert launcher_plan["wandb_identity"] == str(
            out_dir / train.WANDB_IDENTITY_FILENAME
        )
        assert run_spec["paths"]["runtime_metadata"] == str(
            out_dir / train.RUNTIME_METADATA_FILENAME
        )
        assert run_spec["paths"]["workspace_root"] == str(
            train._configured_workspace_root(train.parse_args([]))
        )
        assert launcher_plan["runtime_metadata"] == str(
            out_dir / train.RUNTIME_METADATA_FILENAME
        )
        assert run_spec["args"]["post_training_eval_command"] == ""
        assert run_spec["paths"]["post_training_eval_status"] == str(
            out_dir / train.POST_TRAINING_EVAL_STATUS_FILENAME
        )
        assert launcher_plan["post_training_eval_status"] == str(
            out_dir / train.POST_TRAINING_EVAL_STATUS_FILENAME
        )
        assert not (train._launch_lock_path(out_dir).exists())

    def test_main_rejects_duplicate_launch_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            args = train.parse_args(["--out-dir", str(out_dir)])
            lock_path = train._launch_lock_path(out_dir)

            with train.launch_lock(args):
                with pytest.raises(RuntimeError, match="Launch lock already exists"):
                    train.main(
                        [
                            "--out-dir",
                            str(out_dir),
                            "--smoke-synthetic-buckets",
                            "--max-length",
                            "256",
                            "--buckets",
                            "256",
                            "--bucket-cp",
                            "256:1",
                            "--global-batch-size",
                            "8",
                            "--num-train-epochs",
                            "1",
                            "--dry-run",
                        ]
                    )

            assert not (lock_path.exists())

    def test_main_writes_wandb_identity_for_dry_run_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            dataset_path = Path(tmp) / "dataset"
            hf_assets_path = Path(tmp) / "hf"
            args = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--dataset-path",
                    str(dataset_path),
                    "--hf-assets-path",
                    str(hf_assets_path),
                    "--buckets",
                    "256",
                    "--bucket-cp",
                    "256:1",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }
            self._materialize_with_fake_runtime(args, [example])

            with patch.dict(os.environ, {}, clear=True):
                with contextlib.redirect_stdout(io.StringIO()):
                    train.main(
                        [
                            "--out-dir",
                            str(out_dir),
                            "--dataset-path",
                            str(dataset_path),
                            "--hf-assets-path",
                            str(hf_assets_path),
                            "--buckets",
                            "256",
                            "--bucket-cp",
                            "256:1",
                            "--max-length",
                            "256",
                            "--num-examples",
                            "1",
                            "--skip-data-prep",
                            "--dry-run",
                            "--enable-wandb",
                            "--wandb-mode",
                            "offline",
                            "--wandb-run-name",
                            "dry-run-wandb",
                        ]
                    )

            identity = json.loads(
                (args.out_dir / train.WANDB_IDENTITY_FILENAME).read_text()
            )
            run_spec = json.loads((args.out_dir / train.RUN_SPEC_FILENAME).read_text())
            launcher_plan = json.loads(
                (args.out_dir / "launcher_plan.json").read_text()
            )

        assert identity["run_name"] == "dry-run-wandb"
        assert identity["generated_run_id"]
        assert identity["resume"] == "allow"
        assert run_spec["args"]["wandb_run_id"] == identity["run_id"]
        assert run_spec["args"]["wandb_resume"] == "allow"
        assert run_spec["paths"]["wandb_identity"] == str(
            args.out_dir / train.WANDB_IDENTITY_FILENAME
        )
        assert launcher_plan["wandb_identity"] == str(
            args.out_dir / train.WANDB_IDENTITY_FILENAME
        )

    def test_load_manifest_rejects_corrupt_bucket_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--num-examples",
                    "1",
                ]
            )
            example = {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }

            manifest = self._materialize_with_fake_runtime(args, [example])
            bucket_path = Path(manifest["bucket_files"]["256"])
            with bucket_path.open("a") as handle:
                handle.write('{"unexpected": true}\n')

            with pytest.raises(RuntimeError, match="record|sha256"):
                train._load_manifest(args.out_dir)

    def test_failed_materialization_does_not_publish_partial_data(self):
        def broken_examples():
            yield {
                "instance_id": "short",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--max-streamed-examples",
                    "2",
                ]
            )

            with pytest.raises(RuntimeError, match="boom"):
                self._materialize_with_fake_runtime(args, broken_examples())

            assert not ((args.out_dir / "data").exists())
            assert list(args.out_dir.glob(".data.tmp-*")) == []

    def test_failed_rematerialization_preserves_existing_data(self):
        def broken_examples():
            yield {
                "instance_id": "replacement",
                "trajectory": [
                    {"role": "user", "content": "different issue"},
                    {"role": "assistant", "content": "different answer"},
                ],
            }
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--dataset-path",
                    str(Path(tmp) / "dataset"),
                    "--hf-assets-path",
                    str(Path(tmp) / "hf"),
                    "--buckets",
                    "256",
                    "--max-length",
                    "256",
                    "--max-streamed-examples",
                    "2",
                ]
            )
            original_example = {
                "instance_id": "original",
                "trajectory": [
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "OK"},
                ],
            }
            original_manifest = self._materialize_with_fake_runtime(
                args,
                [original_example],
            )
            original_bucket_path = Path(original_manifest["bucket_files"]["256"])
            original_bucket_bytes = original_bucket_path.read_bytes()

            with pytest.raises(RuntimeError, match="boom"):
                self._materialize_with_fake_runtime(args, broken_examples())

            assert train._load_manifest(args.out_dir) == original_manifest
            assert original_bucket_path.read_bytes() == original_bucket_bytes
            assert list(args.out_dir.glob(".data.tmp-*")) == []

    def test_openhands_messages_render_as_qwen_chatml(self):
        example = {
            "trajectory": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "reported issue"},
                {
                    "role": "assistant",
                    "content": "assistant analysis",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "think",
                                "arguments": '{"thought": "consider options"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "environment output"},
            ],
        }

        rendered = "".join(text for text, _ in train.qwen_openhands_segments(example))

        assert "<|im_start|>system\nsystem prompt<|im_end|>\n" in rendered
        assert "<|im_start|>user\nreported issue<|im_end|>\n" in rendered
        assert (
            '<|im_start|>assistant\nassistant analysis\n<tool_call>\n{"name": "think", "arguments": "{\\"thought\\": \\"consider options\\"}"}\n</tool_call><|im_end|>\n'
            in rendered
        )
        assert (
            "<|im_start|>user\n<tool_response>\nenvironment output\n</tool_response><|im_end|>\n"
            in rendered
        )
        assert "<|system|>" not in rendered
        assert "<|assistant|>" not in rendered
        assert "<|tool_calls|>" not in rendered

    def test_tool_call_serialization_keeps_valid_json_payloads(self):
        text = train._qwen_tool_call_text(
            {
                "function": {
                    "name": 'quoted"tool',
                    "arguments": {"cmd": 'printf "hello"'},
                }
            }
        )
        payload = text.removeprefix("\n<tool_call>\n").removesuffix("\n</tool_call>")

        decoded = json.loads(payload)

        assert decoded["name"] == 'quoted"tool'
        assert decoded["arguments"] == {"cmd": 'printf "hello"'}

    def test_tool_call_serialization_preserves_argument_type(self):
        string_arguments = train._qwen_tool_call_text(
            {
                "function": {
                    "name": "think",
                    "arguments": '{"thought": "keep as string"}',
                }
            }
        )
        mapping_arguments = train._qwen_tool_call_text(
            {"name": "execute_bash", "arguments": {"cmd": "pytest -q"}}
        )

        assert (
            json.loads(
                string_arguments.removeprefix("\n<tool_call>\n").removesuffix(
                    "\n</tool_call>"
                )
            )
        ) == {"name": "think", "arguments": '{"thought": "keep as string"}'}
        assert (
            json.loads(
                mapping_arguments.removeprefix("\n<tool_call>\n").removesuffix(
                    "\n</tool_call>"
                )
            )
        ) == {"name": "execute_bash", "arguments": {"cmd": "pytest -q"}}

    def test_openhands_messages_match_hf_qwen_chat_template_when_available(self):
        hf_assets = Path(
            os.environ.get(
                "SWEHERO_TEST_HF_ASSETS_PATH",
                "/workspace/assets/hf/Qwen2.5-Coder-7B-Instruct",
            )
        )
        if not hf_assets.exists():
            pytest.skip("Qwen HF tokenizer assets are not available")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            pytest.skip(f"transformers is not available: {exc}")

        tokenizer = AutoTokenizer.from_pretrained(
            str(hf_assets),
            local_files_only=True,
        )
        cases = [
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "reported issue"},
                {
                    "role": "assistant",
                    "content": "assistant analysis",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "think",
                                "arguments": {"thought": "consider options"},
                            },
                        },
                        {
                            "name": "execute_bash",
                            "arguments": {"cmd": "pytest -q"},
                        },
                    ],
                },
                {"role": "tool", "content": "first observation"},
                {"role": "tool", "content": "second observation"},
                {"role": "assistant", "content": "done"},
            ],
            [
                {"role": "user", "content": "reported issue"},
                {"role": "assistant", "content": "done"},
            ],
        ]
        for messages in cases:
            with self.subTest(first_role=messages[0]["role"]):
                expected = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                rendered = "".join(
                    text
                    for text, _is_trainable in train.qwen_openhands_segments(
                        {"trajectory": messages}
                    )
                )

                assert rendered == expected

    def test_stage_environment_and_torchrun_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            args = train.parse_args(["--out-dir", str(out_dir)])
            stage = train.BucketStage(
                bucket=32768,
                cp_degree=2,
                example_count=4,
                steps=1,
                cumulative_steps=3,
                bucket_file=out_dir / "data" / "bucket_32768.jsonl",
            )
            env = train.build_stage_env(
                args,
                stage=stage,
                total_steps=5,
                warmup_steps=1,
                pad_token_id=151643,
            )
            command = train.build_torchrun_command(args)

        assert env["SWEHERO_BUCKET_CP"] == "2"
        assert env["SWEHERO_BUCKET_SEQ_LEN"] == "32768"
        assert env["SWEHERO_MODEL_REVISION"] == train.MODEL_REVISION
        assert env["SWEHERO_ENABLE_FP8"] == "1"
        assert env["SWEHERO_CUMULATIVE_STEPS"] == "3"
        assert env["SWEHERO_OPTIMIZER_IMPL"] == "foreach"
        assert env["SWEHERO_TRAINING_DTYPE"] == "float32"
        assert env["SWEHERO_MP_PARAM_DTYPE"] == "bfloat16"
        assert env["SWEHERO_MP_REDUCE_DTYPE"] == "bfloat16"
        assert env["SWEHERO_FSDP_RESHARD_AFTER_FORWARD"] == "never"
        assert env["SWEHERO_DETECT_ANOMALY"] == "0"
        assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
        assert env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"
        assert env["SWEHERO_FINAL_EXPORT_FOLDER"] == train.FINAL_MODEL_EXPORT_FOLDER
        assert env["SWEHERO_SAVE_FINAL_FULL_CHECKPOINT"] == "1"
        assert "-m" in command
        assert "torchtitan.train" in command
        assert "--module" in command
        assert "swehero" in command
        assert "--config" in command
        assert "qwen25_coder7b_direct_to_hero" in command
        assert "--nnodes" not in command
        assert "--node_rank" not in command
        assert "localhost:0" in command

    def test_multinode_torchrun_command_requires_explicit_rendezvous(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args(
                [
                    "--out-dir",
                    str(Path(tmp) / "run"),
                    "--nnodes",
                    "2",
                    "--node-rank",
                    "1",
                    "--rdzv-endpoint",
                    "train-master:29400",
                    "--rdzv-id",
                    "swehero-run",
                ]
            )
            buckets = train.parse_bucket_list(args.buckets)
            bucket_cp = train.parse_bucket_cp_map(args.bucket_cp)

            train.validate_launch_inputs(
                args,
                buckets=buckets,
                bucket_cp=bucket_cp,
            )
            command = train.build_torchrun_command(args)
            distributed = train._distributed_launch_summary(args)

        assert "--nnodes" in command
        assert "2" in command
        assert "--node_rank" in command
        assert "1" in command
        assert "--rdzv_endpoint" in command
        assert "train-master:29400" in command
        assert "--rdzv_id" in command
        assert "swehero-run" in command
        assert distributed["world_size"] == 16
        assert distributed["rdzv_id"] == "swehero-run"

    def test_launch_preflight_checks_executable_assets_and_bucket_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            hf_assets = Path(tmp) / "hf"
            self._write_preflight_hf_assets(hf_assets)
            bucket_file = out_dir / "data" / "bucket_256.jsonl"
            bucket_file.parent.mkdir(parents=True)
            bucket_file.write_text('{"input_ids":[1],"labels":[1]}\n')
            args = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--hf-assets-path",
                    str(hf_assets),
                    "--torchrun-bin",
                    sys.executable,
                ]
            )
            manifest = self._model_assets_manifest(args)
            stage = train.BucketStage(
                bucket=256,
                cp_degree=1,
                example_count=1,
                steps=1,
                cumulative_steps=1,
                bucket_file=bucket_file,
            )
            plan = train.BucketPlan((stage,), total_steps=1, warmup_steps=0)

            with patch.object(
                train,
                "validate_resource_preflights",
                return_value={"resource": "ok"},
            ):
                summary = train.validate_launch_preflight(args, plan, manifest)
            bucket_file.unlink()
            with (
                patch.object(
                    train,
                    "validate_resource_preflights",
                    return_value={"resource": "ok"},
                ),
                pytest.raises(RuntimeError, match="bucket file"),
            ):
                train.validate_launch_preflight(args, plan, manifest)

        assert summary["torchrun_bin"]["resolved"] == sys.executable
        assert summary["resources"] == {"resource": "ok"}
        assert summary["bucket_files"][0]["bucket"] == 256

    def test_production_launch_preflight_rejects_empty_rank_data_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            hf_assets = Path(tmp) / "hf"
            self._write_preflight_hf_assets(hf_assets)
            bucket_file = out_dir / "data" / "bucket_256.jsonl"
            bucket_file.parent.mkdir(parents=True)
            bucket_file.write_text('{"input_ids":[1],"labels":[1]}\n')
            args = train.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--hf-assets-path",
                    str(hf_assets),
                    "--torchrun-bin",
                    sys.executable,
                    "--production-mode",
                ]
            )
            manifest = self._model_assets_manifest(args)
            stage = train.BucketStage(
                bucket=256,
                cp_degree=1,
                example_count=1,
                steps=1,
                cumulative_steps=1,
                bucket_file=bucket_file,
            )
            plan = train.BucketPlan((stage,), total_steps=1, warmup_steps=0)

            with (
                patch.object(
                    train,
                    "validate_resource_preflights",
                    return_value={"resource": "ok"},
                ),
                pytest.raises(
                    RuntimeError, match="would leave data-parallel ranks empty"
                ),
            ):
                train.validate_launch_preflight(args, plan, manifest)

    def test_hf_logits_parity_command_uses_paper_yarn_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            hf_assets = Path(tmp) / "Qwen2.5-Coder-7B-Instruct"
            args = train.parse_args(
                ["--out-dir", str(out_dir), "--hf-assets-path", str(hf_assets)]
            )
            command = train.build_hf_logits_parity_command(args)

        assert "qwen_swehero_logits_parity.py" in " ".join(command)
        assert "--reference-context" in command
        assert command[command.index("--reference-context") + 1] == "paper-yarn-128k"
        assert "--reference-model-path" in command
        assert command[command.index("--reference-model-path") + 1] == str(hf_assets)
        assert "--hf-model-revision" in command
        assert command[command.index("--hf-model-revision") + 1] == train.MODEL_REVISION
        assert "--json-out" in command
        assert command[command.index("--json-out") + 1] == str(
            out_dir / "hf_logits_parity.json"
        )

    def test_dataparallel_mesh_dims_must_come_from_torch(self):
        repo_root = Path(__file__).resolve().parents[1]
        for relative_path in (
            "torchtitan/torchtitan/distributed/full_dtensor.py",
            "torchtitan/torchtitan/models/llama3/parallelize.py",
        ):
            source = (repo_root / relative_path).read_text()
            assert "DataParallelMeshDims" in source
            assert "class DataParallelMeshDims" not in source
            assert "Compatibility shim" not in source
            assert "except ImportError" not in source

    def test_torchtitan_rmsnorm_uses_upstream_forward(self):
        repo_root = Path(__file__).resolve().parents[1]
        source_path = repo_root / "torchtitan/torchtitan/models/common/rmsnorm.py"
        source = source_path.read_text()
        tree = ast.parse(source)
        rmsnorm_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RMSNorm"
        )
        method_names = {
            node.name
            for node in rmsnorm_class.body
            if isinstance(node, ast.FunctionDef)
        }

        assert "forward" not in method_names
        assert "weight.clone" not in source

    def test_pod_setup_uses_pinned_uv(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "scripts/setup_torchtitan_pod_venv.py").read_text()

        assert 'TORCHTITAN_POD_UV_VERSION = "0.11.16"' in source
        assert "UV_X86_64_UNKNOWN_LINUX_GNU_SHA256 =" in source
        assert "UV_VERSION override is not supported" in source
        assert "require_uv_version" in source
        assert 'UV_VERSION="${UV_VERSION:-' not in source
