"""Check HF logits parity for the SWE-HERO Qwen2.5-Coder-7B loader.

The training entrypoint initializes TorchTitan from Hugging Face safetensors via
``Qwen25StateDictAdapter``. This script compares logits from that exact
TorchTitan load path against ``transformers.AutoModelForCausalLM`` on the same
weights.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
DEFAULT_HF_ASSETS_PATH = Path("/workspace/assets/hf/Qwen2.5-Coder-7B-Instruct")
PAPER_CONTEXT_LENGTH = 131_072
QWEN_NATIVE_CONTEXT_LENGTH = 32_768
QWEN_ROPE_THETA = 1_000_000.0
PAPER_YARN_FACTOR = PAPER_CONTEXT_LENGTH / QWEN_NATIVE_CONTEXT_LENGTH
REFERENCE_CONTEXTS = ("paper-yarn-128k", "standard-hf")
DEFAULT_PROMPT = (
    "Implement a Python function that returns the first duplicate item in a list."
)


@dataclass(frozen=True)
class ReferenceContext:
    max_position_embeddings: int
    rope_scaling: dict[str, Any] | None


def reference_context(context: str) -> ReferenceContext:
    if context == "standard-hf":
        return ReferenceContext(
            max_position_embeddings=QWEN_NATIVE_CONTEXT_LENGTH,
            rope_scaling=None,
        )
    if context == "paper-yarn-128k":
        return ReferenceContext(
            max_position_embeddings=PAPER_CONTEXT_LENGTH,
            rope_scaling={
                "factor": PAPER_YARN_FACTOR,
                "original_max_position_embeddings": QWEN_NATIVE_CONTEXT_LENGTH,
                "type": "yarn",
            },
        )
    raise ValueError(f"unknown reference context: {context}")


def patch_hf_config_dict(config: dict[str, Any], context: str) -> dict[str, Any]:
    """Return an HF config dict for the requested reference context."""
    patched = deepcopy(config)
    ref = reference_context(context)
    if context == "standard-hf":
        return patched

    patched["max_position_embeddings"] = ref.max_position_embeddings
    patched["sliding_window"] = ref.max_position_embeddings
    patched["rope_scaling"] = dict(ref.rope_scaling or {})
    return patched


def default_position_offsets(context: str, max_tokens: int) -> list[int]:
    ref = reference_context(context)
    offsets = [0, max(0, QWEN_NATIVE_CONTEXT_LENGTH - max_tokens)]
    if context == "paper-yarn-128k":
        offsets.extend(
            [
                PAPER_CONTEXT_LENGTH // 2,
                max(0, PAPER_CONTEXT_LENGTH - max_tokens),
            ]
        )
    return sorted(
        set(offset for offset in offsets if offset < ref.max_position_embeddings)
    )


def parse_int_csv(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError(f"offsets must be non-negative: {values}")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare logits from HF Qwen2.5-Coder-7B-Instruct against the "
            "TorchTitan SWE-HERO model/state-dict adapter."
        )
    )
    parser.add_argument("--hf-model-id", default=os.environ.get("MODEL_ID", MODEL_ID))
    parser.add_argument(
        "--hf-model-revision",
        default=(
            os.environ.get("MODEL_REVISION")
            or os.environ.get("HF_MODEL_REVISION")
            or MODEL_REVISION
        ),
        help=(
            "Exact Hugging Face model commit SHA used when the parity reference "
            "must resolve assets from the Hub."
        ),
    )
    parser.add_argument(
        "--hf-assets-path",
        type=Path,
        default=Path(os.environ.get("HF_ASSETS_PATH", DEFAULT_HF_ASSETS_PATH)),
        help="Local HF safetensors/config/tokenizer directory used by training.",
    )
    parser.add_argument(
        "--reference-model-path",
        type=Path,
        help=(
            "HF model path for the reference forward pass. Defaults to "
            "--hf-assets-path so the reference uses the same local weights."
        ),
    )
    parser.add_argument(
        "--reference-context",
        choices=REFERENCE_CONTEXTS,
        default=os.environ.get("HF_LOGITS_PARITY_REFERENCE", "paper-yarn-128k"),
        help=(
            "paper-yarn-128k matches the SWE-HERO paper's 32k->128k YaRN "
            "extension; standard-hf leaves the checked HF config unchanged."
        ),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt text to tokenize. Can be provided more than once.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("HF_LOGITS_PARITY_MAX_TOKENS", "16")),
        help="Truncate each prompt to this many tokens for cheap long-position checks.",
    )
    parser.add_argument(
        "--position-offsets",
        type=parse_int_csv,
        help=(
            "Comma-separated starting position IDs. Defaults cover short, near-32k, "
            "mid-128k, and near-128k positions for paper-yarn-128k."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default=os.environ.get("HF_LOGITS_PARITY_DTYPE", "float32"),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("HF_LOGITS_PARITY_DEVICE", "auto"),
        help="auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--hf-attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2", "auto"),
        default=os.environ.get("HF_LOGITS_PARITY_HF_ATTN", "eager"),
        help="Use eager by default to avoid fused-kernel noise in the reference.",
    )
    parser.add_argument(
        "--tt-attn-backend",
        choices=("sdpa", "flex", "flex_flash", "varlen"),
        default=os.environ.get("HF_LOGITS_PARITY_TT_ATTN", "sdpa"),
    )
    parser.add_argument(
        "--force-math-attention",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("HF_LOGITS_PARITY_FORCE_MATH", "1").lower()
        not in {"0", "false", "no", "off"},
        help="Force TorchTitan SDPA to the math backend for deterministic parity.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=float(os.environ.get("HF_LOGITS_PARITY_ATOL", "0.002")),
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=float(os.environ.get("HF_LOGITS_PARITY_RTOL", "0.0002")),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to write the full parity report JSON.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_paths() -> None:
    root = _repo_root()
    for path in (root / "torchtitan", root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _remote_revision_kwargs(path_or_id: str | Path, revision: str) -> dict[str, str]:
    if Path(path_or_id).exists():
        return {}
    return {"revision": revision}


def _torch_dtype(torch: Any, raw: str) -> Any:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[raw]


def _resolve_device(torch: Any, raw: str) -> Any:
    if raw == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(raw)


def _clear_device_cache(torch: Any) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()


def _apply_reference_to_hf_config(config: Any, context: str) -> dict[str, Any]:
    ref = reference_context(context)
    summary: dict[str, Any] = {"reference_context": context}

    if context == "standard-hf":
        summary["max_position_embeddings"] = getattr(
            config, "max_position_embeddings", None
        )
        summary["rope_scaling"] = getattr(config, "rope_scaling", None)
        summary["rope_parameters"] = getattr(config, "rope_parameters", None)
        return summary

    rope_scaling = dict(ref.rope_scaling or {})
    config.max_position_embeddings = ref.max_position_embeddings
    if hasattr(config, "sliding_window"):
        config.sliding_window = ref.max_position_embeddings

    # Transformers 4.x accepts rope_scaling; newer configs may expose
    # rope_parameters. Set both when possible so the intended HF reference is
    # unambiguous across runtime versions.
    config.rope_scaling = rope_scaling
    rope_parameters = {
        "rope_type": rope_scaling["type"],
        "factor": rope_scaling["factor"],
        "rope_theta": getattr(config, "rope_theta", QWEN_ROPE_THETA),
        "original_max_position_embeddings": rope_scaling[
            "original_max_position_embeddings"
        ],
    }
    if hasattr(config, "rope_parameters"):
        config.rope_parameters = rope_parameters

    summary["max_position_embeddings"] = ref.max_position_embeddings
    summary["rope_scaling"] = rope_scaling
    summary["rope_parameters"] = rope_parameters
    return summary


def _apply_reference_to_tt_config(model_config: Any, context: str) -> dict[str, Any]:
    ref = reference_context(context)
    rope = model_config.rope
    if context == "standard-hf":
        rope.max_seq_len = QWEN_NATIVE_CONTEXT_LENGTH
        rope.scaling = "none"
        rope.rope_factor = 1.0
        rope.original_seq_len = QWEN_NATIVE_CONTEXT_LENGTH
    else:
        rope.max_seq_len = ref.max_position_embeddings
        rope.scaling = "yarn"
        rope.rope_factor = PAPER_YARN_FACTOR
        rope.beta_fast = 32.0
        rope.beta_slow = 1.0
        rope.original_seq_len = QWEN_NATIVE_CONTEXT_LENGTH

    return {
        "max_seq_len": rope.max_seq_len,
        "scaling": rope.scaling,
        "rope_factor": rope.rope_factor,
        "beta_fast": rope.beta_fast,
        "beta_slow": rope.beta_slow,
        "original_seq_len": rope.original_seq_len,
    }


def _tokenize_prompts(args: argparse.Namespace, torch: Any) -> list[Any]:
    from transformers import AutoTokenizer

    tokenizer_path: str | Path
    if args.hf_assets_path.exists():
        tokenizer_path = args.hf_assets_path
    else:
        tokenizer_path = args.hf_model_id

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        **_remote_revision_kwargs(tokenizer_path, args.hf_model_revision),
    )
    prompts = args.prompt or [DEFAULT_PROMPT]
    encoded_prompts = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        if encoded.shape[1] > args.max_tokens:
            encoded = encoded[:, : args.max_tokens]
        if encoded.shape[1] == 0:
            raise ValueError("tokenized prompt is empty")
        encoded_prompts.append(encoded.to(dtype=torch.long))
    return encoded_prompts


def _validate_offsets(
    *,
    offsets: list[int],
    encoded_prompts: list[Any],
    max_position_embeddings: int,
) -> None:
    max_prompt_len = max(int(input_ids.shape[1]) for input_ids in encoded_prompts)
    bad = [
        offset
        for offset in offsets
        if offset + max_prompt_len > max_position_embeddings
    ]
    if bad:
        raise ValueError(
            "position offset plus prompt length exceeds reference context: "
            f"bad_offsets={bad}, max_prompt_len={max_prompt_len}, "
            f"max_position_embeddings={max_position_embeddings}"
        )


def _position_ids(torch: Any, input_ids: Any, offset: int, device: Any) -> Any:
    return torch.arange(
        offset,
        offset + input_ids.shape[1],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)


def _collect_hf_logits(
    args: argparse.Namespace,
    *,
    torch: Any,
    dtype: Any,
    device: Any,
    encoded_prompts: list[Any],
    offsets: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from transformers import AutoConfig, AutoModelForCausalLM

    reference_path = args.reference_model_path or args.hf_assets_path
    if not Path(reference_path).exists():
        reference_path = args.hf_model_id

    reference_revision_kwargs = _remote_revision_kwargs(
        reference_path,
        args.hf_model_revision,
    )
    config = AutoConfig.from_pretrained(
        reference_path,
        trust_remote_code=args.trust_remote_code,
        **reference_revision_kwargs,
    )
    config_summary = _apply_reference_to_hf_config(config, args.reference_context)
    model_kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        **reference_revision_kwargs,
    }
    if args.hf_attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.hf_attn_implementation

    model = AutoModelForCausalLM.from_pretrained(reference_path, **model_kwargs)
    model.to(device)
    model.eval()

    logits: dict[str, Any] = {}
    with torch.inference_mode():
        for prompt_index, input_ids in enumerate(encoded_prompts):
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            for offset in offsets:
                position_ids = _position_ids(torch, input_ids, offset, device)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                key = f"prompt_{prompt_index}:offset_{offset}"
                logits[key] = output.logits.detach().cpu()

    del model
    _clear_device_cache(torch)
    return logits, config_summary


def _load_torchtitan_model(
    args: argparse.Namespace,
    *,
    torch: Any,
    dtype: Any,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    _add_repo_paths()

    import torch.distributed.checkpoint as dcp
    from torch.nn.attention import SDPBackend
    from torchtitan.components.checkpoint import ModelWrapper
    from torchtitan.models.common.attention import ScaledDotProductAttention
    from torchtitan.models.qwen2_5 import model_registry

    if not args.hf_assets_path.exists():
        raise FileNotFoundError(
            f"{args.hf_assets_path} does not exist; run the training asset "
            "download first or pass --hf-assets-path to the local HF weights."
        )

    if args.force_math_attention and args.tt_attn_backend == "sdpa":
        ScaledDotProductAttention.sdpa_backends = [SDPBackend.MATH]

    model_spec = model_registry(
        "coder7b",
        attn_backend=args.tt_attn_backend,
        converters=None,
    )
    model_config = model_spec.model
    tt_config_summary = _apply_reference_to_tt_config(
        model_config,
        args.reference_context,
    )

    with torch.device("meta"):
        model = model_config.build()
    model.to_empty(device=device)
    model.init_states(buffer_device=device)
    model.to(dtype=dtype)
    model.eval()

    wrapper = ModelWrapper(model)
    state_dict = wrapper._get_state_dict()
    adapter = model_spec.state_dict_adapter(model_config, str(args.hf_assets_path))
    hf_state_dict = adapter.to_hf(state_dict)
    dcp.load(
        hf_state_dict,
        storage_reader=adapter.get_hf_storage_reader(str(args.hf_assets_path)),
    )
    wrapper.load_state_dict(adapter.from_hf(hf_state_dict))
    model.to(dtype=dtype)
    model.eval()

    return model, tt_config_summary


def _collect_tt_logits(
    args: argparse.Namespace,
    *,
    torch: Any,
    dtype: Any,
    device: Any,
    encoded_prompts: list[Any],
    offsets: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model, config_summary = _load_torchtitan_model(
        args,
        torch=torch,
        dtype=dtype,
        device=device,
    )

    logits: dict[str, Any] = {}
    with torch.inference_mode():
        for prompt_index, input_ids in enumerate(encoded_prompts):
            input_ids = input_ids.to(device)
            for offset in offsets:
                position_ids = _position_ids(torch, input_ids, offset, device)
                key = f"prompt_{prompt_index}:offset_{offset}"
                logits[key] = model(input_ids, positions=position_ids).detach().cpu()

    del model
    _clear_device_cache(torch)
    return logits, config_summary


def _compare_logits(
    *,
    torch: Any,
    hf_logits: dict[str, Any],
    tt_logits: dict[str, Any],
    atol: float,
    rtol: float,
) -> tuple[bool, list[dict[str, Any]]]:
    comparisons = []
    passed = True
    for key in sorted(hf_logits):
        hf = hf_logits[key].to(torch.float32)
        tt = tt_logits[key].to(torch.float32)
        diff = (hf - tt).abs()
        tolerance = atol + rtol * hf.abs()
        exceed = diff > tolerance
        allclose = not bool(exceed.any().item())
        passed = passed and allclose

        last_hf = hf[:, -1, :]
        last_tt = tt[:, -1, :]
        comparisons.append(
            {
                "case": key,
                "allclose": allclose,
                "max_abs_diff": float(diff.max().item()),
                "mean_abs_diff": float(diff.mean().item()),
                "num_exceeding_tolerance": int(exceed.sum().item()),
                "hf_last_token_argmax": int(last_hf.argmax(dim=-1)[0].item()),
                "tt_last_token_argmax": int(last_tt.argmax(dim=-1)[0].item()),
            }
        )
    return passed, comparisons


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")

    import torch

    dtype = _torch_dtype(torch, args.dtype)
    device = _resolve_device(torch, args.device)
    encoded_prompts = _tokenize_prompts(args, torch)
    offsets = args.position_offsets or default_position_offsets(
        args.reference_context,
        args.max_tokens,
    )
    ref = reference_context(args.reference_context)
    _validate_offsets(
        offsets=offsets,
        encoded_prompts=encoded_prompts,
        max_position_embeddings=ref.max_position_embeddings,
    )

    started_at = time.time()
    hf_logits, hf_config = _collect_hf_logits(
        args,
        torch=torch,
        dtype=dtype,
        device=device,
        encoded_prompts=encoded_prompts,
        offsets=offsets,
    )
    tt_logits, tt_config = _collect_tt_logits(
        args,
        torch=torch,
        dtype=dtype,
        device=device,
        encoded_prompts=encoded_prompts,
        offsets=offsets,
    )
    passed, comparisons = _compare_logits(
        torch=torch,
        hf_logits=hf_logits,
        tt_logits=tt_logits,
        atol=args.atol,
        rtol=args.rtol,
    )

    report = {
        "passed": passed,
        "model_id": args.hf_model_id,
        "model_revision": args.hf_model_revision,
        "hf_assets_path": str(args.hf_assets_path),
        "reference_model_path": str(args.reference_model_path or args.hf_assets_path),
        "reference_context": args.reference_context,
        "device": str(device),
        "dtype": args.dtype,
        "hf_attn_implementation": args.hf_attn_implementation,
        "tt_attn_backend": args.tt_attn_backend,
        "force_math_attention": args.force_math_attention,
        "atol": args.atol,
        "rtol": args.rtol,
        "position_offsets": offsets,
        "prompt_token_lengths": [int(ids.shape[1]) for ids in encoded_prompts],
        "hf_config": hf_config,
        "torchtitan_config": tt_config,
        "comparisons": comparisons,
        "elapsed_sec": time.time() - started_at,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    report = run(argv)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
