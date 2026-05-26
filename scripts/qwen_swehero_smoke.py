"""Run a paper-aligned Qwen Coder smoke-training job on SWE-HERO traces.

This intentionally keeps four prototype constraints:

* the 7B Qwen2.5-Coder-Instruct model only;
* a tiny streamed subset of ``nvidia/SWE-Hero-openhands-trajectories``;
* no SWE-ZERO warm-start stage;
* no SWE-bench evaluation.

Within those limits, defaults follow the paper where practical: three training
epochs, effective global batch size 32, cosine LR from 1e-5 toward 1e-8 with a
0.1 warmup ratio, assistant/action-only loss masking that excludes tool
observations, and the paper's 128k context length. YaRN is enabled automatically
above Qwen's native 32k context.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

# This script predates the preset-driven TorchTitan launcher and is kept as a
# quick local/pod smoke path. Its settings are still environment-driven so old
# command snippets continue to work; the production training path records these
# same choices through argparse presets instead.
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Coder-7B-Instruct")
DATASET_ID = os.environ.get("DATASET_ID", "nvidia/SWE-Hero-openhands-trajectories")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/workspace/qwen25-coder7b-swehero-smoke"))

# A transformer can only condition on a bounded number of tokens at once. The
# SWE-Hero direct-to-hero recipe uses a 128k-token context so long OpenHands
# traces can keep prior reasoning, tool calls, and tool observations available
# while later assistant actions are supervised.
PAPER_CONTEXT_LENGTH = 131_072

# Qwen2.5-Coder was released with a native 32k-token context. When this smoke
# runs at the paper 128k context, `maybe_enable_yarn` applies YaRN positional
# scaling so the model can represent token positions beyond 32k.
QWEN_NATIVE_CONTEXT_LENGTH = 32_768
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", str(PAPER_CONTEXT_LENGTH)))
NUM_EXAMPLES = int(os.environ.get("NUM_EXAMPLES", "2"))
MAX_STREAMED_EXAMPLES = int(os.environ.get("MAX_STREAMED_EXAMPLES", "200"))

# Supervised fine-tuning (SFT) updates the pretrained model to predict target
# assistant/action tokens from curated traces. These defaults mirror the paper
# recipe where a tiny smoke run can: three epochs, global batch 32, AdamW with a
# 1e-5 peak learning rate, cosine decay toward 1e-8, and 10% warmup.
NUM_TRAIN_EPOCHS = float(os.environ.get("NUM_TRAIN_EPOCHS", "3"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))

# `per_device_train_batch_size` is how many examples one process/GPU handles at
# once. `WORLD_SIZE` is the number of training processes. Gradient accumulation
# simulates a larger batch by adding gradients over several smaller forward/back
# passes before one optimizer update.
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(
    os.environ.get(
        "GRADIENT_ACCUMULATION_STEPS",
        str(
            max(
                1,
                math.ceil(
                    GLOBAL_BATCH_SIZE / (PER_DEVICE_TRAIN_BATCH_SIZE * WORLD_SIZE)
                ),
            )
        ),
    )
)
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5"))
MIN_LEARNING_RATE = float(os.environ.get("MIN_LEARNING_RATE", "1e-8"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.1"))
MIN_TRAINABLE_TOKENS = int(os.environ.get("MIN_TRAINABLE_TOKENS", "1"))

# PyTorch cross entropy ignores labels with value -100. We use that to keep
# prompt/tool-observation tokens visible to the model while excluding them from
# the supervised loss.
IGNORE_INDEX = -100

# Computing logits for a 128k sequence all at once can spike memory. The custom
# Trainer below projects hidden states through the LM head in chunks; this keeps
# the same mathematical loss while lowering peak activation/logit memory.
LOGIT_CHUNK_SIZE = int(os.environ.get("LOGIT_CHUNK_SIZE", "512"))

# The direct-to-hero target is assistant action generation. Training on final
# patches changes the task toward patch emission, so this stays an explicit
# ablation switch rather than part of the default smoke path.
INCLUDE_MODEL_PATCH = os.environ.get("INCLUDE_MODEL_PATCH", "0").lower() in {
    "1",
    "true",
    "yes",
}
ENABLE_YARN = os.environ.get("ENABLE_YARN", "1").lower() in {"1", "true", "yes"}
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "jaxels")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "jaxels-midtraining")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", "qwen25-coder7b-swehero-smoke")
ENV_FILE = os.environ.get("ENV_FILE", "/workspace/.env")


def _dotenv_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    for index, char in enumerate(value):
        if char == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value


def load_env_file(path: str = ENV_FILE, *, required: bool = False) -> bool:
    env_path = Path(path)
    if not env_path.exists():
        if required:
            raise FileNotFoundError(f"Requested env file does not exist: {env_path}")
        return False

    for line_number, raw_line in enumerate(env_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            if required:
                raise ValueError(
                    f"Invalid dotenv line in {env_path}:{line_number}: {raw_line!r}"
                )
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(
                f"Invalid empty dotenv key in {env_path}:{line_number}: {raw_line!r}"
            )
        os.environ.setdefault(key, _dotenv_value(value))
    return True


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def turn_segments(turn: object) -> list[tuple[str, bool]]:
    """Serialize one OpenHands turn into ``(text, is_trainable)`` segments.

    Only assistant content and assistant tool calls are trainable. System/user
    prompts and tool observations stay in the context but are masked out of the
    loss, matching the paper's emphasis on learning action generation rather
    than fitting execution outputs.
    """

    if not isinstance(turn, dict):
        return [(json.dumps(turn, ensure_ascii=False) + "\n", False)]

    role = _stringify(turn.get("role") or "unknown")
    is_assistant = role == "assistant"
    segments = [(f"<|{role}|>\n", False)]

    content = _stringify(turn.get("content"))
    if content:
        segments.append((content.rstrip("\n") + "\n", is_assistant))

    tool_calls = turn.get("tool_calls")
    if tool_calls:
        segments.append(("<|tool_calls|>\n", False))
        segments.append(
            (json.dumps(tool_calls, ensure_ascii=False) + "\n", is_assistant)
        )

    return segments


def example_segments(
    example: dict[str, object], *, include_model_patch: bool = INCLUDE_MODEL_PATCH
) -> list[tuple[str, bool]]:
    """Serialize one raw SWE-Hero row into linear text/loss-mask segments.

    Language models train on one token stream, not nested message dictionaries.
    This helper flattens either `trajectory` or `messages` while preserving a
    boolean that says whether each segment should become a supervised target.
    """

    segments: list[tuple[str, bool]] = []
    trajectory = example.get("trajectory") or example.get("messages") or []
    if isinstance(trajectory, list):
        for turn in trajectory:
            segments.extend(turn_segments(turn))
    else:
        segments.append((_stringify(trajectory) + "\n", False))

    patch = example.get("model_patch")
    if include_model_patch and patch:
        # Final patches are plausible targets, but they are a different behavior
        # from OpenHands action generation. Keeping this branch explicit prevents
        # a smoke experiment from quietly training on a different task.
        segments.append(("<|assistant_final_patch|>\n", False))
        segments.append((_stringify(patch) + "\n", True))

    return segments


def encode_example(
    tokenizer: Any,
    example: dict[str, object],
    *,
    max_length: int = MAX_LENGTH,
    include_model_patch: bool = INCLUDE_MODEL_PATCH,
) -> dict[str, list[int]] | None:
    """Convert one serialized SWE trace into causal-LM input IDs and labels.

    A causal language model learns next-token prediction: given token `n`, it is
    trained to predict token `n+1`. The Hugging Face causal-LM model performs
    that one-token shift internally, so this smoke script keeps input IDs and
    labels aligned and only masks labels that should not contribute to loss.
    """

    input_ids: list[int] = []
    labels: list[int] = []

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        # BOS marks the start of the sample. It helps preserve the base model's
        # expected sequence shape, but it is not an assistant action to imitate.
        input_ids.append(bos_token_id)
        labels.append(IGNORE_INDEX)

    for text, is_trainable in example_segments(
        example, include_model_patch=include_model_patch
    ):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids.extend(token_ids)
        # Non-trainable text remains part of the prompt context. Its labels are
        # IGNORE_INDEX so cross entropy skips those positions.
        labels.extend(token_ids if is_trainable else [IGNORE_INDEX] * len(token_ids))

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        input_ids.append(eos_token_id)
        # If the previous target was trainable, learning the end-of-sequence
        # marker teaches the model when to stop that assistant output.
        labels.append(
            eos_token_id if labels and labels[-1] != IGNORE_INDEX else IGNORE_INDEX
        )

    # This smoke script truncates instead of failing closed because it is meant
    # to find any runnable examples quickly. The production TorchTitan launcher
    # rejects over-context examples by default so real training does not silently
    # change the supervised target.
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]

    trainable_tokens = sum(label != IGNORE_INDEX for label in labels)
    if trainable_tokens < MIN_TRAINABLE_TOKENS:
        return None

    return {"input_ids": input_ids, "labels": labels}


def effective_batch_config() -> dict[str, int]:
    """Return the batch shape that controls optimizer-update frequency.

    The ML-relevant batch size is not just the number of examples on one GPU.
    Optimizer behavior depends on the effective global batch:
    per-device batch * number of training processes * accumulation steps.
    """

    samples_per_optimizer_step = (
        PER_DEVICE_TRAIN_BATCH_SIZE * WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS
    )
    return {
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "world_size": WORLD_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch_size": samples_per_optimizer_step,
    }


def build_cosine_with_min_lr_lambda(
    total_steps: int,
    *,
    learning_rate: float = LEARNING_RATE,
    min_learning_rate: float = MIN_LEARNING_RATE,
    warmup_ratio: float = WARMUP_RATIO,
):
    """Build the learning-rate multiplier used by the AdamW scheduler.

    Warmup ramps the LR up gradually at the beginning, which reduces the chance
    of destabilizing a pretrained transformer with a large first update. Cosine
    decay then lowers the LR smoothly toward a non-zero floor so later updates
    become smaller but training does not abruptly stop.
    """

    min_lr_ratio = min_learning_rate / learning_rate
    warmup_steps = (
        max(1, math.ceil(total_steps * warmup_ratio))
        if warmup_ratio > 0 and total_steps > 1
        else 0
    )

    def lr_lambda(current_step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps and current_step < warmup_steps:
            return max(min_lr_ratio, (current_step + 1) / warmup_steps)

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (current_step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def maybe_enable_yarn(
    config: Any,
    *,
    max_length: int = MAX_LENGTH,
    enable_yarn: bool = ENABLE_YARN,
) -> None:
    """Enable Qwen long-context positional scaling when the smoke exceeds 32k.

    Qwen uses RoPE positional encodings, so token IDs alone do not tell the model
    where a token appeared in the long trace. YaRN rescales those positional
    encodings from the native 32k window to the requested context length. Both
    `rope_scaling` and `rope_parameters` are written because different
    Transformers/Qwen code paths have used different config field names.
    """

    if not enable_yarn or max_length <= QWEN_NATIVE_CONTEXT_LENGTH:
        return

    yarn_factor = max_length / QWEN_NATIVE_CONTEXT_LENGTH
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = getattr(config, "rope_scaling", {}).get("rope_theta", 1_000_000.0)

    config.rope_scaling = {
        "factor": yarn_factor,
        "original_max_position_embeddings": QWEN_NATIVE_CONTEXT_LENGTH,
        "rope_theta": rope_theta,
        "type": "yarn",
    }
    config.rope_parameters = {
        "factor": yarn_factor,
        "original_max_position_embeddings": QWEN_NATIVE_CONTEXT_LENGTH,
        "rope_theta": rope_theta,
        "rope_type": "yarn",
    }
    config.max_position_embeddings = max(max_length, PAPER_CONTEXT_LENGTH)


def main() -> None:
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    load_env_file()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
        os.environ.setdefault("WANDB_RUN_NAME", WANDB_RUN_NAME)
    else:
        os.environ.setdefault("WANDB_DISABLED", "true")

    print(f"model={MODEL_ID}")
    print(f"dataset={DATASET_ID}")
    print(f"wandb={'enabled' if use_wandb else 'disabled'}")
    print(f"max_length={MAX_LENGTH}")
    print(f"effective_batch={effective_batch_config()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Decoder-only LMs often ship without a dedicated pad token because
        # generation handles one sequence at a time. Batching variable-length
        # training examples still needs a pad value, and reusing EOS is the
        # standard safe fallback because padding positions are masked from loss.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max(MAX_LENGTH, PAPER_CONTEXT_LENGTH)

    raw = load_dataset(DATASET_ID, split="train", streaming=True)
    encoded_examples = []
    streamed_examples = 0
    for example in raw:
        streamed_examples += 1
        encoded = encode_example(tokenizer, example)
        if encoded is not None:
            encoded_examples.append(encoded)
        if len(encoded_examples) >= NUM_EXAMPLES:
            break
        if streamed_examples >= MAX_STREAMED_EXAMPLES:
            break

    if not encoded_examples:
        raise RuntimeError(
            "No usable training examples found. Increase MAX_LENGTH or "
            "MAX_STREAMED_EXAMPLES if assistant/action tokens were truncated."
        )

    batch_config = effective_batch_config()
    samples_per_optimizer_step = batch_config["effective_global_batch_size"]
    # Trainer epochs are defined over `len(train_dataset)`. A tiny smoke subset
    # can be smaller than one effective global batch, so we repeat examples until
    # every optimizer step receives a complete accumulated batch.
    items_per_epoch = max(
        samples_per_optimizer_step,
        math.ceil(len(encoded_examples) / samples_per_optimizer_step)
        * samples_per_optimizer_step,
    )
    if MAX_STEPS > 0:
        items_per_epoch = max(items_per_epoch, samples_per_optimizer_step * MAX_STEPS)

    class TinyDataset(torch.utils.data.Dataset):
        """Repeat the small encoded subset to satisfy Trainer's epoch contract."""

        def __len__(self) -> int:
            return items_per_epoch

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return encoded_examples[idx % len(encoded_examples)]

    def collate(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        """Pad a batch while preserving the loss mask.

        `attention_mask` tells the transformer which positions are real tokens.
        The label tensor is padded with IGNORE_INDEX so padding never contributes
        to cross entropy.
        """

        max_batch_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full(
            (len(features), max_batch_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(features), max_batch_len), dtype=torch.long)
        labels = torch.full(
            (len(features), max_batch_len), IGNORE_INDEX, dtype=torch.long
        )

        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.tensor(
                feature["input_ids"], dtype=torch.long
            )
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(feature["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    maybe_enable_yarn(config)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to("cuda")
    # KV cache speeds up autoregressive generation, but training uses full
    # sequences and backpropagation. Disabling it avoids storing inference-only
    # cache tensors.
    model.config.use_cache = False
    # Gradient checkpointing discards intermediate activations during the forward
    # pass and recomputes them during backward. That costs extra compute but is
    # the practical memory tradeoff for long-context smoke runs.
    model.gradient_checkpointing_enable()

    steps_per_epoch = math.ceil(
        items_per_epoch
        / (PER_DEVICE_TRAIN_BATCH_SIZE * WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS)
    )
    total_steps = (
        MAX_STEPS if MAX_STEPS > 0 else math.ceil(steps_per_epoch * NUM_TRAIN_EPOCHS)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_cosine_with_min_lr_lambda(total_steps)
    )

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=["wandb"] if use_wandb else [],
        run_name=WANDB_RUN_NAME if use_wandb else None,
        do_train=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    class ChunkedCausalLMTrainer(Trainer):
        """Trainer variant that computes long-context LM-head loss in chunks.

        Hugging Face's default causal-LM loss projects every hidden state through
        the vocabulary-sized LM head at once. At 128k context, that temporary
        logits tensor can dominate memory. This keeps the transformer forward
        pass intact, but slices the final projection/loss computation.
        """

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch=None,
        ):
            labels = inputs.pop("labels")
            # `model.model` is the transformer body. Calling it directly returns
            # hidden states before the vocabulary projection, letting us apply
            # `lm_head` in LOGIT_CHUNK_SIZE slices below.
            outputs = model.model(**inputs, use_cache=False)
            hidden_states = outputs.last_hidden_state

            # Next-token prediction: hidden state at position i predicts the
            # label at position i+1. The final hidden state has no next token, so
            # it is dropped; the first label has no preceding hidden state, so it
            # is dropped.
            shifted_hidden_states = hidden_states[:, :-1, :].reshape(
                -1, hidden_states.shape[-1]
            )
            shifted_labels = labels[:, 1:].reshape(-1)

            loss_sum = shifted_hidden_states.new_zeros(())
            token_count = shifted_hidden_states.new_zeros(())
            for start in range(0, shifted_hidden_states.shape[0], LOGIT_CHUNK_SIZE):
                end = min(start + LOGIT_CHUNK_SIZE, shifted_hidden_states.shape[0])
                label_chunk = shifted_labels[start:end]
                valid_tokens = label_chunk.ne(IGNORE_INDEX).sum()
                if valid_tokens.item() == 0:
                    continue

                logits = model.lm_head(shifted_hidden_states[start:end])
                loss_sum = loss_sum + torch.nn.functional.cross_entropy(
                    logits.float(),
                    label_chunk,
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
                token_count = token_count + valid_tokens

            # Average only over supervised assistant/action tokens. Prompt,
            # observation, and padding labels were set to IGNORE_INDEX and are
            # excluded from both `loss_sum` and `token_count`.
            loss = loss_sum / token_count.clamp_min(1)
            if return_outputs:
                return loss, outputs
            return loss

    trainer = ChunkedCausalLMTrainer(
        model=model,
        args=args,
        train_dataset=TinyDataset(),
        data_collator=collate,
        optimizers=(optimizer, lr_scheduler),
    )
    result = trainer.train()

    losses = [
        entry["loss"]
        for entry in trainer.state.log_history
        if "loss" in entry and isinstance(entry["loss"], float)
    ]
    summary = {
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "paper_alignment": {
            "kept": {
                "base_model_family": "Qwen2.5-Coder-Instruct",
                "training_epochs": NUM_TRAIN_EPOCHS,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "learning_rate_schedule": "cosine",
                "peak_learning_rate": LEARNING_RATE,
                "minimum_learning_rate": MIN_LEARNING_RATE,
                "warmup_ratio": WARMUP_RATIO,
                "loss_masking": "assistant/action tokens only; tool observations masked",
                "context_length": PAPER_CONTEXT_LENGTH,
            },
            "intentional_deviations": [
                "7B model only",
                "tiny SWE-HERO subset only",
                "SWE-ZERO stage skipped",
                "evaluation skipped",
            ],
        },
        "num_unique_examples": len(encoded_examples),
        "streamed_examples_scanned": streamed_examples,
        "items_per_epoch": items_per_epoch,
        "max_length": MAX_LENGTH,
        "max_steps": MAX_STEPS if MAX_STEPS > 0 else None,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        **batch_config,
        "optimizer": "AdamW",
        "lr_scheduler": "cosine_with_min_lr",
        "loss_implementation": "chunked_causal_lm",
        "logit_chunk_size": LOGIT_CHUNK_SIZE,
        "losses": losses,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "train_runtime": result.metrics.get("train_runtime"),
        "wandb_entity": os.environ.get("WANDB_ENTITY") if use_wandb else None,
        "wandb_project": os.environ.get("WANDB_PROJECT") if use_wandb else None,
        "wandb_run_name": os.environ.get("WANDB_RUN_NAME") if use_wandb else None,
    }
    (OUT_DIR / "smoke_result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
