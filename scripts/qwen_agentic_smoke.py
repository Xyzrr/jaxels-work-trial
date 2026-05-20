import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Coder-1.5B")
DATASET_ID = os.environ.get("DATASET_ID", "nvidia/SWE-Hero-openhands-trajectories")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/workspace/qwen-agentic-smoke"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "768"))
NUM_EXAMPLES = int(os.environ.get("NUM_EXAMPLES", "4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "12"))
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "jaxels")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "jaxels-midtraining")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", "qwen-agentic-smoke")


def load_env_file(path="/workspace/.env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def stringify_turn(turn):
    if isinstance(turn, dict):
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return f"<|{role}|>\n{content}"
    return json.dumps(turn, ensure_ascii=False)


def format_example(example):
    parts = []
    trajectory = example.get("trajectory") or example.get("messages") or []
    if isinstance(trajectory, list):
        parts.extend(stringify_turn(turn) for turn in trajectory)
    else:
        parts.append(str(trajectory))

    patch = example.get("model_patch")
    if patch:
        parts.append("<|final_patch|>\n" + str(patch))
    return "\n\n".join(parts)


def main():
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(DATASET_ID, split="train", streaming=True)
    texts = []
    for example in raw:
        text = format_example(example)
        if len(text) > 200:
            texts.append(text)
        if len(texts) >= NUM_EXAMPLES:
            break

    if not texts:
        raise RuntimeError("No usable training examples found")

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors=None,
    )
    tokenized["labels"] = [
        [tok if mask else -100 for tok, mask in zip(ids, attention)]
        for ids, attention in zip(tokenized["input_ids"], tokenized["attention_mask"])
    ]

    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(tokenized["input_ids"])

        def __getitem__(self, idx):
            return {k: torch.tensor(v[idx]) for k, v in tokenized.items()}

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        warmup_steps=0,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=["wandb"] if use_wandb else [],
        run_name=WANDB_RUN_NAME if use_wandb else None,
        do_train=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=TinyDataset())
    result = trainer.train()

    losses = [
        entry["loss"]
        for entry in trainer.state.log_history
        if "loss" in entry and isinstance(entry["loss"], float)
    ]
    summary = {
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "num_examples": len(texts),
        "max_length": MAX_LENGTH,
        "max_steps": MAX_STEPS,
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
