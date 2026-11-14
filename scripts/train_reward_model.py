"""Train a small LoRA reward model on chosen/rejected pairs.

Run: python -m scripts.train_reward_model --steps 50
"""
import argparse
import json
from pathlib import Path
from collections import Counter

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

from src.data.toy import load
from src.reward_model.data import preference_conversation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--train", default="data/generated/train.jsonl")
    p.add_argument("--val", default="data/generated/val.jsonl")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--output", default="outputs/reward_model")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    a = p.parse_args()
    if a.steps <= 0 or a.learning_rate <= 0:
        p.error("--steps and --learning-rate must be positive")

    train_rows = load(a.train, ("prompt", "chosen", "rejected"))
    val_rows = load(a.val, ("prompt", "chosen", "rejected"))
    if set(x["prompt"] for x in train_rows) & set(x["prompt"] for x in val_rows):
        p.error("Train/validation preferences overlap by prompt")
    train_ds = Dataset.from_list([preference_conversation(row) for row in train_rows])
    val_ds = Dataset.from_list([preference_conversation(row) for row in val_rows])

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=1, dtype="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    config = RewardConfig(
        output_dir=a.output, max_steps=a.steps, max_length=256,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=4, gradient_checkpointing=True,
        learning_rate=a.learning_rate,
        logging_steps=1, save_strategy="no", report_to="none", seed=42,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["score"], task_type="SEQ_CLS",
    )
    trainer = RewardTrainer(
        model=model, args=config, train_dataset=train_ds,
        eval_dataset=val_ds, processing_class=tokenizer, peft_config=lora,
    )
    trainer.train()
    trainer.save_model(a.output)
    tokenizer.save_pretrained(a.output)
    out = Path(a.output)
    (out / "experiment.json").write_text(json.dumps({
        "base_model": a.model, "train_pairs": len(train_rows),
        "validation_pairs": len(val_rows), "steps": a.steps,
        "learning_rate": a.learning_rate,
        "train_file": a.train, "validation_file": a.val,
        "negative_type_counts": dict(Counter(r.get("negative_type", "unspecified") for r in train_rows)),
        "method": "pairwise reward modeling with LoRA",
        "warning": "Synthetic arithmetic preferences; not evidence of human alignment",
    }, indent=2), encoding="utf-8")
    print(f"Saved reward-model LoRA adapter and tokenizer to {out}")


if __name__ == "__main__":
    main()
