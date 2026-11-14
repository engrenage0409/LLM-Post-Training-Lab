"""Diagnose reward-model generalization and synthetic shortcut learning.

Evaluate trained RM on train/validation plus multiple wrong-answer styles.
Optionally compare to an independently random, untrained score head.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.toy import load
from src.reward_model.data import preference_conversation, pairwise_statistics


def make_negative_rows(rows, mode, seed=2026):
    """Create deterministic alternative negatives; preserve correct chosen."""
    rng = random.Random(seed)
    result = []
    for row in rows:
        answer = int(row["answer"])
        if mode == "original":
            rejected = str(row["rejected"])
        elif mode == "minus_one":
            rejected = str(answer - 1)
        elif mode == "plus_ten":
            rejected = str(answer + 10)
        elif mode == "random_error":
            rejected = str(answer + rng.choice([-25, -10, -2, -1, 1, 2, 10, 25]))
        elif mode == "fixed_zero":
            rejected = "0" if answer != 0 else "1"
        else:
            raise ValueError(f"Unknown negative mode: {mode}")
        result.append({
            "prompt": row["prompt"], "chosen": str(row["answer"]),
            "rejected": rejected,
        })
    return result


def score_text(model, tokenizer, device, messages):
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=256)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        return float(model(**inputs).logits.reshape(-1)[0].item())


def evaluate_rows(model, tokenizer, device, rows, mode):
    data = make_negative_rows(rows, mode)
    values = []
    examples = []
    for row in data:
        pair = preference_conversation(row)
        chosen = score_text(model, tokenizer, device, pair["chosen"])
        rejected = score_text(model, tokenizer, device, pair["rejected"])
        values.append((chosen, rejected))
        examples.append({
            "prompt": row["prompt"], "chosen": row["chosen"],
            "rejected": row["rejected"], "chosen_score": chosen,
            "rejected_score": rejected, "margin": chosen - rejected,
            "correct": chosen > rejected,
        })
    return {"metrics": pairwise_statistics(values), "examples": examples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="outputs/reward_model")
    parser.add_argument("--untrained", action="store_true",
                        help="Use independently random score head, no trained adapter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", default="data/generated/train.jsonl")
    parser.add_argument("--val", default="data/generated/val.jsonl")
    parser.add_argument("--output", default="outputs/reward_diagnostics.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model if args.untrained else args.adapter
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, dtype="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if not args.untrained:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    modes = ["original", "minus_one", "plus_ten", "random_error", "fixed_zero"]
    results = {
        "kind": "untrained_random_head" if args.untrained else "trained_adapter",
        "seed": args.seed, "base_model": args.model, "adapter": None if args.untrained else args.adapter,
        "note": "Same known arithmetic data distribution; exploratory shortcut tests",
        "results": {},
    }
    for split, path in (("train", args.train), ("validation", args.val)):
        rows = load(path, ("prompt", "chosen", "rejected", "answer"))
        results["results"][split] = {}
        for mode in modes:
            result = evaluate_rows(model, tokenizer, device, rows, mode)
            results["results"][split][mode] = result
            metrics = result["metrics"]
            print(f"{split:10s} {mode:13s} "
                  f"{metrics['preference_accuracy']:6.2%} "
                  f"margin={metrics['mean_margin']:+.4f}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
