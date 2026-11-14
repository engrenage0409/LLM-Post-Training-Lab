"""Evaluate pairwise preference accuracy of a trained reward adapter.

Usage:
  python -m scripts.evaluate_reward_model --adapter outputs/reward_model
"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.toy import load
from src.reward_model.data import preference_conversation, pairwise_statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="outputs/reward_model")
    parser.add_argument("--data", default="data/generated/val.jsonl")
    parser.add_argument("--output", default="outputs/reward_model_eval.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, dtype="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rows = load(args.data, ("prompt", "chosen", "rejected"))
    records = []
    with torch.inference_mode():
        for row in rows:
            pair = preference_conversation(row)
            scores = []
            for key in ("chosen", "rejected"):
                content = tokenizer.apply_chat_template(
                    pair[key], tokenize=False, add_generation_prompt=False
                )
                inputs = tokenizer(
                    content, return_tensors="pt", truncation=True, max_length=256
                ).to(device)
                score = model(**inputs).logits.reshape(-1)[0].item()
                scores.append(score)
            records.append({
                "prompt": row["prompt"],
                "chosen": row["chosen"], "rejected": row["rejected"],
                "chosen_score": scores[0], "rejected_score": scores[1],
                "margin": scores[0] - scores[1],
                "correct": scores[0] > scores[1],
            })

    metrics = pairwise_statistics(
        [(row["chosen_score"], row["rejected_score"]) for row in records]
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"metrics": metrics, "predictions": records}, indent=2, ensure_ascii=False
    ), encoding="utf-8")
    print(
        f"Preference accuracy: {metrics['preference_accuracy']:.2%} "
        f"on {metrics['pairs']} pairs; mean margin: {metrics['mean_margin']:.4f}"
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
