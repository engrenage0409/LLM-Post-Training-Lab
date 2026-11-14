"""Generate balanced mixed-negative preferences for Reward Model V2.

Keep the original toy dataset untouched. V2 uses five negative types per
source question, with +/- offsets on both sides of the correct answer.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.data.toy import load


OFFSETS = {
    "plus_one": 1,
    "minus_one": -1,
    "plus_ten": 10,
    "minus_ten": -10,
}


def make_mixed_pairs(rows, seed=2026):
    rng = random.Random(seed)
    pairs = []
    for i, row in enumerate(rows):
        correct = int(row["answer"])
        candidates = dict(OFFSETS)
        # Use a signed random error, balanced by row index.
        magnitude = rng.choice((2, 3, 5, 15, 20, 25))
        candidates["random_signed"] = magnitude if i % 2 == 0 else -magnitude
        for kind, offset in candidates.items():
            rejected = str(correct + offset)
            pairs.append({
                "prompt": str(row["prompt"]),
                "chosen": str(correct),
                "rejected": rejected,
                "answer": str(correct),
                "negative_type": kind,
            })
    rng.shuffle(pairs)
    return pairs


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/generated/train.jsonl")
    parser.add_argument("--val", default="data/generated/val.jsonl")
    parser.add_argument("--output-dir", default="data/reward_v2")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    train = load(args.train, ("prompt", "answer"))
    val = load(args.val, ("prompt", "answer"))
    train_prompts = {r["prompt"] for r in train}
    val_prompts = {r["prompt"] for r in val}
    if train_prompts & val_prompts:
        parser.error("Train and validation prompts overlap")
    out = Path(args.output_dir)
    results = {}
    for name, rows in (("train", train), ("val", val)):
        pairs = make_mixed_pairs(rows, args.seed)
        write_jsonl(out / f"{name}.jsonl", pairs)
        stats = Counter(row["negative_type"] for row in pairs)
        results[name] = {"source_questions": len(rows), "preference_pairs": len(pairs),
                         "negative_types": dict(sorted(stats.items()))}
    manifest = {
        "seed": args.seed,
        "train_source": args.train,
        "val_source": args.val,
        "details": results,
        "limitations": [
            "Same arithmetic question distribution as V1",
            "All negatives are constructed offsets, not model-generated mistakes",
            "Validation questions differ but validation negative types are seen in training",
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
