"""Run a reproducible, leakage-checked six-model arithmetic benchmark.

Usage:
  python -m scripts.benchmark --samples 600 --prepare-only
  python -m scripts.benchmark --samples 600
  python -m scripts.benchmark --samples 60 --output-dir outputs/benchmark_smoke

Runs each model in a separate Python process to release GPU memory.
"""
import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from src.data.toy import load

MODELS = {
    "base": (),
    "sft": ("--adapter", "outputs/sft"),
    "dpo": ("--adapter", "outputs/dpo"),
    "grpo": ("--adapter", "outputs/grpo_chat_100"),
    "sft_dpo": ("--sft-adapter", "outputs/sft", "--adapter", "outputs/sft_dpo"),
    "sft_grpo": ("--sft-adapter", "outputs/sft", "--adapter", "outputs/sft_grpo_100"),
}
PATTERN = re.compile(r"^What is (\d+) \+ (\d+)\? Answer with only an integer\.$")


def pair_from_prompt(prompt):
    match = PATTERN.fullmatch(prompt)
    if not match:
        raise ValueError(f"Unexpected arithmetic prompt format: {prompt!r}")
    return tuple(sorted((int(match.group(1)), int(match.group(2)))))


def difficulty(a, b):
    if a + b >= 100:
        return "sum_ge_100"
    if a % 10 + b % 10 >= 10:
        return "ones_carry"
    return "no_carry"


def make_test_rows(train_path, val_path, count=600, seed=2026):
    if count <= 0:
        raise ValueError("--samples must be positive")
    seen = set()
    for path in (train_path, val_path):
        for row in load(path, ("prompt",)):
            seen.add(pair_from_prompt(row["prompt"]))

    # Treat a+b and b+a as the same problem, including across splits.
    buckets = {k: [] for k in ("no_carry", "ones_carry", "sum_ge_100")}
    for a in range(100):
        for b in range(a, 100):
            if (a, b) not in seen:
                buckets[difficulty(a, b)].append((a, b))
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)

    result = []
    categories = list(buckets)
    while len(result) < count and any(buckets.values()):
        progress = False
        for category in categories:
            if buckets[category] and len(result) < count:
                a, b = buckets[category].pop()
                result.append({
                    "prompt": f"What is {a} + {b}? Answer with only an integer.",
                    "answer": str(a + b),
                    "difficulty": category,
                })
                progress = True
        if not progress:
            break
    if len(result) != count:
        raise ValueError(f"Only {len(result)} non-overlapping test problems available; requested {count}")
    rng.shuffle(result)
    return result


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize(models, rows):
    expected_prompts = [x["prompt"] for x in rows]
    groups = {row["prompt"]: row["difficulty"] for row in rows}
    summary = {"total": len(rows), "models": {}, "comparisons_vs_sft": {}}
    by_model = {}
    for name, path in models.items():
        result = json.loads(path.read_text(encoding="utf-8"))
        predictions = result["predictions"]
        prompts = [x["prompt"] for x in predictions]
        if len(prompts) != len(set(prompts)) or set(prompts) != set(expected_prompts):
            raise ValueError(f"Predictions for {name} do not match benchmark test prompts")
        by_model[name] = {x["prompt"]: x for x in predictions}
        counts = Counter()
        correct = Counter()
        for p in expected_prompts:
            label = groups[p]
            counts[label] += 1
            correct[label] += int(bool(by_model[name][p]["correct"]))
        summary["models"][name] = {
            "accuracy": result["accuracy"],
            "correct": sum(correct.values()),
            "by_difficulty": {
                key: {"correct": correct[key], "total": counts[key],
                      "accuracy": correct[key] / counts[key]}
                for key in sorted(counts)
            },
        }
    baseline = by_model["sft"]
    for name, candidate in by_model.items():
        if name == "sft":
            continue
        improved, regressed = [], []
        for prompt in expected_prompts:
            before, after = baseline[prompt], candidate[prompt]
            record = {
                "prompt": prompt,
                "expected": before["expected"],
                "sft_prediction": before["predicted"],
                "other_prediction": after["predicted"],
            }
            if not before["correct"] and after["correct"]:
                improved.append(record)
            elif before["correct"] and not after["correct"]:
                regressed.append(record)
        summary["comparisons_vs_sft"][name] = {
            "improved_count": len(improved), "regressed_count": len(regressed),
            "improved": improved, "regressed": regressed,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train", default="data/generated/train.jsonl")
    parser.add_argument("--val", default="data/generated/val.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = make_test_rows(args.train, args.val, args.samples, args.seed)
    data_path = out / "test.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(out / "manifest.json", {
        "samples": len(rows), "seed": args.seed,
        "train": args.train, "val": args.val, "model": args.model,
        "difficulty_distribution": dict(Counter(x["difficulty"] for x in rows)),
        "note": "Arithmetic-only, not a general reasoning benchmark",
    })
    print(f"Prepared {len(rows)} non-overlapping arithmetic problems: {data_path}", flush=True)
    if args.prepare_only:
        return

    paths = {}
    for name, flags in MODELS.items():
        # Fail rather than silently evaluate a missing or wrong adapter.
        adapter_args = list(flags)
        for i, arg in enumerate(adapter_args):
            if arg in ("--adapter", "--sft-adapter"):
                path = Path(adapter_args[i + 1])
                if not (path / "adapter_config.json").is_file():
                    raise FileNotFoundError(f"{name}: missing adapter at {path}")
        output = out / f"{name}.json"
        command = [
            sys.executable, "-m", "scripts.evaluate",
            "--model", args.model, "--data", str(data_path),
            "--output", str(output), *adapter_args,
        ]
        print(f"Evaluating {name}...", flush=True)
        subprocess.run(command, check=True)
        paths[name] = output

    summary = summarize(paths, rows)
    write_json(out / "summary.json", summary)
    print("\nBenchmark results:")
    for name, result in summary["models"].items():
        print(f"  {name:10s} {result['accuracy']:.2%} ({result['correct']}/{args.samples})")
    for name, result in summary["comparisons_vs_sft"].items():
        print(f"  SFT -> {name}: +{result['improved_count']} / -{result['regressed_count']}")
    print(f"Detailed summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
