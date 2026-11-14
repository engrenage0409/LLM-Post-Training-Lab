"""Run matched-budget PPO variants and evaluate on a fresh disjoint arithmetic set.

The sealed set excludes train/val and the old 600-question development test.
Run once after committing the experimental protocol, not as a tuning loop.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from scripts.benchmark import difficulty, pair_from_prompt
from src.data.toy import load


def make_sealed_rows(train_path, val_path, old_benchmark_path, count=600, seed=91283):
    if count <= 0:
        raise ValueError("count must be positive")
    excluded = set()
    for path in (train_path, val_path, old_benchmark_path):
        for item in load(path, ("prompt",)):
            excluded.add(pair_from_prompt(item["prompt"]))
    candidates = [
        (a, b) for a in range(100) for b in range(a, 100)
        if (a, b) not in excluded
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if count > len(candidates):
        raise ValueError(f"Requested {count}, only {len(candidates)} eligible questions")
    # Stable mixed categories; no accidental overlap with historical dev questions.
    rows = []
    for a, b in candidates[:count]:
        rows.append({
            "prompt": f"What is {a} + {b}? Answer with only an integer.",
            "answer": str(a + b), "difficulty": difficulty(a, b),
        })
    return rows


def metrics_from_eval(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return {x["prompt"]: x for x in report["predictions"]}


def compare_predictions(path_a, path_b):
    a, b = metrics_from_eval(path_a), metrics_from_eval(path_b)
    if set(a) != set(b):
        raise ValueError("Evaluations must use identical prompts")
    counts = {"both_correct": 0, "both_wrong": 0, "a_only": 0, "b_only": 0}
    for prompt in a:
        ca, cb = bool(a[prompt]["correct"]), bool(b[prompt]["correct"])
        key = "both_correct" if ca and cb else "both_wrong" if not ca and not cb else "a_only" if ca else "b_only"
        counts[key] += 1
    return counts


def train_one(module, output, args):
    cmd = [
        sys.executable, "-m", module, "--sft-adapter", args.sft_adapter,
        "--data", args.train, "--steps", str(args.steps),
        "--rollout-batch", str(args.rollout_batch),
        "--ppo-epochs", str(args.ppo_epochs),
        "--max-new-tokens", str(args.max_new_tokens),
        "--seed", str(args.seed), "--output", str(output),
    ]
    if module == "scripts.train_ppo":
        cmd += ["--temperature", "1.0"]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def evaluate_one(adapter, sft, data, output):
    cmd = [
        sys.executable, "-m", "scripts.evaluate", "--sft-adapter", sft,
        "--adapter", str(adapter), "--data", str(data), "--output", str(output),
    ]
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft-adapter", default="outputs/sft")
    p.add_argument("--train", default="data/generated/train.jsonl")
    p.add_argument("--val", default="data/generated/val.jsonl")
    p.add_argument("--old-benchmark", default="outputs/benchmark/test.jsonl")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--rollout-batch", type=int, default=2)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-seed", type=int, default=91283)
    p.add_argument("--samples", type=int, default=600)
    p.add_argument("--output-dir", default="outputs/ppo_final")
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    if min(a.steps, a.rollout_batch, a.ppo_epochs, a.max_new_tokens, a.samples) <= 0:
        p.error("steps/rollout-batch/ppo-epochs/max-new-tokens/samples must be positive")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_test = out / "sealed_test.jsonl"
    if final_test.exists() and not a.prepare_only:
        raise FileExistsError(
            f"{final_test} already exists; use a NEW output-dir for a new protocol, "
            "or remove it only if this is an intentional fresh run."
        )
    rows = make_sealed_rows(a.train, a.val, a.old_benchmark, a.samples, a.test_seed)
    with final_test.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    protocol = {
        "description": "matched rollout budget, single-seed comparison; not isolated GAE ablation",
        "train_questions": a.train, "validation_questions": a.val,
        "excluded_dev_benchmark": a.old_benchmark, "sealed_test": str(final_test),
        "samples": a.samples, "test_seed": a.test_seed,
        "train_seed": a.seed, "steps": a.steps, "rollout_batch": a.rollout_batch,
        "ppo_epochs": a.ppo_epochs, "max_new_tokens": a.max_new_tokens,
        "policy_sampling": "temperature 1.0, top_p 1.0, top_k disabled",
        "interpretation": "Exploratory arithmetic only; sealed set should be inspected only after training",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Prepared {len(rows)} new sealed arithmetic questions.", flush=True)
    if a.prepare_only:
        print("Note: do not inspect/tune against sealed_test.jsonl before final comparison.")
        return
    if not (Path(a.sft_adapter) / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing SFT adapter: {a.sft_adapter}")

    seq = out / "sequence"
    token = out / "token"
    for name, module, dest in (
        ("sequence", "scripts.train_ppo", seq),
        ("token", "scripts.train_ppo_token", token),
    ):
        train_one(module, dest, a)

    # Only now touch the final dataset with policy inference.
    seq_eval, tok_eval, sft_eval = (out / f"{x}_eval.json" for x in ("sequence", "token", "sft"))
    evaluate_one(seq, a.sft_adapter, final_test, seq_eval)
    evaluate_one(token, a.sft_adapter, final_test, tok_eval)
    subprocess.run([
        sys.executable, "-m", "scripts.evaluate", "--adapter", a.sft_adapter,
        "--data", str(final_test), "--output", str(sft_eval)
    ], check=True)
    scores = {}
    for name, file in (("sequence", seq_eval), ("token", tok_eval), ("sft", sft_eval)):
        data = json.loads(file.read_text(encoding="utf-8"))
        scores[name] = {"accuracy": data["accuracy"], "total": data["samples"],
                        "correct": sum(int(bool(x["correct"])) for x in data["predictions"])}
    summary = {
        "protocol": protocol, "accuracy": scores,
        "paired_sequence_vs_token": compare_predictions(seq_eval, tok_eval),
        "paired_sequence_vs_sft": compare_predictions(seq_eval, sft_eval),
        "paired_token_vs_sft": compare_predictions(tok_eval, sft_eval),
        "note": "One run/seed only; no statistical superiority established",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["accuracy"], indent=2), flush=True)
    print(f"Saved {out / 'summary.json'}")


if __name__ == "__main__":
    main()
