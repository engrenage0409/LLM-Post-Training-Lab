"""Render a concise markdown summary of a completed PPO comparison."""
import argparse
import json
from pathlib import Path


def mean_metrics(directory):
    file = Path(directory) / "train_metrics.jsonl"
    rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No training metrics at {file}")
    fields = [
        "mean_reward", "critic_target_mae", "raw_advantage_mean",
        "raw_advantage_std", "clip_fraction", "mean_sampled_log_ratio",
        "value_loss",
    ]
    return {key: sum(float(x[key]) for x in rows if key in x) /
            sum(key in x for x in rows) if any(key in x for x in rows) else None
            for key in fields}, len(rows)


def render_report(root, summary):
    lines = [
        "# LLM Post-Training Lab — PPO Final Comparison",
        "",
        "> Exploratory arithmetic study. One seed and one training budget do not establish algorithm superiority.",
        "",
        "## Protocol",
        "",
        f"- Training seed: {summary['protocol']['train_seed']}",
        f"- Steps: {summary['protocol']['steps']}",
        f"- Rollout batch: {summary['protocol']['rollout_batch']}",
        f"- PPO epochs: {summary['protocol']['ppo_epochs']}",
        f"- Max generated tokens: {summary['protocol']['max_new_tokens']}",
        f"- Final test: {summary['protocol']['samples']} unique held-out arithmetic prompts, excluding old development prompts and commuted counterparts",
        "",
        "## Evaluation",
        "",
        "| Model | Correct | Accuracy |",
        "|---|---:|---:|",
    ]
    for model, values in summary["accuracy"].items():
        lines.append(f"| {model} | {values['correct']}/{values['total']} | {values['accuracy']:.2%} |")
    lines.extend(["", "## Paired disagreements", ""])
    for key, counts in summary.items():
        if key.startswith("paired_"):
            lines.append(
                f"- **{key.removeprefix('paired_')}**: first only {counts['a_only']}, "
                f"second only {counts['b_only']}, both correct {counts['both_correct']}, "
                f"both wrong {counts['both_wrong']}"
            )
    lines.extend(["", "## PPO training diagnostics", "", "| Metric | Sequence-level | Token-level |", "|---|---:|---:|"])
    seq, seq_steps = mean_metrics(root / "sequence")
    token, tok_steps = mean_metrics(root / "token")
    for key in seq:
        first, second = seq[key], token[key]
        lines.append(f"| {key} | {first:.4f} | {second:.4f} |" if first is not None and second is not None else f"| {key} | n/a | n/a |")
    lines.extend([
        "", f"- Sequence steps: {seq_steps}; token steps: {tok_steps}.",
        "- Training rewards and critic targets are not identical definitions across PPO variants.",
        "- A positive or negative policy loss alone does not determine output quality.",
        "- The sampled log-ratio is not a guaranteed nonnegative exact KL divergence.",
        "",
        "## Limitations and next steps",
        "",
        "- One seed, one narrow synthetic addition task, no real preference-based RLHF evaluation.",
        "- V1 treats an entire response as one action and estimates value after the response; V2 uses prefix-state token values and GAE.",
        "- Multiple seeds, external datasets, and reliable learned reward modeling are future research extensions.",
        "- Freeze this final test after the first complete comparison; do not tune against it and still call it sealed.",
        "",
    ])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", default="outputs/ppo_final")
    args = p.parse_args()
    root = Path(args.directory)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    content = render_report(root, summary)
    output = root / "FINAL_REPORT.md"
    output.write_text(content, encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
