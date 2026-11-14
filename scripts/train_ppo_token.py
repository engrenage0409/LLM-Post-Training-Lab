"""Token-level PPO + GAE learning experiment for Qwen2.5-0.5B on 8GB GPU.

Run a one-step smoke test before longer training. This trains policy LoRA
and a small value head; rule-based correctness is the terminal reward.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.toy import load
from src.grpo.rewards import accuracy_reward, format_reward
from src.model_loading import load_sft_merged_model
from src.ppo.token_level import (
    gae_terminal, shaped_rewards, token_logprobs_and_values, token_ppo_losses,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--sft-adapter", default="outputs/sft")
    p.add_argument("--data", default="data/generated/train.jsonl")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--rollout-batch", type=int, default=2)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--value-learning-rate", type=float, default=1e-4)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--output", default="outputs/ppo_token")
    a = p.parse_args()

    if min(a.steps, a.rollout_batch, a.ppo_epochs, a.max_new_tokens) <= 0:
        p.error("Steps, rollout batch, PPO epochs and token budget must be positive")
    if a.learning_rate <= 0 or a.value_learning_rate <= 0:
        p.error("Learning rates must be positive")
    if not (0 < a.clip_epsilon < 1 and a.kl_coef >= 0):
        p.error("clip-epsilon must be in (0,1) and kl-coef >=0")
    if not (0 <= a.gamma <= 1 and 0 <= a.gae_lambda <= 1):
        p.error("gamma and gae-lambda must be in [0,1]")

    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load(a.data, ("prompt", "answer"))
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    backbone = (
        load_sft_merged_model(a.model, a.sft_adapter) if a.sft_adapter
        else AutoModelForCausalLM.from_pretrained(a.model, dtype="auto")
    )
    backbone.config.use_cache = False
    policy = get_peft_model(backbone.to(device), LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    ))
    critic = nn.Linear(policy.config.hidden_size, 1, dtype=torch.float32).to(device)
    nn.init.zeros_(critic.weight)
    nn.init.zeros_(critic.bias)
    actor_params = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": actor_params, "lr": a.learning_rate},
        {"params": list(critic.parameters()), "lr": a.value_learning_rate},
    ])

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "train_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    for step in range(1, a.steps + 1):
        policy.eval()
        critic.eval()
        rollouts = []
        for row in random.choices(rows, k=a.rollout_batch):
            text = tok.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            prompt_ids = tok(text, return_tensors="pt")["input_ids"].to(device)
            length = prompt_ids.size(1)
            with torch.no_grad():
                # Sampling at temperature=1 matches the raw-policy logprobs
                # used as the PPO behavior distribution; no top-k/top-p.
                sequence = policy.generate(
                    input_ids=prompt_ids, max_new_tokens=a.max_new_tokens,
                    do_sample=True, temperature=1.0, top_p=1.0,
                    pad_token_id=tok.eos_token_id,
                )
                completion = tok.decode(sequence[0, length:], skip_special_tokens=True)
                old_logps, old_values = token_logprobs_and_values(
                    policy, critic, sequence, length
                )
                with policy.disable_adapter():
                    ref_logps, _ = token_logprobs_and_values(
                        policy, critic, sequence, length
                    )
                correct = float(accuracy_reward([completion], [row["answer"]])[0])
                format_score = float(format_reward([completion])[0])
                reward = correct + format_score
                token_rewards, log_ratios = shaped_rewards(
                    old_logps, ref_logps, reward, a.kl_coef
                )
                advantages, returns = gae_terminal(
                    token_rewards, old_values, a.gamma, a.gae_lambda
                )
            rollouts.append({
                "ids": sequence.detach(),
                "prompt_length": length,
                "old_logps": old_logps.detach(),
                "old_values": old_values.detach(),
                "advantages": advantages.detach(),
                "returns": returns.detach(),
                "reward": reward,
                "correct": correct,
                "sampled_log_ratio": log_ratios.sum().item(),
                "num_tokens": int(old_logps.numel()),
                "completion": completion,
            })

        # Normalize across all action tokens, not independently by response.
        combined = torch.cat([x["advantages"] for x in rollouts])
        mean = combined.mean()
        std = combined.std(unbiased=False)
        if combined.numel() > 1 and std.item() > 1e-6:
            for rollout in rollouts:
                rollout["advantages"] = (rollout["advantages"] - mean) / (std + 1e-8)

        last = {"policy_loss": 0.0, "value_loss": 0.0, "clip_fraction": 0.0}
        total_tokens = sum(x["num_tokens"] for x in rollouts)
        for epoch in range(a.ppo_epochs):
            policy.train()
            critic.train()
            optimizer.zero_grad(set_to_none=True)
            for rollout in rollouts:
                new_logps, new_values = token_logprobs_and_values(
                    policy, critic, rollout["ids"], rollout["prompt_length"]
                )
                loss, policy_loss, value_loss, clip = token_ppo_losses(
                    new_logps, rollout["old_logps"], rollout["advantages"],
                    new_values, rollout["returns"], a.clip_epsilon
                )
                weight = rollout["num_tokens"] / total_tokens
                (loss * weight).backward()
                last["policy_loss"] += policy_loss.detach().item() * weight / a.ppo_epochs
                last["value_loss"] += value_loss.detach().item() * weight / a.ppo_epochs
                last["clip_fraction"] += clip.detach().item() * weight / a.ppo_epochs
            torch.nn.utils.clip_grad_norm_(
                actor_params + list(critic.parameters()), max_norm=1.0
            )
            optimizer.step()

        record = {
            "step": step,
            "mean_reward": sum(x["reward"] for x in rollouts) / len(rollouts),
            "rollout_accuracy": sum(x["correct"] for x in rollouts) / len(rollouts),
            "mean_sampled_log_ratio": sum(x["sampled_log_ratio"] for x in rollouts) / len(rollouts),
            "sampled_tokens": total_tokens,
            **last,
            "responses": [
                {"completion": x["completion"], "reward": x["reward"],
                 "num_tokens": x["num_tokens"]} for x in rollouts
            ],
        }
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"step={step}/{a.steps} accuracy={record['rollout_accuracy']:.1%} "
            f"reward={record['mean_reward']:.3f} "
            f"policy={last['policy_loss']:.4f} value={last['value_loss']:.4f} "
            f"clip={last['clip_fraction']:.2%} tokens={total_tokens}",
            flush=True,
        )

    policy.save_pretrained(out)
    tok.save_pretrained(out)
    torch.save(critic.cpu().state_dict(), out / "value_head.pt")
    (out / "experiment.json").write_text(json.dumps({
        "algorithm": "educational token-level PPO with GAE",
        "base_model": a.model, "sft_adapter": a.sft_adapter,
        "data": a.data, "steps": a.steps, "rollout_batch": a.rollout_batch,
        "ppo_epochs": a.ppo_epochs, "max_new_tokens": a.max_new_tokens,
        "learning_rate": a.learning_rate, "value_learning_rate": a.value_learning_rate,
        "clip_epsilon": a.clip_epsilon, "kl_coef": a.kl_coef,
        "gamma": a.gamma, "gae_lambda": a.gae_lambda,
        "value_definition": "Prefix hidden state before each sampled token",
        "reward_definition": "Terminal correctness + format with per-token sampled KL shaping",
        "limitations": "No model-based reward, no adaptive KL or distributed PPO",
    }, indent=2), encoding="utf-8")
    print(f"Saved token PPO adapter, critic and logs to {out}")


if __name__ == "__main__":
    main()
