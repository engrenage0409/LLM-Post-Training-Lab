"""Educational low-memory sequence-level PPO on deterministic arithmetic rewards.

Policy: SFT-merged Qwen + new LoRA.
Reference: same frozen SFT-merged backbone with PPO LoRA disabled.
Critic: small trainable scalar head on the final-token hidden state.
Reward: exact arithmetic correctness (+ optional format reward).

Each full completion is a *single action* with a terminal reward. This is
not TRL's full token-level PPO/GAE trainer.
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
from src.ppo.objectives import ppo_losses, terminal_advantage


def logprob_and_value(policy, value_head, full_ids, prompt_length):
    """Return summed completion log probability and terminal-state value."""
    result = policy(input_ids=full_ids, output_hidden_states=True, use_cache=False)
    # Predict token t from logit position (t-1).
    token_logits = result.logits[:, prompt_length - 1:-1, :]
    target_ids = full_ids[:, prompt_length:]
    selected = token_logits.float().log_softmax(dim=-1).gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)
    logp = selected.sum(dim=-1)
    final_hidden = result.hidden_states[-1][:, -1, :].float()
    value = value_head(final_hidden).squeeze(-1)
    return logp, value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--sft-adapter", default="outputs/sft")
    p.add_argument("--data", default="data/generated/train.jsonl")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--rollout-batch", type=int, default=2)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--value-learning-rate", type=float, default=1e-4)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/ppo")
    args = p.parse_args()
    if min(args.steps, args.rollout_batch, args.ppo_epochs, args.max_new_tokens) <= 0:
        p.error("steps, rollout-batch, ppo-epochs and max-new-tokens must be positive")
    if args.temperature <= 0 or args.learning_rate <= 0 or args.value_learning_rate <= 0:
        p.error("temperature and learning rates must be positive")
    if not 0 < args.clip_epsilon < 1 or args.kl_coef < 0:
        p.error("clip-epsilon must be in (0,1); kl-coef must be nonnegative")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load(args.data, ("prompt", "answer"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.sft_adapter:
        backbone = load_sft_merged_model(args.model, args.sft_adapter)
    else:
        backbone = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")
    backbone.config.use_cache = False
    backbone = backbone.to(device)
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
    )
    policy = get_peft_model(backbone, lora)
    value_head = nn.Linear(policy.config.hidden_size, 1, dtype=torch.float32).to(device)
    nn.init.zeros_(value_head.weight)
    nn.init.zeros_(value_head.bias)

    trainable_policy = [x for x in policy.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": trainable_policy, "lr": args.learning_rate},
        {"params": list(value_head.parameters()), "lr": args.value_learning_rate},
    ])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_metrics.jsonl"
    # Initializing a new log prevents appending mixed experimental runs.
    log_path.write_text("", encoding="utf-8")

    for step in range(1, args.steps + 1):
        selected_rows = random.choices(rows, k=args.rollout_batch)
        rollouts = []
        policy.eval()
        value_head.eval()
        for row in selected_rows:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True
            )
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_ids = encoded["input_ids"]
            prompt_length = prompt_ids.size(1)
            with torch.no_grad():
                generated = policy.generate(
                    input_ids=prompt_ids, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                    attention_mask=encoded["attention_mask"],
                    pad_token_id=tokenizer.eos_token_id,
                )
                completion = tokenizer.decode(
                    generated[0, prompt_length:], skip_special_tokens=True
                )
                old_logp, old_value = logprob_and_value(
                    policy, value_head, generated, prompt_length
                )
                # The same backbone without PPO LoRA is the frozen reference.
                # Since the starting backbone had SFT merged, reference = SFT.
                with policy.disable_adapter():
                    ref_logp, _ = logprob_and_value(
                        policy, value_head, generated, prompt_length
                    )
                correctness = float(accuracy_reward([completion], [row["answer"]])[0])
                formatting = float(format_reward([completion])[0])
                reward = torch.tensor(
                    [correctness + formatting], device=device, dtype=torch.float32
                )
                target, advantage, sampled_ratio = terminal_advantage(
                    reward, old_logp, ref_logp, old_value, args.kl_coef
                )
            rollouts.append({
                "full_ids": generated.detach(),
                "prompt_length": prompt_length,
                "old_logp": old_logp.detach(),
                "ref_logp": ref_logp.detach(),
                "advantage": advantage.detach(),
                "target": target.detach(),
                "reward": reward.item(),
                "correct": correctness,
                "sampled_log_ratio": sampled_ratio.item(),
                "completion": completion,
            })

        raw_advantages = torch.cat([r["advantage"] for r in rollouts])
        # Raw unnormalized advantage and value error are recorded for diagnostics.
        raw_value_errors = raw_advantages
        # Avoid overly large updates from the first few random critic estimates.
        advantages = torch.cat([r["advantage"] for r in rollouts])
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)
        # For small rollout batches, keep advantages centered only when
        # their spread is nonzero; otherwise preserve a learning signal.
        if adv_std.item() > 1e-6 and args.rollout_batch > 1:
            for r in rollouts:
                r["advantage"] = (r["advantage"] - adv_mean) / (adv_std + 1e-8)

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_clip_fraction = 0.0
        for _ in range(args.ppo_epochs):
            policy.train()
            value_head.train()
            optimizer.zero_grad(set_to_none=True)
            for rollout in rollouts:
                new_logp, new_value = logprob_and_value(
                    policy, value_head, rollout["full_ids"], rollout["prompt_length"]
                )
                loss, ploss, vloss, clip_fraction = ppo_losses(
                    new_logp, rollout["old_logp"], rollout["advantage"],
                    new_value, rollout["target"], args.clip_epsilon
                )
                (loss / len(rollouts)).backward()
                last_policy_loss += ploss.detach().item() / (args.ppo_epochs * len(rollouts))
                last_value_loss += vloss.detach().item() / (args.ppo_epochs * len(rollouts))
                last_clip_fraction += clip_fraction.detach().item() / (args.ppo_epochs * len(rollouts))
            torch.nn.utils.clip_grad_norm_(
                trainable_policy + list(value_head.parameters()), max_norm=1.0
            )
            optimizer.step()

        record = {
            "step": step,
            "raw_advantage_mean": raw_advantages.mean().item(),
            "raw_advantage_std": raw_advantages.std(unbiased=False).item(),
            "critic_target_mae": raw_value_errors.abs().mean().item(),
            "mean_reward": sum(r["reward"] for r in rollouts) / len(rollouts),
            "exact_accuracy": sum(r["correct"] for r in rollouts) / len(rollouts),
            "mean_sampled_log_ratio": sum(r["sampled_log_ratio"] for r in rollouts) / len(rollouts),
            "ppo_policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "clip_fraction": last_clip_fraction,
            "responses": [{"completion": r["completion"], "reward": r["reward"]} for r in rollouts],
        }
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"step={step}/{args.steps} reward={record['mean_reward']:.3f} "
            f"accuracy={record['exact_accuracy']:.1%} "
            f"policy_loss={last_policy_loss:.3f} value_loss={last_value_loss:.3f} "
            f"sampled_log_ratio={record['mean_sampled_log_ratio']:.3f}",
            flush=True
        )

    policy.save_pretrained(out)
    tokenizer.save_pretrained(out)
    torch.save(value_head.cpu().state_dict(), out / "value_head.pt")
    (out / "experiment.json").write_text(json.dumps({
        "algorithm": "educational single-terminal-reward sequence-level PPO",
        "starting_model": "SFT-merged" if args.sft_adapter else "base",
        "base_model": args.model,
        "sft_adapter": args.sft_adapter,
        "data": args.data,
        "steps": args.steps,
        "seed": args.seed,
        "temperature": args.temperature,
        "rollout_batch": args.rollout_batch,
        "ppo_epochs": args.ppo_epochs,
        "max_new_tokens": args.max_new_tokens,
        "learning_rate": args.learning_rate,
        "value_learning_rate": args.value_learning_rate,
        "clip_epsilon": args.clip_epsilon,
        "kl_coef": args.kl_coef,
        "note": "One response = one action; no token-wise PPO/GAE or trained reward model",
    }, indent=2), encoding="utf-8")
    print(f"Saved PPO adapter, critic head and metrics to {out}")


if __name__ == "__main__":
    main()
