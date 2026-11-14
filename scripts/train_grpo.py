import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from src.data.toy import load
from src.model_loading import load_sft_merged_model
from src.grpo.batch_config import validate_group_batch
from src.grpo.rewards import accuracy_reward, format_reward, reward_functions_for_mode

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--data", default="data/generated/train.jsonl")
p.add_argument("--steps", type=int, default=1)
p.add_argument("--num-generations", type=int, choices=(2, 4, 8), default=2,
               help="Completions per prompt in GRPO group-size ablation")
p.add_argument("--gradient-accumulation-steps", type=int, default=4,
               help="Keep at 4 for all G=2,4,8 experiments to hold effective batch at 8")
p.add_argument("--sft-adapter", default=None, help="Continue from an SFT LoRA adapter")
p.add_argument("--reward-mode", choices=("accuracy", "accuracy_format"), default="accuracy_format",
               help="GRPO reward ablation: correctness only or correctness plus formatting")
p.add_argument("--output", default="outputs/grpo")
p.add_argument("--debug-samples", type=int, default=12,
               help="Maximum number of generated completions to print and save (0 disables logging)")
a = p.parse_args()
effective_batch = validate_group_batch(
    a.num_generations, per_device_batch_size=2,
    gradient_accumulation_steps=a.gradient_accumulation_steps,
)
print(f"GRPO group size={a.num_generations}, effective batch={effective_batch}, "
      f"unique prompts per step={effective_batch // a.num_generations}", flush=True)

# Conversational prompts let the Instruct tokenizer apply its chat template.
ds = Dataset.from_list([
    {"prompt": [{"role": "user", "content": x["prompt"]}], "answer": x["answer"]}
    for x in load(a.data, ("prompt", "answer"))
])

output_dir = Path(a.output)
output_dir.mkdir(parents=True, exist_ok=True)
debug_path = output_dir / "debug_samples.jsonl"
debug_count = 0
reward_calls = 0
if a.debug_samples > 0:
    debug_path.write_text("", encoding="utf-8")


def completion_text(completion):
    if isinstance(completion, list):
        return str(completion[-1].get("content", "")) if completion else ""
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def conversational_format_reward(completions, **kwargs):
    return format_reward([completion_text(c) for c in completions], **kwargs)


def logged_accuracy_reward(completions, answer, **kwargs):
    """Score normally, and log a bounded number of raw training generations."""
    global debug_count, reward_calls
    texts = [completion_text(c) for c in completions]
    scores = accuracy_reward(texts, answer, **kwargs)
    if debug_count < a.debug_samples:
        format_scores = format_reward(texts, **kwargs)
        prompts = kwargs.get("prompts", [])
        records = []
        for i, (completion, expected, score, fmt_score) in enumerate(
            zip(texts, answer, scores, format_scores)
        ):
            if debug_count >= a.debug_samples:
                break
            record = {
                "reward_call": reward_calls,
                "prompt": str(prompts[i]) if i < len(prompts) else None,
                "expected": str(expected),
                "completion": str(completion),
                "accuracy_reward": float(score),
                "format_reward": float(fmt_score),
            }
            records.append(record)
            debug_count += 1
            print("[GRPO DEBUG] " + json.dumps(record, ensure_ascii=False), flush=True)
        if records:
            with debug_path.open("a", encoding="utf-8") as fp:
                for record in records:
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    reward_calls += 1
    return scores


cfg = GRPOConfig(
    output_dir=a.output, max_steps=a.steps,
    per_device_train_batch_size=2, gradient_accumulation_steps=a.gradient_accumulation_steps,
    num_generations=a.num_generations, max_completion_length=64,
    learning_rate=5e-6, use_vllm=False, logging_steps=1,
    save_strategy="no", report_to="none", seed=42,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
)
lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
)
starting_model = load_sft_merged_model(a.model, a.sft_adapter) if a.sft_adapter else a.model
trainer = GRPOTrainer(
    model=starting_model, args=cfg, train_dataset=ds,
    reward_funcs=reward_functions_for_mode(a.reward_mode, logged_accuracy_reward, conversational_format_reward),
    peft_config=lora,
)
trainer.train()
trainer.save_model(a.output)
