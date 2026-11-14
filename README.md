# LLM Post-Training Lab

A small **SFT / DPO / GRPO learning scaffold** built with Hugging Face TRL and PEFT. Default model: `Qwen/Qwen2.5-0.5B-Instruct`, intended as a lower-memory starting point for an RTX 5060 8GB GPU.

**Status:** SFT, DPO and GRPO smoke tests have run on the target GPU. The current 32-question toy evaluation is not a general reasoning benchmark. Reward Model and PPO are roadmap-only. Independent experiments remain available; optional SFT -> DPO and SFT -> GRPO chaining is now supported (not yet validated end-to-end).

## Layout

```text
src/data/            deterministic arithmetic-data generator
src/evaluation/      strict exact-answer metric
src/grpo/            verifiable math and formatting rewards
src/sft/ src/dpo/    reserved for later algorithm internals
src/reward_model/    future: scalar reward modeling
src/ppo/             future: PPO / RLHF
scripts/             dataset, training and evaluation entrypoints
tests/               lightweight unit tests
docs/                papers and training documentation
```

## Setup on Windows

Install CUDA-compatible PyTorch for your GPU (see [PyTorch selector](https://pytorch.org/get-started/locally/)), then from the repository root:

```powershell
git clone https://github.com/engrenage0409/LLM-Post-Training-Lab.git
cd LLM-Post-Training-Lab
python -m pip install -r requirements.txt
python -m pytest -q
```

Trainer APIs can change: the requirement ranges are not a reproducibility lockfile. Capture `pip freeze` and library versions before reporting results.

## Generate sample data

```powershell
python -m scripts.prepare_data
```

Creates disjoint `data/generated/train.jsonl` and `val.jsonl`. Data has the following fields: `prompt, completion, chosen, rejected, answer` for three tasks. The toy questions involve integer addition with synthetic off-by-one rejected answers. **Smoke testing only; NOT a real mathematical reasoning benchmark.**

## Training commands

```powershell
python -m scripts.train_sft --steps 20
python -m scripts.train_dpo --steps 20
python -m scripts.train_grpo --steps 1
```

Scripts train LoRA adapters from the same base model *separately* and save to `outputs/sft`, `outputs/dpo`, `outputs/grpo`. Try GRPO with 1 step first. **8 GB feasibility is not yet verified**, particularly for GRPO generation/rollouts. Model downloads require internet access.

## Optional chained post-training (SFT -> DPO / GRPO)

First train SFT (or reuse an existing `outputs/sft` adapter). Then run:

```powershell
python -m scripts.train_dpo --sft-adapter outputs/sft --steps 20 --output outputs/sft_dpo
python -m scripts.train_grpo --sft-adapter outputs/sft --steps 1 --debug-samples 12 --output outputs/sft_grpo
```

The stage-one SFT LoRA is **merged into the base model in memory**, and a **new LoRA** is trained for stage two. SFT weights are not overwritten. For DPO, the reference policy is the SFT-merged model with the new LoRA disabled. For GRPO, generation and policy optimization start from the SFT-merged model.

**Important:** the stage-two adapter alone is **not** a complete model. To reproduce chained predictions, supply the original SFT adapter **and** the stage-two adapter:

```powershell
python -m scripts.evaluate --sft-adapter outputs/sft --adapter outputs/sft_dpo --output outputs/sft_dpo_eval.json
python -m scripts.evaluate --sft-adapter outputs/sft --adapter outputs/sft_grpo --output outputs/sft_grpo_eval.json
```

Do not evaluate `outputs/sft_dpo` or `outputs/sft_grpo` without `--sft-adapter`: that would attach stage-two weights to the wrong starting model. The merged-model step may increase RAM/VRAM demand. This chained path requires an actual local test before reporting results.

## Compare on identical validation set

```powershell
python -m scripts.evaluate --output outputs/base_eval.json
python -m scripts.evaluate --adapter outputs/sft --output outputs/sft_eval.json
python -m scripts.evaluate --adapter outputs/dpo --output outputs/dpo_eval.json
python -m scripts.evaluate --adapter outputs/grpo --output outputs/grpo_eval.json
```

Score is strict integer exact-match, accepting standalone integers or `\\boxed{n}` (not verbose unboxed answers). It is deliberately narrow and cannot establish general reasoning ability.

## Six-model benchmark (held-out arithmetic)

Create a deterministic **600-question** test file, excluding original training and validation questions *and their commuted equivalents* (`a+b` vs `b+a`). Questions are sampled across no-carry, ones-carry, and sums >= 100 categories.

```powershell
python -m pytest -q
python -m scripts.benchmark --samples 600 --prepare-only
```

Then evaluate all six saved checkpoints sequentially in separate processes to release GPU memory:

```powershell
python -m scripts.benchmark --samples 600
```

For a short smoke run, use `--samples 30 --output-dir outputs/benchmark_smoke`; the full 600-question run requires **3,600 generations** and may take significantly longer. Ensure your adapters exist at `outputs/sft`, `outputs/dpo`, `outputs/grpo_chat_100`, `outputs/sft_dpo`, and `outputs/sft_grpo_100`, or adjust the `MODELS` mapping in `scripts/benchmark.py`.

Results are written to `outputs/benchmark/test.jsonl`, `manifest.json`, six per-model JSON files, and `summary.json` with accuracy per difficulty and individual improved/regressed examples relative to SFT. Each invocation recreates its test set deterministically with seed 2026. Do **not** use this test set repeatedly to tune hyperparameters and then claim unbiased final performance. Synthetic arithmetic is not a general reasoning benchmark.

## GRPO reward ablation: accuracy vs. accuracy + format

The existing GRPO setup uses two reward functions: exact integer correctness (0 or 1) and valid answer format (0 or 0.1). The new `--reward-mode` flag lets us isolate the effect of formatting rewards while keeping training configuration unchanged. The default, `accuracy_format`, preserves earlier behavior.

Train **two independent GRPO runs from the same SFT adapter** with separate output directories:

```powershell
python -m scripts.train_grpo --sft-adapter outputs/sft --reward-mode accuracy --steps 100 --debug-samples 0 --output outputs/ablation_grpo_accuracy
python -m scripts.train_grpo --sft-adapter outputs/sft --reward-mode accuracy_format --steps 100 --debug-samples 0 --output outputs/ablation_grpo_accuracy_format
```

Evaluate on the **same already-generated, untouched test file**:

```powershell
python -m scripts.evaluate --sft-adapter outputs/sft --adapter outputs/ablation_grpo_accuracy --data outputs/benchmark/test.jsonl --output outputs/ablation_grpo_accuracy_eval.json
python -m scripts.evaluate --sft-adapter outputs/sft --adapter outputs/ablation_grpo_accuracy_format --data outputs/benchmark/test.jsonl --output outputs/ablation_grpo_accuracy_format_eval.json
```

For a fair comparison, use the same data, seed, base/SFT checkpoint, step budget and evaluation protocol. This is an exploratory analysis on the already-inspected 600-question set, **not a new unbiased held-out test**. GRPO samples stochastically, so one seed alone cannot establish robust superiority. Each stage-two adapter must be evaluated together with the SFT adapter.

## GRPO group-size ablation (G=2, 4, 8)

`--num-generations` sets how many completions are sampled for each prompt. For a **single GPU**, the default is now `per_device_train_batch_size=2` and `gradient_accumulation_steps=4`: an **effective batch of 8 generated completions per optimizer step**, divisible by every tested group size. This changes the earlier default accumulation of 2; explicitly record the configuration when comparing with older checkpoints.

First run a short smoke test for the largest group, especially on 8 GB VRAM:

```powershell
python -m scripts.train_grpo --sft-adapter outputs/sft --num-generations 8 --steps 1 --debug-samples 8 --output outputs/grpo_g8_smoke
```

Then train the three independent ablation runs (same SFT starting point, seed=42, reward mode, update count, and effective completion batch size):

```powershell
python -m scripts.train_grpo --sft-adapter outputs/sft --num-generations 2 --steps 100 --debug-samples 0 --output outputs/grpo_g2
python -m scripts.train_grpo --sft-adapter outputs/sft --num-generations 4 --steps 100 --debug-samples 0 --output outputs/grpo_g4
python -m scripts.train_grpo --sft-adapter outputs/sft --num-generations 8 --steps 100 --debug-samples 0 --output outputs/grpo_g8
```

Evaluate each using the same previously generated arithmetic test set (exploratory; it has already been inspected):

```powershell
python -m scripts.evaluate --sft-adapter outputs/sft --adapter outputs/grpo_g2 --data outputs/benchmark/test.jsonl --output outputs/grpo_g2_eval.json
python -m scripts.evaluate --sft-adapter outputs/sft --adapter outputs/grpo_g4 --data outputs/benchmark/test.jsonl --output outputs/grpo_g4_eval.json
python -m scripts.evaluate --sft-adapter outputs/sft --adapter outputs/grpo_g8 --data outputs/benchmark/test.jsonl --output outputs/grpo_g8_eval.json
```

The effective batch is 8 **completions**, so G=2 uses 4 distinct prompts per optimizer step, G=4 uses 2, and G=8 uses 1. Thus the number of distinct questions seen differs even with the same number of optimizer updates. Compare `reward_std`, `frac_reward_zero_std`, throughput, VRAM, and accuracy; **do not infer a pure group-size causal effect without also controlling the number of unique prompts or doing additional matched-budget experiments**. The 8 GB feasibility of G=8 is not yet verified. Trainer APIs and group batching may differ by TRL version. Use `--gradient-accumulation-steps` to adjust the effective batch deliberately.

## DPO beta ablation (0.05, 0.10, 0.20)

`--beta` adjusts the preference log-ratio scale in the DPO loss (default `0.1` to preserve prior runs). Compare these three **independent** runs from the same `outputs/sft` starting model with identical training examples, update count, LoRA, and seed:

```powershell
python -m scripts.train_dpo --sft-adapter outputs/sft --beta 0.05 --steps 20 --output outputs/dpo_beta005
python -m scripts.train_dpo --sft-adapter outputs/sft --beta 0.10 --steps 20 --output outputs/dpo_beta010
python -m scripts.train_dpo --sft-adapter outputs/sft --beta 0.20 --steps 20 --output outputs/dpo_beta020
```

Score all three on the **same existing exploratory benchmark**:

```powershell
python -m scripts.evaluate --sft-adapter outputs/sft --adapter outputs/dpo_beta005 --data outputs/benchmark/test.jsonl --output outputs/dpo_beta005_eval.json
python -m scripts.evaluate --sft-adapter outputs/sft --adapter outputs/dpo_beta010 --data outputs/benchmark/test.jsonl --output outputs/dpo_beta010_eval.json
python -m scripts.evaluate --sft-adapter outputs/sft --adapter outputs/dpo_beta020 --data outputs/benchmark/test.jsonl --output outputs/dpo_beta020_eval.json
```

Inspect `loss`, `rewards/accuracies`, `rewards/margins`, and `logps/rejected` / `logps/chosen` where emitted by TRL, plus exact-match accuracy and per-example changes. DPO beta is not simply a learning rate; it scales the policy-versus-reference preference log-ratio. This toy dataset uses synthetic off-by-one rejections and is **not** representative of realistic human preference data. The already-inspected 600-question benchmark is for exploratory comparison, not an untouched final test. Validate conclusions with multiple seeds and a fresh sealed set.

## Milestones

- [x] Organized source, SFT/DPO/GRPO training entrypoints, toy data, reward and evaluation code.
- [x] Smoke-test SFT, DPO and GRPO on target GPU (small toy dataset).
- [ ] Validate chained SFT -> DPO and SFT -> GRPO runs and pin exact dependency versions.
- [ ] Record held-out accuracy, reward/loss curves and per-sample regressions.
- [ ] Add real preference and verifiable-math datasets, careful split protocols.
- [ ] Controlled ablations: DPO beta, GRPO generations/rewards, LoRA rank.
- [ ] Implement Reward Model + PPO/RLHF, including actor/reference/critic/reward metrics.

See [docs/reading-list.md](docs/reading-list.md) for official references and papers.
