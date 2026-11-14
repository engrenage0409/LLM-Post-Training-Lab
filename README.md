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

## Phase 7: Train and evaluate a Reward Model

This stage is **reward modeling**, not PPO yet. Train a separate Qwen2.5-0.5B scalar sequence-classification head with LoRA (including the `score` head in `modules_to_save`). The training data contains *chosen vs. rejected* complete user/assistant conversations; the objective is to assign the chosen conversation a higher scalar score. We deliberately **do not load the SFT adapter** into this separate classifier: the existing SFT LoRA targets a causal-LM task, and reward modeling uses its own sequence-classification architecture. A future PPO actor can still start from the SFT adapter.

```powershell
git pull origin main
python -m pytest -q
python -m scripts.train_reward_model --steps 1 --output outputs/reward_model_smoke
python -m scripts.train_reward_model --steps 50 --output outputs/reward_model
python -m scripts.evaluate_reward_model --adapter outputs/reward_model --output outputs/reward_model_eval.json
```

The test reports **preference accuracy** (fraction of validation pairs with `score(chosen) > score(rejected)`) and mean **reward margin** (`score(chosen) - score(rejected)`). JSON includes per-pair scores. `outputs/reward_model` saves a PEFT adapter *and* a learned `score` head, not a standalone full model; keep the base checkpoint available. Run the one-step smoke test first on an 8GB card. This uses **128 synthetic arithmetic preference pairs** for training and **32** for validation; it is a learning/demo dataset, NOT evidence of human-aligned preferences. It only distinguishes correct sums from a deliberately off-by-one wrong answer. No preference model should be claimed to generalize beyond this task.

The current prototype loads one 0.5B sequence-classification model; it does not allocate policy, critic, and reward models concurrently. PPO/RLHF will be a subsequent separate implementation, after local RewardTrainer compatibility and GPU memory are verified. Library version pinning remains recommended.

## Reward Model diagnostics: baseline, overfitting, shortcut patterns

The trained reward model scored 62.50% (20/32) on original validation pairs with mean margin +0.0159 after 50 steps. These values are **not enough** to establish reliable learned correctness.

Run the following diagnostics with exactly the same trained adapter; no retraining required:

```powershell
git pull origin main
python -m pytest -q
python -m scripts.diagnose_reward_model --adapter outputs/reward_model --output outputs/reward_diagnostics_trained.json
python -m scripts.diagnose_reward_model --untrained --seed 42 --output outputs/reward_diagnostics_untrained.json
```

Each command compares original off-by-one preferences with negative answers formed using -1, +10, a reproducible random nonzero offset, and a fixed incorrect answer. It separately reports *training* (128) and *validation* (32) preference accuracy, mean chosen-minus-rejected margin, and per-example outputs.

`--untrained` loads an independently randomly initialized scalar score head, **not an exact reconstruction of the pre-training initialization of the trained reward model**. Its random performance is a rough sanity-check baseline, not a paired controlled estimate. Compare original versus alternate negatives to detect possible shortcut reliance, but small samples and simple arithmetic prevent definitive conclusions. In particular, the always-zero negative is an easy distributional artifact and should not be used as the final benchmark.

## Phase 7b: Reward Model V2 — balanced mixed negatives

The original Reward Model V1 (50 steps, 128 train pairs) scored 62.5% on 32 held-out +1 pairs, but only 37.5% on -1 and 0% against a fixed wrong answer of zero. This suggests shortcut risk (not proof of a specific internal heuristic). V2 retains the original data and checkpoint and creates **a new preference file** with five negative candidates per question: correct+1, correct-1, correct+10, correct-10, and a deterministic balanced positive/negative random offset.

```powershell
git pull origin main
python -m pytest -q
python -m scripts.prepare_reward_v2
```

With the existing 128/32 question split, this produces **640 training preference pairs and 160 validation pairs**, saved in `data/reward_v2/train.jsonl` and `val.jsonl`. The five pairs for a question stay in the same split. Original `data/generated/*.jsonl` files remain untouched.

Train a fresh reward adapter, separately from V1:

```powershell
python -m scripts.train_reward_model --train data/reward_v2/train.jsonl --val data/reward_v2/val.jsonl --steps 160 --output outputs/reward_model_v2
```

Evaluate the mixed-negative validation preferences:

```powershell
python -m scripts.evaluate_reward_model --adapter outputs/reward_model_v2 --data data/reward_v2/val.jsonl --output outputs/reward_model_v2_eval.json
```

Compare **V1 vs V2 on exactly the original 32 questions**, using the same diagnostic modes and no re-training. Running full diagnostics on the V2 expanded file would repeat each prompt five times:

```powershell
python -m scripts.diagnose_reward_model --adapter outputs/reward_model_v2 --train data/generated/train.jsonl --val data/generated/val.jsonl --output outputs/reward_diagnostics_v2.json
```

Check whether performance on `minus_one`, `random_error` and `fixed_zero` actually improves, and whether the +1 performance is retained. Because the diagnostic modes have now been inspected multiple times, they are exploratory; a brand new, sealed final benchmark is still required. V2 still uses synthetic arithmetic offsets—not human preferences or sampled policy failures—and may learn other shortcuts. PPO should not treat this RM as validated without further tests.

## Phase 8: Low-memory educational PPO (single terminal reward)

This is a **working-code prototype awaiting a local GPU smoke test**. It deliberately implements **sequence-level PPO**, where one sampled full completion counts as **one action**, not a production-style token-level RLHF PPO with per-token rewards, GAE, or an independently pretrained value function.

- **Policy / actor:** merge SFT LoRA into Qwen2.5-0.5B, then train a fresh LoRA.
- **Reference:** same SFT-merged backbone with PPO LoRA temporarily disabled (`policy.disable_adapter()`); no second 0.5B reference model on GPU.
- **Critic / value:** new small linear value head trained alongside policy (saved separately).
- **Reward:** exact integer correctness (1.0) and valid integer format (0.1) using Python, **not** the unreliable V1/V2 learned Reward Model.
- **Return:** sampled terminal reward minus `kl_coef * (old_sequence_logp - reference_sequence_logp)`. That sampled log-ratio is a noisy KL estimator and may be negative.
- **Update:** sequence probability ratio `exp(new_logp - old_logp)` with PPO clipping and critic MSE.
- **Data / GPU:** one base model, one prompt at a time, small LoRA and head. Default rollout batch=2, two PPO epochs, max completion 12 tokens.

Run first with only one step and inspect logs / memory:

```powershell
git pull origin main
python -m pytest -q
python -m scripts.train_ppo --steps 1 --rollout-batch 2 --ppo-epochs 2 --output outputs/ppo_smoke
```

If that succeeds, train a new adapter, **not overwriting SFT**:

```powershell
python -m scripts.train_ppo --steps 40 --rollout-batch 4 --ppo-epochs 2 --output outputs/ppo_40
```

Outputs: `adapter_model.safetensors`, `value_head.pt`, `train_metrics.jsonl`, `experiment.json`. The critic head is for continued PPO training and **not** needed for ordinary text generation.

PPO adapter must be evaluated with its original SFT start (as with chained DPO/GRPO):

```powershell
python -m scripts.evaluate --sft-adapter outputs/sft --adapter outputs/ppo_40 --data outputs/benchmark/test.jsonl --output outputs/ppo_40_eval.json
```

Compare against the pre-PPO SFT policy on the same test questions. Because `outputs/benchmark/test.jsonl` has already been used in extensive tuning, these results are **exploratory**, not untouched final generalization. Inspect `mean_reward`, `exact_accuracy`, `ppo_policy_loss`, `value_loss`, `mean_sampled_log_ratio` (not a guaranteed positive KL), and samples in the log. Training is stochastic; 40 steps or one seed do not prove PPO improvement. This code does not load a learned reward model or implement GAE; those are natural follow-up extensions.

## Phase 8b: Token-level PPO + GAE (experimental)

The original `scripts/train_ppo.py` is preserved as the **sequence-level baseline** (which scored 75.33% on the repeatedly used 600-question development benchmark after 40 steps). The new `scripts/train_ppo_token.py` implements a more faithful autoregressive token-action formulation:

1. Actor samples a response *from its raw policy* (temperature 1, top-k disabled, top-p 1).
2. For every output token `t`, evaluate `log pi_old(y_t | prompt, y_<t)` and `V(prompt, y_<t)`; **Critic sees only the prefix before the action**, not the sampled token itself.
3. Disable PPO LoRA to compute frozen SFT-reference token log-probabilities; no duplicate base model is held in GPU memory.
4. Apply token-level sampled log-ratio reward `-kl_coef * (log pi_old - log pi_ref)`, and add exact correctness + format reward **only to the final generated token**.
5. Compute backward GAE with terminal value zero, `delta_t = r_t + gamma V_(t+1) - V_t` and `A_t = delta_t + gamma*lambda*A_(t+1)`. Returns = detached rollout values + GAE.
6. Normalize advantage across *all* sampled tokens per rollout batch; re-evaluate new policy log-probabilities and prefix values for PPO clipped actor and critic MSE losses over repeated epochs.
7. Save new policy LoRA and separate critic head. This is educational single-GPU PPO, **not a full production RLHF pipeline**; the reward remains deterministic correctness, not the learned RM.

Run tests and a memory-safe 1-step smoke test **before** the full training:

```powershell
git pull origin main
python -m pytest -q
python -m scripts.train_ppo_token --steps 1 --rollout-batch 2 --ppo-epochs 2 --output outputs/ppo_token_smoke
```

If it succeeds, train an independent 40-step run:

```powershell
python -m scripts.train_ppo_token --steps 40 --rollout-batch 2 --ppo-epochs 2 --output outputs/ppo_token_40
```

Evaluate its *actor adapter*, always supplying the original SFT start:

```powershell
python -m scripts.evaluate --sft-adapter outputs/sft --adapter outputs/ppo_token_40 --data outputs/benchmark/test.jsonl --output outputs/ppo_token_40_eval.json
```

Inspect `outputs/ppo_token_40/train_metrics.jsonl` for token count, reward, rollout accuracy, actor/critic losses, sampled log ratio and clipping. The score is **not** a perfectly controlled comparison against sequence-level PPO (token-level PPO has different reward credit assignment and rollout budget). Benchmark already used during tuning => developmental, not an untouched final test.

**Implementation caveats:** all tokens in an unpadded, single-response rollout are actions; generated EOS is included as an action if emitted, truncated completions are treated as terminal; GAE assumes terminal bootstrap zero. Critic is a linear head on a shared actor backbone, so its gradient can also update the actor's trainable LoRA via shared features. No adaptive KL, clipped value loss, partial-rollout bootstrap, optimizer scheduling, or distributed inference. Use with tiny experiments and verify training logs locally before reporting numerical results.

## Milestones

- [x] Organized source, SFT/DPO/GRPO training entrypoints, toy data, reward and evaluation code.
- [x] Smoke-test SFT, DPO and GRPO on target GPU (small toy dataset).
- [ ] Validate chained SFT -> DPO and SFT -> GRPO runs and pin exact dependency versions.
- [ ] Record held-out accuracy, reward/loss curves and per-sample regressions.
- [ ] Add real preference and verifiable-math datasets, careful split protocols.
- [ ] Controlled ablations: DPO beta, GRPO generations/rewards, LoRA rank.
- [x] Implement LoRA Reward Model training and pairwise evaluation scripts (local execution pending).
- [x] Implement low-memory sequence-level PPO prototype (local execution pending).
- [x] Implement experimental token-level PPO + GAE scripts and unit tests (local GPU smoke test pending).
- [ ] Validate token-level PPO training and evaluate against an independent held-out dataset and learned reward model.

See [docs/reading-list.md](docs/reading-list.md) for official references and papers.
