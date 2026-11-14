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

## Milestones

- [x] Organized source, SFT/DPO/GRPO training entrypoints, toy data, reward and evaluation code.
- [x] Smoke-test SFT, DPO and GRPO on target GPU (small toy dataset).
- [ ] Validate chained SFT -> DPO and SFT -> GRPO runs and pin exact dependency versions.
- [ ] Record held-out accuracy, reward/loss curves and per-sample regressions.
- [ ] Add real preference and verifiable-math datasets, careful split protocols.
- [ ] Controlled ablations: DPO beta, GRPO generations/rewards, LoRA rank.
- [ ] Implement Reward Model + PPO/RLHF, including actor/reference/critic/reward metrics.

See [docs/reading-list.md](docs/reading-list.md) for official references and papers.
