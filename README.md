# LLM Post-Training Lab

A small **SFT / DPO / GRPO learning scaffold** built with Hugging Face TRL and PEFT. Default model: `Qwen/Qwen2.5-0.5B-Instruct`, intended as a lower-memory starting point for an RTX 5060 8GB GPU.

**Status:** Code uploaded; actual model training and benchmark results **not yet completed**. Reward Model and PPO are roadmap-only. SFT, DPO and GRPO are *independent experiments*, not an automatically chained pipeline.

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
- [ ] Run and debug on target GPU; pin exact dependency versions.
- [ ] Record held-out accuracy, reward/loss curves and per-sample regressions.
- [ ] Add real preference and verifiable-math datasets, careful split protocols.
- [ ] Controlled ablations: DPO beta, GRPO generations/rewards, LoRA rank.
- [ ] Implement Reward Model + PPO/RLHF, including actor/reference/critic/reward metrics.

See [docs/reading-list.md](docs/reading-list.md) for official references and papers.
