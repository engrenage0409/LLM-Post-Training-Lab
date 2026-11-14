# LLM Post-Training Lab — Project Summary / 项目总结

> Status: core educational implementation and final single-seed arithmetic comparison completed on an RTX 5060 8GB. This is an **algorithm-learning and reproducibility project**, not a production RLHF platform.

## 1. 项目概述

围绕 `Qwen/Qwen2.5-0.5B-Instruct`，独立搭建轻量化 LLM Post-Training 实验框架，覆盖 **SFT、DPO、GRPO、Reward Modeling、Sequence-level PPO、Token-level PPO + GAE**，并完成多种训练策略消融、Reward Model 泛化诊断及独立算术测试。

技术栈：**Python、PyTorch、Transformers、TRL、PEFT/LoRA、Datasets**；Windows + RTX 5060 8GB。

### 训练流程

```text
Qwen2.5-0.5B-Instruct
    ├── SFT (LoRA)
    │     ├── SFT → DPO (preference optimization; beta ablation)
    │     ├── SFT → GRPO (verifiable rewards; reward/group-size ablation)
    │     ├── SFT → Sequence-level PPO
    │     └── SFT → Token-level PPO + GAE
    └── Reward Model V1/V2
          └── Pairwise preference scoring; diagnostics (not used as PPO reward)
```

## 2. 个人实现的关键技术

- **SFT 与参数高效微调**：基于 LoRA 实现监督微调与独立训练 / 级联微调；正确恢复 SFT merged backbone + stage-two adapter 进行评测，避免 Adapter 与 Base Model 起点不一致。
- **DPO**：通过 chosen/rejected 偏好对直接优化生成策略，实现 `beta` 参数消融（0.05 / 0.10 / 0.20）。
- **GRPO**：基于数学正确性和答案格式构造可验证奖励，验证奖励组合与 Group Size（2 / 4 / 8）对结果的影响。
- **Reward Modeling**：实现 Qwen scalar score head + LoRA 的偏好排序训练（pairwise ranking），对比 V1 单一 `correct+1` 负样本与 V2 多类型正负偏移样本，诊断偏好捷径。
- **PPO / Critic / GAE**：实现序列级 PPO 及 Token-level PPO；包含 Actor、SFT Reference、Critic Value Head、KL penalty、clipped surrogate objective、value regression、GAE；使用 LoRA adapter switching 避免额外加载一份完整 Reference，适配单张 8GB GPU。
- **可复现实验**：固定随机种子与训练预算，开发 Benchmark 与新的独立算术测试集，排除训练/验证及原开发测试题目的交换律等价形式；保存逐题预测、训练指标与可自动生成的报告。

## 3. 最终实验结果

**数据集**：600 道独立于训练、验证和原开发 Benchmark 的整数加法题，尚不代表通用数学推理能力；**单随机种子**，PPO 两版使用匹配的训练步数、Rollout Batch、PPO Epochs 与最大生成 Token 数，但优化目标仍存在结构差异。

| Model | Correct / 600 | Accuracy | vs SFT |
|---|---:|---:|---:|
| SFT | 423 / 600 | 70.50% | — |
| Sequence-level PPO | 443 / 600 | 73.83% | +3.33 pp |
| Token-level PPO + GAE | 444 / 600 | 74.00% | +3.50 pp |

PPO V1 vs V2 **逐题配对**：441 道都对，154 道都错，只有 V1 对 2 道，只有 V2 对 3 道。两版仅差 **1 题（0.17 pp）**；**没有证据可据此主张 V2 显著优于 V1**。

训练日志中的模型间诊断指标：

| Metric | Sequence-level | Token-level |
|---|---:|---:|
| mean_reward | 0.8488 | 0.7950 |
| critic_target_mae | 0.4957 | 0.4113 |
| raw_advantage_mean | 0.0118 | 0.0219 |
| raw_advantage_std | 0.2296 | 0.3257 |
| clip_fraction | 2.50% | 1.61% |
| mean_sampled_log_ratio | +0.0486 | -0.0213 |
| value_loss | 0.3350 | 0.2575 |

**Interpretation notes:** sequence-level PPO and token-level PPO have different action granularity, reward/return targets and Critic state definitions; their losses and average rewards cannot be interpreted as strictly comparable estimators of the same target. Sampled log ratio is not exact KL.

### 以前的开发测试结果（只供探索，不与最终测试混用）

- PPO Sequence-level: 75.33% (452/600)
- PPO Token-level: 73.50% (441/600)
- GRPO Group Size 4: 74.00% (444/600)
- Reward Model V1: 62.50% preference accuracy (32 off-by-one validation pairs)
- Reward Model V2: 54.37% preference accuracy (160 mixed-negative validation pairs)

开发集曾被多次用于选择训练方案，因此这些结果**不属于全新的最终测试集**。RM V1/V2 属于人工合成偏好排序，**尚未训练出可靠的通用奖励模型**，且 PPO 实验使用规则奖励而不是这两版 RM。

## 4. 简历描述

### 中文：适合 LLM / Agent / 多模态算法岗位（紧凑版）

**LLM 后训练与强化学习实验框架（个人项目）｜PyTorch · TRL · PEFT · LoRA · PPO · GRPO**

- **端到端后训练实现：** 基于 Qwen2.5-0.5B-Instruct 独立实现 SFT、DPO、GRPO、Reward Modeling 和 PPO 训练链路；支持 SFT→偏好优化级联训练、LoRA 参数管理及一致化模型评测，适配单卡 8GB 显存。
- **强化学习优化与消融：** 实现 Sequence-level / Token-level PPO、Actor-Critic、GAE、KL 惩罚与 Clipped Objective，开展 DPO β、GRPO 奖励组合和 Group Size 消融，并分析奖励建模中的负样本偏差。
- **可复现实验与量化评估：** 建立独立算术 Benchmark 和配对逐题评测；在新生成且与先前训练、验证和开发集去重的 600 题上，Token-level PPO 达到 **74.00%**，较 SFT **70.50% 提升 3.50 pp**；两版 PPO 表现接近，避免过度解释单次实验差异。

### English resume version

**LLM Post-Training & Reinforcement Learning Lab (Independent Project)** | PyTorch, TRL, PEFT, LoRA, DPO, GRPO, PPO

- Built a lightweight post-training pipeline for Qwen2.5-0.5B-Instruct spanning SFT, DPO, GRPO, pairwise reward modeling, and PPO, with LoRA-based stage chaining and memory-aware single-GPU execution on an 8GB GPU.
- Implemented sequence- and token-level PPO with a learned value head, GAE, KL regularization and clipped policy objectives; conducted controlled exploratory ablations on DPO beta, GRPO reward design and group size, and synthetic reward-model bias.
- Designed a disjoint 600-question arithmetic evaluation protocol; token-level PPO reached **74.00% accuracy vs. 70.50% for SFT (+3.50 percentage points)** in a single-seed comparison, with no meaningful accuracy separation from sequence-level PPO.

### 一页简历更短的两条版本

- 基于 Qwen2.5-0.5B 独立实现 SFT、DPO、GRPO、Reward Modeling 及 Token-level PPO + GAE 后训练流程，并开展偏好数据诊断、DPO/GRPO 消融与单卡 8GB LoRA 优化。
- 设计去重独立算术评测，600 题上 PPO **74.00%**，较 SFT **70.50%** 提升 **3.50 pp**；实现完整训练日志、配对比较及自动实验报告。

## 5. 面试常见问题（回答要点）

**Q1. DPO、Reward Model、GRPO、PPO 的区别？**

DPO 直接用 chosen/rejected 的对数概率比优化 Policy，不显式训练 RM。Reward Model 学习为 prompt+answer 输出标量偏好分。GRPO 用组内相对奖励估计 Advantage，通常无独立 Critic。PPO 通常用 Critic/GAE 估计 Advantage，并通过新旧策略概率比的裁剪目标优化 Policy。GRPO 与 PPO 都可使用规则奖励或训练好的 RM。

**Q2. 为什么 PPO 需要 Reference？**

防止后训练 Policy 为追逐奖励而大幅偏离 SFT。此实现通过在相同的 SFT-merged backbone 上禁用 PPO LoRA 得到 Reference 的 log-probability，无需额外常驻一份模型。采样的 log-ratio 是有噪声的 KL 估计，不是每次都非负。

**Q3. Critic 和 Reward Model 有什么不同？**

Reward Model 评价一个候选回答；Critic 估计给定状态从当前策略继续执行的期望回报，服务于 Advantage 和 Value Loss。V2 的 Critic 使用**生成每个 Token 之前的 prefix hidden state**；V1 的简化 Critic 使用完成回答后的 hidden state。

**Q4. GAE 怎么计算？**

逐时间步计算 TD 残差 `delta_t = r_t + gamma V(s_{t+1}) - V(s_t)`，从终点倒推 `A_t = delta_t + gamma lambda A_{t+1}`。此项目正确性奖励出现在最终 Token，KL shaping 按生成 Token 分配；假定 Episode 完结，终止状态值为零。

**Q5. PPO V2 为什么没有明显强于 V1？**

单次 600 题最终测试仅差 1 题；两个 PPO 的优化目标、奖励分配和状态定义存在差异，且任务较简单、样本/随机种子有限。不能将差异全部归因于 GAE，也不能宣称 V2 显著更优。

**Q6. Reward Model 出现了什么数据问题？**

V1 的 rejected 仅为正确答案 +1，容易形成数据捷径；诊断发现 V1 在 `correct-1`、固定错误零等设置上泛化不足。V2 混合负样本改善部分偏差，但 160 对验证样本仅得到 54.37% 排序准确率。因此 PPO 最终采用确定性的数学正确性规则奖励，未将不可靠 RM 用作 PPO 奖励源。

**Q7. 实验如何控制泄漏与可复现性？**

固定 seed、初始 SFT、训练/采样预算；最终 600 题排除训练/验证集和旧开发 Benchmark 题目，包括 `a+b` 与 `b+a` 的重复。最终结果只用于结项，不继续调参。局限是只有一次 seed 和狭窄合成任务。

## 6. 诚实的项目边界 / Limitations

- 当前数学任务是小规模、合成的整数加法任务；不能用 74% 推断通用推理、真实偏好对齐或通用 RLHF 收益。
- Token-level PPO 是面向学习的轻量实现：未做完整工业级 RLHF 的自适应 KL、分布式扩展或高可信真实奖励建模。
- 只有一次随机种子；最终测试上两版 PPO 差距仅 1 题，不足以构成算法优劣结论。
- 如果未来要进一步升级，可开展多种子重复、真实偏好/可验证数据训练、奖励模型质量验证和更广泛任务评测；这些不属于本次结项范围。

## 7. 复现入口

```powershell
python -m pytest -q
python -m scripts.ppo_final_comparison --steps 40 --rollout-batch 2 --ppo-epochs 2 --samples 600 --output-dir outputs/ppo_final
python -m scripts.ppo_final_report --directory outputs/ppo_final
```

注意：已有 `outputs/ppo_final/sealed_test.jsonl` 的运行环境不要重跑同一路径；当前最终测试结果应冻结为结项基准。具体实验原始指标以该次运行生成的 `FINAL_REPORT.md` 为准。

