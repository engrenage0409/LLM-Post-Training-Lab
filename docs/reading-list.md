# Papers and official references
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer): prompt-completion training and label masking.
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer): preferred/rejected pairs, reference policy and beta.
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer): grouped rollouts and reward functions.
- [TRL RewardTrainer](https://huggingface.co/docs/trl/reward_trainer): planned scalar reward model.
- [DPO paper](https://arxiv.org/abs/2305.18290).
- [DeepSeekMath / GRPO paper](https://arxiv.org/abs/2402.03300).
- [PPO paper](https://arxiv.org/abs/1707.06347).
- [InstructGPT / RLHF](https://arxiv.org/abs/2203.02155).

## Suggested progression
1. Understand next-token SFT and completion masking.
2. Compare chosen vs rejected sequence log-probabilities for DPO.
3. Analyze verifiable rewards and within-group normalization for GRPO.
4. Evaluate all variants on the SAME held-out examples; report cost and runtime.
5. Implement reward modeling and PPO explicitly as a future phase.
