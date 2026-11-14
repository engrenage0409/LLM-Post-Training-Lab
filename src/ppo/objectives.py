"""Small, inspectable sequence-level PPO objective for a single terminal reward.

A sampled complete response is one action. This is intentionally simpler
than token-level PPO with per-token values and generalized advantage estimation.
"""
import torch
import torch.nn.functional as F


def ppo_losses(new_logp, old_logp, advantage, new_value, value_target, clip_epsilon=0.2, value_coef=0.5):
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in (0, 1)")
    # old_logp/advantage/target are rollout constants; never differentiate
    # through the reference or reward.
    ratio = torch.exp((new_logp - old_logp.detach()).clamp(-20, 20))
    unclipped = ratio * advantage.detach()
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage.detach()
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = F.mse_loss(new_value.float(), value_target.detach().float())
    loss = policy_loss + value_coef * value_loss
    clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
    return loss, policy_loss, value_loss, clip_fraction


def terminal_advantage(reward, old_logp, ref_logp, old_value, kl_coef):
    if kl_coef < 0:
        raise ValueError("kl_coef must be non-negative")
    # This per-sample log-ratio can be negative; it is a Monte Carlo
    # estimator of sequence-level KL, not an always-positive exact KL.
    sampled_log_ratio = old_logp.detach() - ref_logp.detach()
    target = reward.detach() - kl_coef * sampled_log_ratio
    advantage = target - old_value.detach()
    return target, advantage, sampled_log_ratio
