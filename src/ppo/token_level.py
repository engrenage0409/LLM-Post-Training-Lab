"""Token-level PPO helpers with reward shaping and GAE.

T generated tokens = T actions; each value is computed at the prefix
*before* its associated token is sampled.
"""
import torch
from src.ppo.objectives import ppo_losses


def token_logprobs_and_values(policy, value_head, full_ids, prompt_length):
    if full_ids.ndim != 2 or full_ids.shape[0] != 1:
        raise ValueError("Expected a single unpadded sequence [1, length]")
    if not (1 <= prompt_length < full_ids.shape[1]):
        raise ValueError("Need at least one completion token")
    out = policy(input_ids=full_ids, output_hidden_states=True, use_cache=False)
    # Token i is predicted from hidden/logit at position i-1.
    logits = out.logits[:, prompt_length - 1:-1, :].float()
    targets = full_ids[:, prompt_length:]
    logps = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
    prefix_hidden = out.hidden_states[-1][:, prompt_length - 1:-1, :]
    values = value_head(prefix_hidden.float()).squeeze(-1)[0]
    assert logps.shape == values.shape
    return logps, values


def gae_terminal(token_rewards, old_values, gamma=1.0, lam=0.95):
    """GAE for a finite response, with V(s_terminal)=0 and no padding.

    Reward r_t corresponds to the token action at t. old_values[t]
    estimates V(prefix before token t); last transition terminates.
    """
    if token_rewards.ndim != 1 or old_values.shape != token_rewards.shape:
        raise ValueError("Expected matching 1D rewards and values")
    if not token_rewards.numel():
        raise ValueError("Expected at least one token")
    if not (0 <= gamma <= 1 and 0 <= lam <= 1):
        raise ValueError("gamma and lam must be in [0,1]")
    advantage = torch.empty_like(token_rewards)
    last_gae = torch.zeros_like(token_rewards[0])
    for t in reversed(range(len(token_rewards))):
        next_value = old_values[t + 1] if t + 1 < len(token_rewards) else torch.zeros_like(old_values[t])
        delta = token_rewards[t] + gamma * next_value - old_values[t]
        last_gae = delta + gamma * lam * last_gae
        advantage[t] = last_gae
    return advantage, advantage + old_values


def shaped_rewards(old_logps, ref_logps, terminal_reward, kl_coef):
    if old_logps.ndim != 1 or ref_logps.shape != old_logps.shape:
        raise ValueError("Expected matching 1D log probability vectors")
    if kl_coef < 0:
        raise ValueError("kl_coef must be nonnegative")
    if old_logps.numel() == 0:
        raise ValueError("Need at least one token")
    log_ratio = old_logps.detach() - ref_logps.detach()
    rewards = -kl_coef * log_ratio
    rewards = rewards.clone()
    rewards[-1] += float(terminal_reward)
    return rewards, log_ratio


def token_ppo_losses(new_logps, old_logps, advantages, new_values, targets, clip_epsilon=0.2, value_coef=0.5):
    return ppo_losses(new_logps, old_logps, advantages, new_values, targets, clip_epsilon, value_coef)
