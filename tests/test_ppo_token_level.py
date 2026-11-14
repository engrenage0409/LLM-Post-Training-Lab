import pytest
import torch

from src.ppo.token_level import gae_terminal, shaped_rewards, token_ppo_losses


def test_gae_one_terminal_reward_exact_when_gamma_lambda_one():
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.2, 0.3, 0.4])
    advantages, targets = gae_terminal(rewards, values, gamma=1.0, lam=1.0)
    assert torch.allclose(targets, torch.ones(3))
    assert torch.allclose(advantages, torch.tensor([0.8, 0.7, 0.6]))


def test_gae_lambda_zero_is_one_step_td():
    rewards = torch.tensor([0.0, 1.0])
    values = torch.tensor([0.2, 0.3])
    adv, targets = gae_terminal(rewards, values, gamma=1.0, lam=0.0)
    assert torch.allclose(adv, torch.tensor([0.1, 0.7]))
    assert torch.allclose(targets, torch.tensor([0.3, 1.0]))


def test_kl_reward_shaping_adds_terminal_reward_only_at_last_action():
    old = torch.tensor([-1.0, -0.3])
    ref = torch.tensor([-1.5, -0.5])
    shaped, log_ratio = shaped_rewards(old, ref, terminal_reward=1.0, kl_coef=0.1)
    assert torch.allclose(log_ratio, torch.tensor([0.5, 0.2]))
    assert torch.allclose(shaped, torch.tensor([-0.05, 0.98]))


def test_token_clip_policy_and_critic_gradients():
    new = torch.tensor([0.0, 0.5], requires_grad=True)
    values = torch.tensor([0.0, 0.0], requires_grad=True)
    loss, ploss, vloss, fraction = token_ppo_losses(
        new, torch.tensor([0.0, 0.0]), torch.tensor([-1.0, 1.0]),
        values, torch.tensor([1.0, 1.0]), clip_epsilon=0.2,
    )
    assert ploss.item() == pytest.approx(-0.1)
    assert fraction.item() == pytest.approx(0.5)
    loss.backward()
    assert new.grad is not None
    assert values.grad.abs().sum().item() > 0


def test_invalid_gae():
    with pytest.raises(ValueError):
        gae_terminal(torch.empty(0), torch.empty(0))
    with pytest.raises(ValueError):
        gae_terminal(torch.ones(2), torch.ones(3))
