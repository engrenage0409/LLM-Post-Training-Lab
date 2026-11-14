import pytest
import torch

from src.ppo.objectives import ppo_losses, terminal_advantage


def test_terminal_advantage_uses_reference_and_value():
    target, advantage, sampled_ratio = terminal_advantage(
        torch.tensor([1.0]), torch.tensor([-1.0]),
        torch.tensor([-1.5]), torch.tensor([0.3]), 0.2
    )
    assert target.item() == pytest.approx(0.9)
    assert advantage.item() == pytest.approx(0.6)
    assert sampled_ratio.item() == pytest.approx(0.5)


def test_clipped_surrogate_applies_upper_bound_to_positive_advantage():
    new_logp = torch.tensor([0.5], requires_grad=True)
    total, policy, value, fraction = ppo_losses(
        new_logp, torch.tensor([0.0]), torch.tensor([1.0]),
        torch.tensor([0.0], requires_grad=True), torch.tensor([0.0]),
        clip_epsilon=0.2
    )
    assert policy.item() == pytest.approx(-1.2)
    assert value.item() == pytest.approx(0.0)
    assert fraction.item() == pytest.approx(1.0)
    total.backward()
    assert new_logp.grad.item() == pytest.approx(0.0)


def test_critic_receives_gradient():
    current_value = torch.tensor([0.0], requires_grad=True)
    total, _, _, _ = ppo_losses(
        torch.tensor([0.0], requires_grad=True), torch.tensor([0.0]),
        torch.tensor([0.0]), current_value, torch.tensor([1.0])
    )
    total.backward()
    assert current_value.grad.abs().item() > 0


def test_invalid_hyperparameters():
    with pytest.raises(ValueError):
        terminal_advantage(torch.tensor([1.0]), torch.tensor([0.0]),
                           torch.tensor([0.0]), torch.tensor([0.0]), -0.1)
    with pytest.raises(ValueError):
        ppo_losses(torch.tensor([0.0]), torch.tensor([0.0]),
                   torch.tensor([0.0]), torch.tensor([0.0]),
                   torch.tensor([0.0]), clip_epsilon=1)
