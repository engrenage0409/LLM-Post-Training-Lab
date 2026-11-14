import pytest

from src.grpo.rewards import reward_functions_for_mode


def test_reward_mode_accuracy_only():
    def accuracy(*args, **kwargs):
        return [1.0]

    def fmt(*args, **kwargs):
        return [0.1]

    assert reward_functions_for_mode("accuracy", accuracy, fmt) == [accuracy]
    assert reward_functions_for_mode("accuracy_format", accuracy, fmt) == [accuracy, fmt]


def test_reward_mode_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported reward mode"):
        reward_functions_for_mode("invalid", lambda: None, lambda: None)
