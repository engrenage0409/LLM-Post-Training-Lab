import pytest

from src.reward_model.data import preference_conversation, pairwise_statistics


def test_preference_pair_includes_context():
    item = preference_conversation({
        "prompt": "What is 12 + 43?",
        "chosen": "55", "rejected": "56",
    })
    assert item["chosen"][0] == item["rejected"][0]
    assert item["chosen"][-1]["content"] == "55"
    assert item["rejected"][-1]["content"] == "56"


def test_preference_pair_rejects_ties():
    with pytest.raises(ValueError, match="must differ"):
        preference_conversation({"prompt": "Q", "chosen": "1", "rejected": "1"})


def test_pairwise_accuracy_and_margin():
    result = pairwise_statistics([(1.2, 0.2), (0.0, 0.4), (0.2, 0.2)])
    assert result["pairs"] == 3
    assert result["preference_accuracy"] == pytest.approx(1 / 3)
    assert result["mean_margin"] == pytest.approx(0.2)
    assert result["ties"] == 1
