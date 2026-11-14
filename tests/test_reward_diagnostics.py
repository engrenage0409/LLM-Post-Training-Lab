import pytest

from scripts.diagnose_reward_model import make_negative_rows


@pytest.fixture
def data():
    return [
        {"prompt": "What is 12 + 43?", "answer": "55", "chosen": "55", "rejected": "56"},
        {"prompt": "What is 1 + 2?", "answer": "3", "chosen": "3", "rejected": "4"},
    ]


@pytest.mark.parametrize("mode", ["original", "minus_one", "plus_ten", "random_error", "fixed_zero"])
def test_negative_modes_are_deterministic_and_wrong(data, mode):
    first = make_negative_rows(data, mode)
    assert first == make_negative_rows(data, mode)
    assert all(x["chosen"] != x["rejected"] for x in first)
    assert all(x["chosen"] == original["answer"] for x, original in zip(first, data))


def test_negative_modes_expected_offsets(data):
    assert make_negative_rows(data, "minus_one")[0]["rejected"] == "54"
    assert make_negative_rows(data, "plus_ten")[0]["rejected"] == "65"


def test_negative_modes_reject_unknown(data):
    with pytest.raises(ValueError, match="Unknown negative mode"):
        make_negative_rows(data, "unknown")
