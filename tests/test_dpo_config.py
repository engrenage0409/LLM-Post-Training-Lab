import pytest
from argparse import ArgumentTypeError

from src.dpo.config import positive_beta


@pytest.mark.parametrize("value, expected", [
    ("0.05", 0.05), ("0.1", 0.1), ("0.20", 0.2), (0.1, 0.1),
])
def test_valid_dpo_beta(value, expected):
    assert positive_beta(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "abc"])
def test_invalid_dpo_beta(value):
    with pytest.raises(ArgumentTypeError, match="DPO beta|Invalid DPO beta"):
        positive_beta(value)
