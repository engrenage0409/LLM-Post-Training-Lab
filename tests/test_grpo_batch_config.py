import pytest
from src.grpo.batch_config import validate_group_batch


@pytest.mark.parametrize("group", [2, 4, 8])
def test_supported_groups_share_effective_batch(group):
    assert validate_group_batch(group, 2, 4) == 8


def test_reject_incompatible_batch():
    with pytest.raises(ValueError, match="divisible"):
        validate_group_batch(8, 2, 2)


def test_reject_unknown_group_size():
    with pytest.raises(ValueError, match="2, 4, 8"):
        validate_group_batch(3, 2, 4)
