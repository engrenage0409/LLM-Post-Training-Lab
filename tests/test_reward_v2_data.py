from collections import Counter

from scripts.prepare_reward_v2 import make_mixed_pairs


def test_mixed_negative_pairs_are_balanced_and_repeatable():
    rows = [
        {"prompt": f"What is {i} + {i+1}?", "answer": str(2*i+1)}
        for i in range(10)
    ]
    first = make_mixed_pairs(rows, seed=2026)
    assert first == make_mixed_pairs(rows, seed=2026)
    assert len(first) == 50
    types = Counter(r["negative_type"] for r in first)
    assert types == {
        "plus_one": 10, "minus_one": 10,
        "plus_ten": 10, "minus_ten": 10,
        "random_signed": 10,
    }
    assert all(r["chosen"] != r["rejected"] for r in first)
    for row in first:
        difference = int(row["rejected"]) - int(row["chosen"])
        if row["negative_type"] == "plus_one":
            assert difference == 1
        if row["negative_type"] == "minus_one":
            assert difference == -1
        if row["negative_type"] == "plus_ten":
            assert difference == 10
        if row["negative_type"] == "minus_ten":
            assert difference == -10
    random_offsets = [
        int(r["rejected"]) - int(r["chosen"])
        for r in first if r["negative_type"] == "random_signed"
    ]
    assert sum(x > 0 for x in random_offsets) == 5
    assert sum(x < 0 for x in random_offsets) == 5
