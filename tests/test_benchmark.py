import json

import pytest

from scripts.benchmark import make_test_rows, pair_from_prompt, summarize
from src.data.toy import generate, load


def test_benchmark_no_leakage_and_reproducible(tmp_path):
    data_dir = generate(tmp_path / "data", train=128, val=32, seed=42)
    train = data_dir / "train.jsonl"
    val = data_dir / "val.jsonl"
    first = make_test_rows(train, val, count=600, seed=2026)
    second = make_test_rows(train, val, count=600, seed=2026)
    assert first == second
    forbidden = {
        pair_from_prompt(x["prompt"])
        for file in (train, val)
        for x in load(file, ("prompt",))
    }
    pairs = [pair_from_prompt(x["prompt"]) for x in first]
    assert len(set(pairs)) == 600
    assert not forbidden.intersection(pairs)
    assert set(x["difficulty"] for x in first) == {
        "no_carry", "ones_carry", "sum_ge_100"
    }


def test_benchmark_rejects_bad_sample_count(tmp_path):
    data_dir = generate(tmp_path / "data", train=10, val=10)
    with pytest.raises(ValueError, match="positive"):
        make_test_rows(data_dir / "train.jsonl", data_dir / "val.jsonl", 0)


def test_summary_improved_and_regressed(tmp_path):
    rows = [
        {"prompt": "Q1", "difficulty": "no_carry"},
        {"prompt": "Q2", "difficulty": "ones_carry"},
    ]
    paths = {}
    for name, correct in (("sft", [0, 1]), ("dpo", [1, 0])):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({
            "accuracy": 0.5,
            "predictions": [
                {"prompt": row["prompt"], "expected": "2",
                 "predicted": "2" if flag else "3", "correct": float(flag)}
                for row, flag in zip(rows, correct)
            ],
        }), encoding="utf-8")
        paths[name] = path
    result = summarize(paths, rows)
    assert result["comparisons_vs_sft"]["dpo"]["improved_count"] == 1
    assert result["comparisons_vs_sft"]["dpo"]["regressed_count"] == 1
