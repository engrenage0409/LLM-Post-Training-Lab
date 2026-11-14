import json

import pytest

from scripts.ppo_final_comparison import make_sealed_rows, compare_predictions
from scripts.benchmark import pair_from_prompt


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def test_final_set_disjoint_with_train_validation_and_old_benchmark(tmp_path):
    train, val, old = [tmp_path / x for x in ("train.jsonl", "val.jsonl", "old.jsonl")]
    write_jsonl(train, [{"prompt": "What is 1 + 2? Answer with only an integer."}])
    write_jsonl(val, [{"prompt": "What is 3 + 4? Answer with only an integer."}])
    write_jsonl(old, [{"prompt": "What is 5 + 6? Answer with only an integer."}])
    first = make_sealed_rows(train, val, old, 30, seed=77)
    assert first == make_sealed_rows(train, val, old, 30, seed=77)
    keys = [pair_from_prompt(x["prompt"]) for x in first]
    assert len(keys) == len(set(keys))
    assert not set(keys) & {(1, 2), (3, 4), (5, 6)}


def test_sealed_set_excludes_commuted_duplicate(tmp_path):
    train, val, old = [tmp_path / x for x in ("train.jsonl", "val.jsonl", "old.jsonl")]
    write_jsonl(train, [{"prompt": "What is 2 + 1? Answer with only an integer."}])
    write_jsonl(val, [{"prompt": "What is 4 + 3? Answer with only an integer."}])
    write_jsonl(old, [{"prompt": "What is 6 + 5? Answer with only an integer."}])
    rows = make_sealed_rows(train, val, old, count=100, seed=42)
    assert not {(1, 2), (3, 4), (5, 6)} & {
        pair_from_prompt(x["prompt"]) for x in rows
    }


def test_compare_predictions_paired(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"predictions": [
        {"prompt": "p1", "correct": True},
        {"prompt": "p2", "correct": False},
        {"prompt": "p3", "correct": True},
        {"prompt": "p4", "correct": False},
    ]}), encoding="utf-8")
    b.write_text(json.dumps({"predictions": [
        {"prompt": "p1", "correct": True},
        {"prompt": "p2", "correct": True},
        {"prompt": "p3", "correct": False},
        {"prompt": "p4", "correct": False},
    ]}), encoding="utf-8")
    assert compare_predictions(a, b) == {
        "both_correct": 1, "both_wrong": 1, "a_only": 1, "b_only": 1,
    }


def test_invalid_size(tmp_path):
    data = tmp_path / "d.jsonl"
    write_jsonl(data, [{"prompt": "What is 1 + 1? Answer with only an integer."}])
    with pytest.raises(ValueError):
        make_sealed_rows(data, data, data, count=0)
