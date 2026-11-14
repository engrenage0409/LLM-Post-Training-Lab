"""Prepare paired conversations for outcome reward modeling.

The *same* prompt is included in the winning and losing conversations,
so the classifier must score a candidate answer in its question context.
"""


def preference_conversation(row):
    required = ("prompt", "chosen", "rejected")
    if any(key not in row for key in required):
        raise ValueError(f"Preference row missing fields: {required}")
    if str(row["chosen"]) == str(row["rejected"]):
        raise ValueError("Chosen and rejected answers must differ")
    user = {"role": "user", "content": str(row["prompt"])}
    return {
        "chosen": [user, {"role": "assistant", "content": str(row["chosen"])}],
        "rejected": [user, {"role": "assistant", "content": str(row["rejected"])}],
    }


def pairwise_statistics(scores):
    """Compute margin and preference accuracy from (chosen, rejected) scores."""
    if not scores:
        raise ValueError("Need at least one scored preference pair")
    margins = [float(chosen) - float(rejected) for chosen, rejected in scores]
    return {
        "pairs": len(margins),
        "preference_accuracy": sum(m > 0 for m in margins) / len(margins),
        "mean_margin": sum(margins) / len(margins),
        "ties": sum(m == 0 for m in margins),
    }
