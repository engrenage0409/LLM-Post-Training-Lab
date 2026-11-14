from src.evaluation.metrics import accuracy,parse_integer
def accuracy_reward(completions,answer,**kwargs):
    return [accuracy(c,a) for c,a in zip(completions,answer)]
def format_reward(completions,**kwargs):
    return [0.1 if parse_integer(c) is not None else 0.0 for c in completions]


def reward_functions_for_mode(mode, accuracy_fn, format_fn):
    """Select only the reward functions used by a GRPO ablation."""
    if mode == "accuracy":
        return [accuracy_fn]
    if mode == "accuracy_format":
        return [accuracy_fn, format_fn]
    raise ValueError(f"Unsupported reward mode: {mode}")
