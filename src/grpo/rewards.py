from src.evaluation.metrics import accuracy,parse_integer
def accuracy_reward(completions,answer,**kwargs):
    return [accuracy(c,a) for c,a in zip(completions,answer)]
def format_reward(completions,**kwargs):
    return [0.1 if parse_integer(c) is not None else 0.0 for c in completions]
