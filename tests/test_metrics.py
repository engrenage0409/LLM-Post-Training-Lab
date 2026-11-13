from src.evaluation.metrics import parse_integer,accuracy
from src.grpo.rewards import accuracy_reward,format_reward
def test_metrics():
    assert parse_integer("42")==42
    assert parse_integer("Answer: 42") is None
    assert parse_integer(r"\\boxed{42}")==42
    assert accuracy("42","42")==1.
    assert accuracy("41","42")==0.
def test_rewards():
    assert accuracy_reward(["3","4"],["3","3"])==[1.,0.]
    assert format_reward(["3","hello"])==[0.1,0.]
