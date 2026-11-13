from src.data.toy import generate,load
def test_split(tmp_path):
    out=generate(tmp_path,20,10)
    tr=load(out/"train.jsonl",("prompt","completion"))
    val=load(out/"val.jsonl",("prompt","answer"))
    assert len(tr)==20 and len(val)==10
    assert set(x["prompt"] for x in tr).isdisjoint(x["prompt"] for x in val)
