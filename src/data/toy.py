import json,random
from pathlib import Path

def generate(out="data/generated",train=128,val=32,seed=42):
    rng=random.Random(seed)
    pairs=[(a,b) for a in range(100) for b in range(100)]
    rng.shuffle(pairs)
    if train<=0 or val<=0 or train+val>len(pairs): raise ValueError("Invalid split sizes")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,selected in (("train",pairs[:train]),("val",pairs[train:train+val])):
        with (out/f"{name}.jsonl").open("w",encoding="utf-8") as f:
            for a,b in selected:
                y=str(a+b)
                item={"prompt":f"What is {a} + {b}? Answer with only an integer.",
                      "completion":y,"chosen":y,"rejected":str(a+b+1),"answer":y}
                f.write(json.dumps(item)+"\n")
    return out

def load(path,fields):
    with open(path,encoding="utf-8") as f:
        rows=[json.loads(line) for line in f if line.strip()]
    if not rows: raise ValueError("Empty JSONL")
    if any(any(k not in row for k in fields) for row in rows):
        raise ValueError("Missing required fields")
    return rows
