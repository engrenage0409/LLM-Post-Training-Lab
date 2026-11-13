import re
def parse_integer(text):
    s=str(text).strip()
    if re.fullmatch(r"[+-]?\d+",s): return int(s)
    matches=re.findall(r"\\boxed\{([+-]?\d+)\}",s)
    return int(matches[-1]) if matches else None
def accuracy(prediction,reference):
    p=parse_integer(prediction);r=parse_integer(reference)
    return float(p is not None and r is not None and p==r)
