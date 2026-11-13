import argparse
from src.data.toy import generate
p=argparse.ArgumentParser()
p.add_argument("--train",type=int,default=128)
p.add_argument("--val",type=int,default=32)
p.add_argument("--out",default="data/generated")
a=p.parse_args()
print(generate(a.out,a.train,a.val))
