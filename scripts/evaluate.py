import argparse,json,torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer,AutoModelForCausalLM
from src.data.toy import load
from src.evaluation.metrics import accuracy
p=argparse.ArgumentParser()
p.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--adapter",default=None)
p.add_argument("--data",default="data/generated/val.jsonl")
p.add_argument("--output",default="outputs/evaluation.json")
a=p.parse_args()
device="cuda" if torch.cuda.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained(a.model)
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype="auto").to(device)
if a.adapter: model=PeftModel.from_pretrained(model,a.adapter)
model.eval();results=[]
for row in load(a.data,("prompt","answer")):
    prompt=tok.apply_chat_template([{"role":"user","content":row["prompt"]}],
                                   tokenize=False,add_generation_prompt=True)
    inputs=tok(prompt,return_tensors="pt").to(device)
    with torch.inference_mode():
        ids=model.generate(**inputs,max_new_tokens=32,do_sample=False,
                           pad_token_id=tok.eos_token_id)
    prediction=tok.decode(ids[0,inputs.input_ids.shape[1]:],skip_special_tokens=True)
    results.append({"prompt":row["prompt"],"expected":row["answer"],
                    "predicted":prediction,"correct":accuracy(prediction,row["answer"])})
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
score=sum(x["correct"] for x in results)/len(results)
out.write_text(json.dumps({"accuracy":score,"samples":len(results),"predictions":results},indent=2))
print(f"Accuracy: {score:.2%} on {len(results)} examples; saved {out}")
