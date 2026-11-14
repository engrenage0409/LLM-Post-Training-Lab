import argparse,torch
from datasets import Dataset
from peft import LoraConfig
from trl import DPOConfig,DPOTrainer
from src.data.toy import load
from src.model_loading import load_sft_merged_model
from src.dpo.config import positive_beta
p=argparse.ArgumentParser()
p.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--data",default="data/generated/train.jsonl")
p.add_argument("--steps",type=int,default=20)
p.add_argument("--beta",type=positive_beta,default=0.1,help="DPO preference scaling (default: 0.1)")
p.add_argument("--sft-adapter",default=None,help="Continue from an SFT LoRA adapter")
p.add_argument("--output",default="outputs/dpo")
a=p.parse_args()
ds=Dataset.from_list([{k:x[k] for k in ("prompt","chosen","rejected")} for x in load(a.data,("prompt","chosen","rejected"))])
cfg=DPOConfig(output_dir=a.output,max_steps=a.steps,per_device_train_batch_size=1,
 gradient_accumulation_steps=4,learning_rate=5e-5,beta=a.beta,max_length=256,
 logging_steps=1,save_strategy="no",report_to="none",seed=42,
 bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
 fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported())
lora=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM")
starting_model = load_sft_merged_model(a.model, a.sft_adapter) if a.sft_adapter else a.model
trainer=DPOTrainer(model=starting_model,args=cfg,train_dataset=ds,peft_config=lora)
trainer.train();trainer.save_model(a.output)
