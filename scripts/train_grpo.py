import argparse,torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig,GRPOTrainer
from src.data.toy import load
from src.grpo.rewards import accuracy_reward,format_reward
p=argparse.ArgumentParser()
p.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--data",default="data/generated/train.jsonl")
p.add_argument("--steps",type=int,default=1)
p.add_argument("--output",default="outputs/grpo")
a=p.parse_args()
ds=Dataset.from_list([{k:x[k] for k in ("prompt","answer")} for x in load(a.data,("prompt","answer"))])
cfg=GRPOConfig(output_dir=a.output,max_steps=a.steps,per_device_train_batch_size=2,
 gradient_accumulation_steps=2,num_generations=2,max_completion_length=32,
 learning_rate=5e-6,use_vllm=False,logging_steps=1,save_strategy="no",report_to="none",seed=42,
 bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
 fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported())
lora=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM")
trainer=GRPOTrainer(model=a.model,args=cfg,train_dataset=ds,
 reward_funcs=[accuracy_reward,format_reward],peft_config=lora)
trainer.train();trainer.save_model(a.output)
