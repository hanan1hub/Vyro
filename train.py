#!/usr/bin/env python3
"""
train.py — QLoRA fine-tune Qwen2.5-0.5B-Instruct for Pocket-Agent tool calling.

Requirements: pip install -r requirements.txt
Input:  data/train.jsonl
Output: adapter/  (LoRA adapter)
        merged/   (merged FP16 model — for GGUF conversion)

Runtime: ~25-35 min on Colab T4 (16 GB VRAM)
Run:    python train.py
"""

import os, json, torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── Config ────────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH    = "data/train.jsonl"
ADAPTER_DIR  = "adapter"
MERGED_DIR   = "merged"
MAX_SEQ_LEN  = 512
EPOCHS       = 3
BATCH_SIZE   = 4
GRAD_ACCUM   = 4
LR           = 2e-4
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

os.makedirs(ADAPTER_DIR, exist_ok=True)
os.makedirs(MERGED_DIR,  exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────
print("📂  Loading training data …")
records = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
print(f"    {len(records)} examples loaded")

# ── Tokenizer ─────────────────────────────────────────────────────────
print("🔤  Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Format examples using chat template ──────────────────────────────
def format_example(example):
    """Apply the model's chat template to messages."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

print("💬  Applying chat template …")
dataset = Dataset.from_list(records)
dataset = dataset.map(format_example, remove_columns=["messages"])
print(f"    Sample:\n{dataset[0]['text'][:300]}\n")

# ── 4-bit quantization config ─────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── Load base model ───────────────────────────────────────────────────
print(f"🤗  Loading base model: {BASE_MODEL} …")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

# ── LoRA config ───────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# ── SFT Config (modern replacement for TrainingArguments in TRL) ──────
sft_config = SFTConfig(
    output_dir=ADAPTER_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    optim="paged_adamw_8bit",
    report_to="none",
    dataloader_num_workers=0,
    group_by_length=True,
    max_grad_norm=0.3,
    weight_decay=0.001,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
)

# ── Trainer ───────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    peft_config=lora_config,
)

print("🚀  Training started …")
trainer.train()
print("✅  Training complete")

# ── Save LoRA adapter ─────────────────────────────────────────────────
print(f"💾  Saving LoRA adapter → {ADAPTER_DIR}")
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

# ── Merge & save full model (needed for GGUF conversion) ─────────────
print("🔀  Merging LoRA into base weights …")
from peft import PeftModel

# Reload base in fp16 for merging (no quantization)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
merged_model = PeftModel.from_pretrained(base, ADAPTER_DIR)
merged_model = merged_model.merge_and_unload()

print(f"💾 s Saving merged model → {MERGED_DIR}")
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print("✅  Merged model saved — ready for quantize.py")
