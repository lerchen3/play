from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer  
from transformers import TrainingArguments, DataCollatorForSeq2Seq  
from sklearn.model_selection import train_test_split
INPUT_CSV = "LIMO_SFT_BASE_QWEN (1).csv"
NUM_ROWS = 817
BATCH_NUM = 0
GRAD_ACC = 4
DEV_BATCH_SIZE = 1
EFF_BATCH_SIZE = GRAD_ACC * DEV_BATCH_SIZE
MAX_STEPS = NUM_ROWS // EFF_BATCH_SIZE * 0.9 #for train test split i guess.
SAVE_STEPS = MAX_STEPS // 10

from transformers import EarlyStoppingCallback
import torch
import os
import re
from typing import List, Literal, Optional
from datasets import load_dataset, Dataset
import pandas as pd
import wandb

wandb.login(key="f9cca73107f008f4f576c7fc2362bbc2fed22b2c")

max_seq_length = 16390
dtype = None
load_in_4bit = True

# 1. Load model and tokenizer (using unsloth FastLanguageModel)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-32B-Instruct-bnb-4bit",  # Choose ANY! e.g. mistralai/Mistral-7B-Instruct-v0.2
    max_seq_length=max_seq_length,
    trust_remote_code=True,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Explicitly set pad token and ensure it's properly registered
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. Prepare training dataset (from results.csv) and save as csv_dataset_dpo.csv

raw_dataset = load_dataset("csv", data_files=INPUT_CSV)

# Convert to pandas DataFrame for splitting
df = raw_dataset['train'].to_pandas()
train_df, test_df = train_test_split(df, test_size=0.1, random_state=2009)

# Convert back to Dataset objects
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Create a DatasetDict
dataset = {
    'train': train_dataset,
    'test': test_dataset
}

# 3. Adapt model with PEFT (LoRA modifications)
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=2009,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=16390,  # as required
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,  # Keep false for better quality
    args=TrainingArguments(
        per_device_train_batch_size=DEV_BATCH_SIZE,
        per_device_eval_batch_size=DEV_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        eval_strategy="steps",         # Evaluate at the end of each epoch
        save_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_steps=SAVE_STEPS,
        warmup_ratio=0.05,
        num_train_epochs=6,            # Reduced epochs for faster iteration
        learning_rate=3e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.15,
        lr_scheduler_type="linear",
        seed=2009,
        output_dir="model_training_outputs",
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    ),
)

from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    # THIS WORKS
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

trainer_stats = trainer.train()

model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
