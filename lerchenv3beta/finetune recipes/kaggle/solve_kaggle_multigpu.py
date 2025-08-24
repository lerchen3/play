import os
import pandas as pd
from accelerate import Accelerator
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from unsloth.chat_templates import train_on_responses_only

# Configuration
MODEL_PTH = "/kaggle/input/unsloth-qwen3-8b-bnb-4bit-model/8bqwenquant"
SFT_CSV_PATH = "/kaggle/input/dumbassssss/wlogprobs_chat_formatted.csv"
NUM_BATCHES = 1
NUM_ROWS = len(pd.read_csv(SFT_CSV_PATH)) // NUM_BATCHES
NUM_TRAIN_EPOCHS = 3
CTX_LENGTH = 16390
BATCH_NUM = 0
GRAD_ACC = 4
DEV_BATCH_SIZE = 1
EFF_BATCH_SIZE = GRAD_ACC * DEV_BATCH_SIZE * 4 #(for 4 data-parallel GPUs :D)
MAX_STEPS = int(NUM_ROWS // EFF_BATCH_SIZE * 0.9)
EVAL_STEPS = MAX_STEPS // 10
USE_4BIT = False

accelerator = Accelerator()
# Determine the local GPU id for this process
device = accelerator.device
device_index = accelerator.process_index
device_map = {"": device_index}
# Load the model and tokenizer across GPUs
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PTH,
    max_seq_length=CTX_LENGTH,
    dtype=None,
    trust_remote_code=True,
    load_in_4bit=USE_4BIT,
    device_map=device_map,
)
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
df = pd.read_csv(SFT_CSV_PATH)
batch_df = df[BATCH_NUM * NUM_ROWS : (BATCH_NUM + 1) * NUM_ROWS]
batch_df.to_csv(f"sft_data_batch{BATCH_NUM}.csv", index=False)
ds = load_dataset('csv', data_files={"train": f"sft_data_batch{BATCH_NUM}.csv"})
split = ds['train'].train_test_split(test_size=0.1, seed=2009)
train_ds, eval_ds = split['train'], split['test']

# Apply LoRA to the model
peft_model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=2009,
    use_rslora=False,
    loftq_config=None,
)

# Initialize the trainer
trainer = SFTTrainer(
    model=peft_model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    dataset_text_field="text",
    max_seq_length=CTX_LENGTH,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=DEV_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="no",
        warmup_ratio=0.05,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=2e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.005,
        lr_scheduler_type="linear",
        seed=2009,
        output_dir="model_training_outputs",
        ddp_find_unused_parameters=False, # DON'T REMOVE THIS IT'S IMPORTANT
        report_to="none",
    ),
)
# Only train on assistant responses
trainer = train_on_responses_only(
    # THIS WORKS
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_end|>\n<|im_start|>assistant\n",
)

from unsloth import unsloth_train
# unsloth_train fixes gradient_accumulation_steps
# trainer_stats = trainer.train()
trainer_stats = unsloth_train(trainer)

# 1) make sure all processes have reached this point
accelerator.wait_for_everyone()

# 2) only the main process actually writes files
if accelerator.is_main_process:
    model_to_save = accelerator.unwrap_model(peft_model)
    model_to_save.save_pretrained_merged(
        "lora-model",
        tokenizer,
        save_method="merged_16bit",
        save_function=accelerator.save,
    )
    tokenizer.save_pretrained("lora-model")
