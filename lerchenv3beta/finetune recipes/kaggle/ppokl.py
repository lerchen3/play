MAX_LENGTH = 16384
from cut_cross_entropy import linear_cross_entropy

import os
os.environ["WANDB_DISABLED"] = "true"  # disable W&B
from accelerate import Accelerator
from unsloth import unsloth_train

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from transformers import default_data_collator
import pandas as pd
import ast

# Multi-GPU + hyperparameter configuration
MODEL_PTH       = '/kaggle/input/unsloth-qwen3-8b-bnb-4bit-model/8bqwenquant'
SFT_CSV_PATH    = '/kaggle/input/wlogprobsfrfr/wlogprobs_chat_formatted_wlogprobsfr.csv'
NUM_BATCHES = 1
BATCH_NUM = 0
NUM_ROWS = len(pd.read_csv(SFT_CSV_PATH)) // NUM_BATCHES
NUM_TRAIN_EPOCHS = 3
CTX_LENGTH = 16390
GRAD_ACC = 4
DEV_BATCH_SIZE = 1
EFF_BATCH_SIZE = GRAD_ACC * DEV_BATCH_SIZE * 4 #(for 4 data-parallel GPUs :D)
MAX_STEPS = int(NUM_ROWS // EFF_BATCH_SIZE * 0.9)
EVAL_STEPS = MAX_STEPS // 10
USE_4BIT = True

# Initialize Accelerator for DDP
accelerator    = Accelerator()
device         = accelerator.device
device_index   = accelerator.process_index
device_map     = {"": device_index}

# 1. Define the trainer subclass inline
from cut_cross_entropy import linear_cross_entropy

class OfflinePPOTrainerWithKL(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # pop reference log-probs
        ref_logps = inputs.pop("pireflogprobs")
        # fast fused CE via parent (Unsloth-patched) trainer
        ce_loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        # compute per-token log-probs from logits
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        # get labels
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("No 'labels' in inputs.")
        # shift for causal LM: logits at i predict token at i+1
        shift_labels = labels[..., 1:].contiguous()
        shift_ref    = ref_logps[..., 1:].contiguous()
        valid_mask   = (shift_labels != -100).float()
        # gather model log-probs for true tokens
        cur_logps = log_probs[..., :-1, :].gather(
            -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        p_model = cur_logps.exp()
        p_ref   = shift_ref.exp()
        # ratio-based loss per token and average
        ratio   = p_model / (p_ref + 0.1)
        ratio_loss = (ratio * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        return (ratio_loss, outputs) if return_outputs else ratio_loss

# 2. Load & patch the model via Unsloth
orig_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PTH,
    max_seq_length=MAX_LENGTH,
    dtype=None,
    trust_remote_code=True,
    load_in_4bit=USE_4BIT,
    device_map=device_map,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    orig_model,
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

# 3. Load chat-formatted CSV and build dataset
df = pd.read_csv(SFT_CSV_PATH)
batch_df = df[BATCH_NUM * NUM_ROWS : (BATCH_NUM + 1) * NUM_ROWS]
batch_df.to_csv(f"sft_data_batch{BATCH_NUM}.csv", index=False)
df = pd.read_csv(f"sft_data_batch{BATCH_NUM}.csv")
def parse_list(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x
df["pireflogprobs"] = df["pireflogprobs"].apply(parse_list)
records = df.to_dict(orient="records")
for i, rec in enumerate(records):
    rec["example_idx"] = i
dataset = Dataset.from_list(records)

# 4. Tokenize & align labels and pireflogprobs
def tokenize_and_align(example):
    enc = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    assistant_start = example["assistant_start_idx"]
    assistant_end = example["assistant_end_idx"]
    labels = [
        token if (i >= assistant_start and i <= assistant_end and m == 1) else -100
        for i, (token, m) in enumerate(zip(input_ids, attention_mask))
    ]
    def pad_and_mask(seq):
        seq_padded = seq[:MAX_LENGTH] + [0.0] * max(0, MAX_LENGTH - len(seq))
        return [
            v if (i >= assistant_start and i <= assistant_end and m == 1) else 0.0
            for i, (v, m) in enumerate(zip(seq_padded, attention_mask))
        ]
    ref = pad_and_mask(example["pireflogprobs"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pireflogprobs": ref,
        "example_idx": example["example_idx"],
    }

processed_dataset = dataset.map(tokenize_and_align, remove_columns=["text"])

# 5. Set up training arguments with a 'beta' field monkey-patched
training_args = SFTConfig(
    output_dir="outputs_ppokl",
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=DEV_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=2e-5,
    remove_unused_columns=False,
    dataset_text_field="text",
    packing=False,
    max_seq_length=MAX_LENGTH,
    logging_steps=1,
    optim="adamw_8bit",
    report_to="none",
)
# add KL weight
training_args.beta = 0.1
# disable unused-parameter detection for DDP
training_args.ddp_find_unused_parameters = False

# 6. Custom collator to batch pireflogprobs (advantage is removed)
def custom_data_collator(features):
    idxs = [f["example_idx"] for f in features]
    print(f"=== training on example idxs: {idxs} ===")
    batch = default_data_collator(features)
    batch["pireflogprobs"] = torch.tensor([f["pireflogprobs"] for f in features], dtype=torch.float32)
    return batch

# 7. Initialize & run the trainer
trainer = OfflinePPOTrainerWithKL(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,
    data_collator=custom_data_collator,
)

print("Starting OfflinePPO+KL training...")
try:
    # Use unsloth_train to correctly handle gradient accumulation in DDP
    trainer_stats = unsloth_train(trainer)
    print("Training complete.")
except Exception as e:
    print(f"An error occurred during training: {e}")
    raise e

# 1) Synchronize all processes
accelerator.wait_for_everyone()

# 2) Only the main process writes out model files
if accelerator.is_main_process:
    model_to_save = accelerator.unwrap_model(model)
    model_to_save.save_pretrained_merged(
        "lora-model",
        tokenizer,
        save_method="merged_16bit",
        save_function=accelerator.save,
    )
    tokenizer.save_pretrained("lora-model")