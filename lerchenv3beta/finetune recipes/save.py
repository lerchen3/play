model_pth = "model_training_outputs/checkpoint-342"
output_pth = "32b-limo-sft"
NUM_ROWS = 1000
BATCH_NUM = 4
GRAD_ACC = 4
DEV_BATCH_SIZE = 1
EFF_BATCH_SIZE = GRAD_ACC * DEV_BATCH_SIZE
MAX_STEPS = NUM_ROWS // EFF_BATCH_SIZE * 0.9 #for train test split i guess.
SAVE_STEPS = MAX_STEPS // 5
use_4bit = True
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer  
from transformers import TrainingArguments, DataCollatorForSeq2Seq  
from transformers import EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from datasets import load_dataset

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_pth,
    max_seq_length=16390,
    dtype=None,
    trust_remote_code=True,
    load_in_4bit=use_4bit,
)

model.save_pretrained_merged(output_pth, tokenizer, save_method="merged_16bit")