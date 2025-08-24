model_pth = "/kaggle/input/qwen2.5/transformers/7b-instruct/1"
SFT_CSV_PATH = "/kaggle/input/limolimo/LIMO_SFT_BASE.csv"
NUM_ROWS = 817
BATCH_NUM = 0
GRAD_ACC = 4
DEV_BATCH_SIZE = 1
EFF_BATCH_SIZE = GRAD_ACC * DEV_BATCH_SIZE
MAX_STEPS = NUM_ROWS // EFF_BATCH_SIZE * 0.9 #for train test split i guess.
SAVE_STEPS = MAX_STEPS // 5
use_4bit = False

import subprocess
import sys
import glob
import importlib
import os
import shutil
import pandas as pd

def install_dependencies():
    """
    Installs all downloaded wheel files in the 'kevin-ppp' directory, excluding those containing 'unsloth'.
    """
    folder_path = "/kaggle/input/kevin-ppp"  # Path to the directory containing the wheels
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist. Please ensure the packages are downloaded there.")
        sys.exit(1)

    # Find all wheel files in the specified directory
    wheel_files = glob.glob(os.path.join(folder_path, "*.whl"))

    if not wheel_files:
        print(f"No suitable wheel files found in '{folder_path}'. Please ensure the packages are downloaded correctly.")
        sys.exit(1)

    print(f"Installing packages from '{folder_path}' (excluding 'unsloth')...\n")
    for wheel in wheel_files:
        print(f" - {os.path.basename(wheel)}")

    # Construct the pip install command
    cmd = [
        sys.executable, '-m', 'pip', 'install',
        '--no-index',          # Prevent pip from accessing the internet
        '--find-links', folder_path,  # Look for packages in the specified directory
        '--upgrade'
    ] + wheel_files

    try:
        subprocess.run(cmd, check=True)
        print("\nAll packages have been installed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during installation: {e}")
        sys.exit(1)

def test_imports():
    """
    Tests the installation by importing each package and printing its version.
    """
    test_packages = ['torch', 'torchvision', 'torchaudio', 'xformers', 'transformers']

    print("Starting import tests...\n")

    for package in test_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown version')
            print(f"✅ Successfully imported '{package}' (version: {version})")
        except ImportError as e:
            print(f"❌ Failed to import '{package}': {e}")
        except Exception as e:
            print(f"⚠️ An unexpected error occurred while importing '{package}': {e}")

def setup_environment():
    """Handles the initial setup and dependency checks"""
    install_dependencies()
    test_imports()

setup_environment()

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

tokenizer.pad_token = tokenizer.eos_token

# ---------------------------
# New dataset processing code
# ---------------------------
from datasets import load_dataset
ORIGINAL = pd.read_csv(SFT_CSV_PATH)
ROWS = ORIGINAL[BATCH_NUM * NUM_ROWS: (BATCH_NUM + 1) * NUM_ROWS]
ROWS.to_csv(f"anchnot_sft_data_batch{BATCH_NUM}.csv", index=False)
processed_dataset = load_dataset('csv', data_files=f"anchnot_sft_data_batch{BATCH_NUM}.csv")
split_dataset = processed_dataset['train'].train_test_split(test_size=0.1, seed=2009)
dataset = {"train": split_dataset["train"], "test": split_dataset["test"]}

# ---------------------------
# End of new dataset processing
# ---------------------------

# Initialize the PEFT model with LoRA configuration, as before.
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,  # unsloth doesn't support dropout
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=2009,
    use_rslora=False,
    loftq_config=None,
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
        num_train_epochs=2,            # Reduced epochs for faster iteration
        learning_rate=2e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.005,
        lr_scheduler_type="linear",
        seed=2009,
        output_dir="model_training_outputs",
        report_to="none",
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

if os.path.exists("model_training_outputs"):
    shutil.rmtree("model_training_outputs")
    print("Deleted 'model_training_outputs' to free up disk space.")

# Now, we have plenty of disk space free – save the 19GB model (merged 16-bit) to disk.
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
