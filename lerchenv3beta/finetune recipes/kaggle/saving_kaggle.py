model_pth = "/kaggle/input/hihihir1ong/model_training_outputs/checkpoint-324"
output_pth = "r1-5-epochs-veri"
SFT_CSV_PATH = "/kaggle/input/anchnotsftdata/anchnot_sft_data.csv"
NUM_ROWS = 1000
BATCH_NUM = 4
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

model.save_pretrained_merged(output_pth, tokenizer, save_method="merged_16bit")