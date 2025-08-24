import re
import math
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
import pandas as pd
import random
import wandb
wandb.login(key="f9cca73107f008f4f576c7fc2362bbc2fed22b2c")
# --- Hyperparameters and Model Configuration ---
max_seq_length = 12291

# Load model and tokenizer (using a 4-bit model for fast inference)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="notebooka2b473e58c/goodadapter",  # QLORA STILL WORKING LOL YAY
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # Set False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank = 128,
    gpu_memory_utilization=0.22,  # Adjust if out of memory
)

# --- New Conversation Prompt Components ---
USER_SECOND_PROMPT = (
"Okay, now look at the summary you just wrote and then quote it verbatim step by step. "
"For each step in your summary, you must:\n\n"
"1. **QUOTE VERBATIM**: Start with 'Step X: I asserted that [exact quote from your summary]'\n\n"
"2. **RIGOROUS VERIFICATION**: Provide a complete, detailed mathematical proof that this step is correct. "
"Show every calculation, every logical inference, and every mathematical principle used. "
"Be extremely thorough - if multiplying by 10 increases a logarithm by 10, prove why. "
"If adding terms gives a specific sum like 10*11/2 = 55, show the arithmetic step by step.\n\n"
"3. **IMMEDIATE MISTAKE DETECTION**: If you find ANY error - computational, logical, or conceptual - "
"immediately state 'MISTAKE DETECTED' and stop the verification to explain the error in complete detail.\n\n"
"4. **COMPLETE ANALYSIS**: Every assertion in your summary must be fully proved from first principles. "
"Do not skip steps or assume anything is obvious. Verify every single mathematical claim.\n\n"
"5. **FINAL JUDGMENT**: After verifying every single step with rigorous proofs, box either:\n"
"   - '1' if the ENTIRE summary is mathematically correct and completely well-reasoned\n"
"   - '0' if there are ANY mistakes, logical gaps, unjustified steps, or incorrect reasoning\n\n"
"Be extremely careful and catch mistakes immediately. The verification must be mathematically bulletproof."
)
ASSISTANT_PREFILL = """I'll now carefully verify my summary step by step, quoting each assertion verbatim and then proving it rigorously.

**STEP-BY-STEP VERIFICATION:**

Step 1: I asserted that"""
def build_conversation_prompt(question: str, assistant_response: str) -> str:
    """
    Build the conversation prompt in the desired format.
    Takes a question and assistant_response, then asks the assistant to verify the summary.
    """
    full_text = "<｜begin▁of▁sentence｜>" + "<｜User｜>" + question + "<｜Assistant｜>"  + assistant_response + "<｜User｜>" + USER_SECOND_PROMPT + "<｜Assistant｜>" + ASSISTANT_PREFILL
    return full_text

def get_progress_dataset(input_csv = "semihard_advantages_clean.csv") -> Dataset:
    """
    Each example's prompt is built using our conversation format:
      <|User|> question
      <|Assistant|> summary text (from row['summary'])
      <|User|> [USER_SECOND_PROMPT asking for verification]
      <|Assistant|> [ASSISTANT_PREFILL for verification]
    
    The answer is whether the summary is correct (matches Is_Correct field).
    """
    semihard_all = pd.read_csv(input_csv)
    array_data = []
    for idx, row in semihard_all.iterrows():
        # Use the summary field as the assistant response to be verified
        summary_text = row['summary']
        prompt = build_conversation_prompt(row['Question'], summary_text)
        array_data.append({
            'prompt': prompt,
            'answer': row['Is_Correct']
        })
    semihard_df = pd.DataFrame(array_data)
    semihard_df.to_csv("semihard_df.csv", index=False)
    data = Dataset.from_pandas(semihard_df)
    return data

global step_number_lmao
step_number_lmao = -1
global logging_arr
logging_arr = []
def progress_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    This reward function checks if the assistant correctly identified whether the summary is correct.
    It looks for boxed '1' or '0' and compares with the expected correctness.
    """
    global step_number_lmao  # Ensure we are modifying the global counter
    logging_df = pd.DataFrame(logging_arr)
    logging_df.to_csv(f"logs/logging_df_{step_number_lmao + 1}.csv", index=False)
    step_number_lmao += 1
    responses = completions # u dumbass they r just strings
    rewards = []
    
    for idx, (resp, expected_correct) in enumerate(zip(responses, answer)):
        # Look for boxed 1 or 0 in the response
        import re
        boxed_matches = re.findall(r'oxed\{([01])\}', resp)
        
        if not boxed_matches:
            # No boxed answer found, give 0 reward
            rewards.append(0)
            logging_arr.append([{'expected': expected_correct, 'predicted': 'none', 'step number': step_number_lmao}])
            continue
            
        # Take the last boxed answer
        predicted_correct = int(boxed_matches[-1])
        
        # Reward 1 if prediction matches expected, 0 otherwise
        if (predicted_correct == 1 and expected_correct) or (predicted_correct == 0 and not expected_correct):
            rewards.append(1)
        else:
            rewards.append(0)
            
        logging_arr.append([{'expected': expected_correct, 'predicted': predicted_correct, 'step number': step_number_lmao}])
    
    return rewards

dataset = get_progress_dataset()
# --- GRPO Training Configuration ---
training_args = GRPOConfig(
    use_vllm=True,  # Use vLLM for fast inference
    learning_rate=5e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_steps = 100,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # Increase for smoother training if needed
    num_generations=8,  # Decrease if out of memory
    max_prompt_length=max_seq_length,
    num_train_epochs = 1,
    max_completion_length=128,
    save_steps=50,
    max_grad_norm=0.1,
    temperature = 1.25,
    report_to=["wandb"],  # Can use Weights & Biases if desired
    output_dir="outputs",
)

# Initialize the GRPO trainer with our custom reward function.
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[progress_reward_func],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained_merged("solve_model", tokenizer, save_method="merged_16bit")