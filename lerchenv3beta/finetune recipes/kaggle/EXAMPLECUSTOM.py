# Minimal Working Example: Integrating Unsloth with a custom TRL SFTTrainer subclass
import os
os.environ["WANDB_DISABLED"] = "true" # Keep W&B disabled as before

# 1. Import required libraries
from unsloth import FastLanguageModel  # Unsloth's optimized model class
from trl import SFTTrainer, SFTConfig  # TRL's SFT (Supervised Fine-Tuning) trainer and config
from datasets import Dataset          # Hugging Face Datasets for creating dummy data
import torch                          # PyTorch for tensor operations

# 2. Define a custom SFTTrainer subclass that applies token-level weights to the loss
class SequenceWiseWeightedSFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer that applies per-token weights from the dataset (`token_weights`)
    to compute a weighted loss for each sequence.
    """
    # --- MODIFIED LINE: Added 'num_items_in_batch=None' to the signature ---
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # ---------------------------------------------------------------------
        """
        Computes the loss using token-level weights.

        Args:
            model: The model being trained.
            inputs (dict): The input batch from the data collator. Must contain
                           'input_ids', 'labels', and 'token_weights'.
            return_outputs (bool): Whether to return the model's outputs along with the loss.
            num_items_in_batch (Optional[int]): Added for compatibility with the trainer's
                                                internal training step, but not used in
                                                this custom loss calculation.
        """
        # Get labels and token_weights from the batch
        labels = inputs.get("labels")
        weights = inputs.get("token_weights")

        # If we have both labels and weights, compute weighted loss
        if labels is not None and weights is not None:
            # Move labels and weights to the same device as the model
            labels = labels.to(model.device)
            weights = weights.to(model.device)

            # Forward pass: Ensure labels are not passed to the model directly
            # if we are computing custom loss based on logits.
            model_inputs = {
                "input_ids": inputs["input_ids"].to(model.device),
                "attention_mask": inputs.get("attention_mask", None).to(model.device) if inputs.get("attention_mask") is not None else None,
                # Do NOT pass 'labels' here, we compute loss from logits
            }
            outputs = model(**model_inputs)
            logits = outputs.get("logits") # Use .get for safety, though logits should exist
            # logits shape: [batch_size, sequence_length, vocab_size]
            # wait in particular GKD and offline-ppo should both work splendid lmfao

            if logits is None:
                 raise ValueError("Model did not return logits. Cannot compute custom loss.")

            # Shift logits and labels so that logits_{i} is used to predict label_{i+1}
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = weights[..., 1:].contiguous()

            # Compute per-token cross entropy loss (no reduction yet, ignore padding tokens)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

            # Flatten the predictions and labels for computation
            # Ensure correct dimensions: [batch*seq_len, vocab_size] and [batch*seq_len]
            flat_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_token_loss = flat_loss.view(shift_labels.size())  # shape [batch, seq_len-1]

            # Apply the token-level weights
            weighted_per_token_loss = per_token_loss * shift_weights

            # Zero-out any loss where label is -100 (padding or ignored positions)
            # Note: CrossEntropyLoss with ignore_index already effectively does this,
            # but applying weights might make explicit zeroing necessary if weights aren't 0 there.
            # Let's recalculate the mask *after* shifting, as shift_labels contains the relevant -100s.
            valid_mask = (shift_labels != -100)
            weighted_per_token_loss = weighted_per_token_loss * valid_mask.float() # Ensure multiplication zeros out invalid positions

            # Compute the average loss, weighted by the token weights
            # (divide by sum of weights of valid tokens to get a normalized mean loss)
            # Make sure total_weight reflects only valid, weighted tokens
            total_weight = (shift_weights * valid_mask).sum()

            if total_weight.item() > 0:
                loss = weighted_per_token_loss.sum() / total_weight
            else:
                # Handle edge case: no valid tokens or all weights zero
                loss = torch.tensor(0.0, device=weighted_per_token_loss.device, requires_grad=True) # Ensure requires_grad if needed

            # Prepare the return value based on return_outputs
            # Ensure the outputs dict includes loss if required by downstream processing
            if not isinstance(outputs, dict):
                 # If model output is not a dict (e.g., just logits tensor), create one
                 outputs = {"logits": outputs}

            # Store the computed loss in the outputs dict if needed later (e.g., for evaluation)
            # The trainer primarily uses the returned loss scalar, but adding it here is good practice.
            outputs["loss"] = loss

            return (loss, outputs) if return_outputs else loss
        else:
            # Fallback to default loss computation if labels or weights are missing
            # This requires passing labels to the model
            return super().compute_loss(model, inputs, return_outputs=return_outputs)


# 3. Load and patch a model using Unsloth before training
#    We use FastLanguageModel.from_pretrained to get a model and tokenizer, patched with Unsloth optimizations.
#    (Assuming the path '/kaggle/input/...' is correct in your environment)
try:
    orig_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='/kaggle/input/deepseek-r1-distill-qwen-7b-unsloth-bnb-4bit/transformers/default/1/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit',
        max_seq_length=100,
        dtype=None, # Autodetected by Unsloth
        trust_remote_code=True,
        load_in_4bit=True,
    )
except Exception as e:
     print(f"Error loading model: {e}")
     print("Please ensure the model path is correct and accessible.")
     # Handle error appropriately, maybe exit or use a default model if possible
     raise e # Re-raise the exception if loading is critical

tokenizer.pad_token = tokenizer.eos_token
# Set padding side if needed, often 'right' for Causal LM training with TRL
tokenizer.padding_side = 'right'

# Apply LoRA to the model
model = FastLanguageModel.get_peft_model(
    orig_model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth", # Recommended: "unsloth" or True
    random_state=2009,
    use_rslora=False,
    loftq_config=None,
)

# 4. Create a dummy dataset with 'text' and 'token_weights' fields
data = [
    {"text": "Hello world",                      "token_weights": [1.0, 1.0]},       # simple example
    {"text": "Hi there",                        "token_weights": [1.0, 0.5]},       # second token weighted 0.5
    {"text": "This is a longer dummy sentence", "token_weights": [1.0, 1.0, 1.0, 1.0, 1.0]}  # weights for each token (will be padded later)
]
dataset = Dataset.from_list(data)

# 5. Tokenize the dataset and align token_weights with tokenized inputs
MAX_LENGTH = 16 # Define max length clearly
def tokenize_and_align(example):
    # Tokenize the text
    encoded = tokenizer(
        example["text"],
        padding="max_length",   # pad all examples to max_length
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None, # Return lists for easier processing before batching
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Prepare labels identical to input_ids for causal LM, and mark padding tokens to ignore
    labels = input_ids[:] # Create a copy
    # Mark padding tokens as -100
    labels = [label if mask == 1 else -100 for label, mask in zip(labels, attention_mask)]

    # Align token_weights to the sequence length (MAX_LENGTH):
    weights_input = example["token_weights"]
    weights_aligned = weights_input[:MAX_LENGTH] # Truncate if longer
    # Pad with 0.0 for the remaining length up to MAX_LENGTH
    weights_aligned += [0.0] * (MAX_LENGTH - len(weights_aligned))

    # Ensure weights corresponding to padding tokens are 0
    # (This should already happen if padding with 0.0, but explicit check is safer)
    weights_final = [w if mask == 1 else 0.0 for w, mask in zip(weights_aligned, attention_mask)]

    # It's generally recommended NOT to ignore the first token's label for SFT,
    # unless you have a specific reason (like sequence classification pre-training).
    # The standard Causal LM loss shift handles the prediction alignment.
    # Commenting out the first token ignore:
    # if len(labels) > 0:
    #    labels[0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "token_weights": weights_final # Use the fully aligned weights
    }

# Apply the tokenization and alignment to the entire dataset
processed_dataset = dataset.map(tokenize_and_align, remove_columns=["text"])

# 6. Configure training arguments using TRL's SFTConfig
training_args = SFTConfig(
    output_dir="outputs",              # directory to save model checkpoints
    num_train_epochs=1,                # train for 1 epoch (for demo)
    per_device_train_batch_size=2,     # batch size per GPU/CPU
    gradient_accumulation_steps=1,     # Adjust if needed for larger effective batch size
    logging_steps=1,                   # log training loss every step (for demonstration)
    remove_unused_columns=False,       # CRITICAL: Keep 'token_weights'
    dataset_text_field="",             # No raw text field needed after our processing
    packing=False,                     # Set packing=False as we are doing manual padding/labeling
    max_seq_length=MAX_LENGTH,         # Ensure SFTConfig knows the max length
    # dataset_kwargs={"skip_prepare_dataset": True} # This is often not needed if dataset_text_field="" and packing=False
    report_to="none",                  # Disable W&B/other reporting
    learning_rate=2e-4,                # Example LR, adjust as needed
    optim="adamw_8bit",                # Example optimizer
)

# 7. Define a custom data collator to batch the data
# Since SFTTrainer uses DataCollatorForCompletionOnlyLM by default if packing=False,
# and our data is already prepared, a simple default collator might work,
# but defining one explicitly ensures 'token_weights' is handled.
from transformers import default_data_collator
def custom_data_collator(features):
    # Use default collator for standard fields (input_ids, attention_mask, labels)
    batch = default_data_collator(features)

    # Manually handle 'token_weights' if present
    if "token_weights" in features[0]:
        weights = [f["token_weights"] for f in features]
        batch["token_weights"] = torch.tensor(weights, dtype=torch.float32)

    return batch


# 8. Initialize the custom trainer with the patched model and processed dataset
trainer = SequenceWiseWeightedSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,
    data_collator=custom_data_collator,  # Use the custom collator
    # We don't need dataset_text_field here because we manually processed
)

# 9. Train the model on the dummy dataset
print("Starting training...")
try:
    trainer.train()
    print("Training finished successfully.")
except Exception as e:
    print(f"An error occurred during training: {e}")
    # Potentially print more debugging info or re-raise
    raise e

# (After training, the model's weights have been updated according to the weighted loss.)