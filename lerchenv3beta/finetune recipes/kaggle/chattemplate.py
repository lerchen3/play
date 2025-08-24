import pandas as pd
from transformers import AutoTokenizer
import json

def main():
    # Load the CSV containing pre-formatted chat data
    df = pd.read_csv("wlogprobs_pre_chat_sft.csv")
    
    # Initialize the AutoTokenizer for the Qwen chat model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", use_fast=True)
    results = []
    
    # Process each row using the tokenizer's chat template
    for idx, row in df.iterrows():
        # Create a conversation in the format expected by the tokenizer
        messages = [
            {"role": "user", "content": row['Question']},
            {"role": "assistant", "content": row['Assistant_Response']}
        ]
        
        # Apply the chat template to format the conversation
        full_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=False
        )
        full_tokens = tokenizer.encode(full_text)
        assistant_tokens = tokenizer.encode(row['Assistant_Response'])
        occurrences = find_sublist(full_tokens, assistant_tokens)
        if len(occurrences) == 0:
            raise ValueError("Assistant response token sequence not found in full conversation.")
        elif len(occurrences) > 1:
            raise ValueError("Multiple occurrences of assistant response token sequence found.")
        assistant_start_idx = occurrences[0]
        assistant_end_idx = assistant_start_idx + len(assistant_tokens) - 1
        result = {
            "text": full_text,
            "assistant_start_idx": assistant_start_idx,
            "assistant_end_idx": assistant_end_idx,
            "pireflogprobs": row['logprobs']
        }
        results.append(result)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("wlogprobs_chat_formatted.csv", index=False)
    
    # For debugging: print the first few formatted conversations
    for idx, item in enumerate(results[:3]):
        print(f"Example {idx+1}:\n{item['text']}\nAssistant token indices: {item['assistant_start_idx']} to {item['assistant_end_idx']}\n")

def find_sublist(lst, sublst):
    indices = []
    n = len(sublst)
    for i in range(len(lst) - n + 1):
        if lst[i:i+n] == sublst:
            indices.append(i)
    return indices

if __name__ == "__main__":
    main()
