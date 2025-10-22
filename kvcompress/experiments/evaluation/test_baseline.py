#!/usr/bin/env python
"""Test baseline (no compression) perplexity"""
import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser("Test baseline perplexity")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--prompts-path", type=Path, default=Path("analysis/aime_prompts.json"))
    parser.add_argument("--num-prompts", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args()


def load_prompts(path: Path, limit: int) -> List[str]:
    prompts = json.loads(path.read_text())
    return prompts[:limit]


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompt: str,
    generated_text: str,
) -> float:
    """Compute perplexity using uncompressed model"""
    full_text = prompt + " " + generated_text
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        neg_log_likelihood = outputs.loss.item()
    
    perplexity = torch.exp(torch.tensor(neg_log_likelihood)).item()
    return perplexity


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompts = load_prompts(args.prompts_path, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts")
    
    print(f"\nGenerating with NO compression (baseline)...")
    
    all_perplexities = []
    all_tokens = []
    
    for idx, prompt in enumerate(prompts):
        print(f"  Prompt {idx + 1}/{len(prompts)}...", end=" ")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        actual_tokens = len(generated_ids)
        
        # Compute perplexity
        ppl = compute_perplexity(model, tokenizer, device, prompt, generated_text)
        
        all_perplexities.append(ppl)
        all_tokens.append(actual_tokens)
        
        print(f"tokens={actual_tokens}, ppl={ppl:.4f}")
    
    # Compute average
    avg_ppl = sum(all_perplexities) / len(all_perplexities)
    avg_tokens = sum(all_tokens) / len(all_tokens)
    
    print("\n" + "=" * 80)
    print("BASELINE (No Compression) - FINAL RESULTS")
    print("=" * 80)
    print(f"Prompts: {len(prompts)}")
    print(f"Avg tokens generated: {avg_tokens:.1f}")
    print(f"Avg perplexity: {avg_ppl:.4f}")
    print(f"Perplexity range: {min(all_perplexities):.4f} - {max(all_perplexities):.4f}")
    print("=" * 80)
    
    # Save results
    results = {
        "method": "baseline",
        "num_prompts": len(prompts),
        "avg_tokens": avg_tokens,
        "avg_perplexity": avg_ppl,
        "min_perplexity": min(all_perplexities),
        "max_perplexity": max(all_perplexities),
        "all_perplexities": all_perplexities,
        "all_tokens": all_tokens,
    }
    
    output_path = Path("analysis/baseline_perplexity.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

