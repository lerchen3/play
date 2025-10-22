#!/usr/bin/env python
"""Test KVMerger with averaged metrics over multiple prompts"""
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

from baselines.kvmerger_cache import KVMergerCache


def parse_args():
    parser = argparse.ArgumentParser("Test KVMerger with averaging")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--prompts-path", type=Path, default=Path("analysis/aime_prompts.json"))
    parser.add_argument("--num-prompts", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--use-whitening", action="store_true")
    parser.add_argument("--stats-path", type=Path, default=Path("analysis/qk_stats.pt"))
    parser.add_argument("--cosine-threshold", type=float, default=None)
    parser.add_argument("--l2-threshold", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--merge-interval", type=int, default=4)
    parser.add_argument("--merge-window", type=int, default=8)
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
        attn_implementation="eager",  # For attention matrix capture
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompts = load_prompts(args.prompts_path, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts")
    
    mode = "whitened" if args.use_whitening else "original"
    print(f"\nRunning KVMerger ({mode})...")
    
    all_metrics = []
    
    for idx, prompt in enumerate(prompts):
        print(f"  Prompt {idx + 1}/{len(prompts)}...", end=" ")
        
        cache = KVMergerCache(
            num_layers=model.config.num_hidden_layers,
            num_kv_heads=model.config.num_key_value_heads,
            num_query_heads=model.config.num_attention_heads,
            head_dim=model.config.hidden_size // model.config.num_attention_heads,
            merge_interval=args.merge_interval,
            merge_window=args.merge_window,
            cosine_threshold=args.cosine_threshold,
            l2_threshold=args.l2_threshold,
            use_whitening=args.use_whitening,
            stats_path=args.stats_path if args.use_whitening else None,
            sigma=args.sigma,
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                past_key_values=cache,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Get cache stats
        num_layers = cache.num_layers
        num_kv_heads = cache.num_kv_heads
        decode_tokens = cache.decode_tokens
        total_entries = sum(len(cache._keys[l][h]) for l in range(num_layers) for h in range(num_kv_heads))
        original_size = decode_tokens * num_layers * num_kv_heads
        compression = total_entries / original_size if original_size > 0 else 0.0
        
        # Compute perplexity
        ppl = compute_perplexity(model, tokenizer, device, prompt, generated_text)
        
        metrics = {
            "tokens": decode_tokens,
            "total_entries": total_entries,
            "compression": compression,
            "perplexity": ppl,
        }
        all_metrics.append(metrics)
        
        print(f"tokens={decode_tokens}, compression={compression:.1%}, ppl={ppl:.4f}")
    
    # Compute averages
    avg_tokens = sum(m['tokens'] for m in all_metrics) / len(all_metrics)
    avg_compression = sum(m['compression'] for m in all_metrics) / len(all_metrics)
    avg_ppl = sum(m['perplexity'] for m in all_metrics) / len(all_metrics)
    
    print("\n" + "=" * 80)
    print(f"KVMerger ({mode}) - FINAL RESULTS")
    print("=" * 80)
    print(f"Prompts: {len(all_metrics)}")
    print(f"Avg tokens: {avg_tokens:.1f}")
    print(f"Avg compression: {avg_compression:.1%}")
    print(f"Avg perplexity: {avg_ppl:.4f}")
    print(f"PPL range: {min(m['perplexity'] for m in all_metrics):.4f} - {max(m['perplexity'] for m in all_metrics):.4f}")
    print("=" * 80)
    
    # Save results
    results = {
        "method": f"kvmerger_{mode}",
        "cosine_threshold": args.cosine_threshold,
        "l2_threshold": args.l2_threshold,
        "num_prompts": len(all_metrics),
        "avg_tokens": avg_tokens,
        "avg_compression": avg_compression,
        "avg_perplexity": avg_ppl,
        "min_perplexity": min(m['perplexity'] for m in all_metrics),
        "max_perplexity": max(m['perplexity'] for m in all_metrics),
        "all_metrics": all_metrics,
    }
    
    output_path = Path(f"analysis/kvmerger_{mode}_avg.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

