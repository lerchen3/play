#!/usr/bin/env python
"""Test LSE epsilon values for whitened chunk merging"""
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

from src.whitened_chunk_cache import WhitenedChunkCache


def parse_args():
    parser = argparse.ArgumentParser("Test LSE epsilon values")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--prompts-path", type=Path, default=Path("analysis/aime_prompts.json"))
    parser.add_argument("--stats-path", type=Path, default=Path("analysis/qk_stats.pt"))
    parser.add_argument("--num-prompts", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--epsilons", type=float, nargs="+", 
                        default=[35, 40, 45, 50, 55, 60, 65, 70])
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
    """
    Compute perplexity of generated text using the ORIGINAL MODEL (no cache compression).
    This measures how good the generated text is according to the uncompressed model.
    """
    full_text = prompt + " " + generated_text
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
    # Forward pass with NO cache (original model)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        neg_log_likelihood = outputs.loss.item()
    
    perplexity = torch.exp(torch.tensor(neg_log_likelihood)).item()
    return perplexity


def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompt: str,
    epsilon: float,
    stats_path: Path,
    max_new_tokens: int,
):
    """Run inference with given epsilon and return metrics"""
    cache = WhitenedChunkCache(
        epsilon=epsilon,
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        num_query_heads=model.config.num_attention_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        stats_path=stats_path,
        query_cache_path=Path("dumps/deepseek_r1_qkv"),
        query_bank_size=16384,
        solver_query_count=1024,
        seed=0,
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            past_key_values=cache,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Finalize all open chunks
    cache.finalize_all_chunks()
    
    generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Compute perplexity using UNCOMPRESSED model (no cache passed)
    # This measures: how good is the generated text according to the original model?
    perplexity = compute_perplexity(model, tokenizer, device, prompt, generated_text)
    
    # Compute metrics
    prefill_tokens = cache.prefill_tokens
    decode_tokens = cache.decode_tokens
    total_tokens = prefill_tokens + decode_tokens
    
    # Get chunk size distribution and count total chunks
    all_chunk_sizes = []
    total_chunks = 0
    for layer_chunks in cache._chunk_sizes:
        for head_chunks in layer_chunks:
            all_chunk_sizes.extend(head_chunks)
            total_chunks += len(head_chunks)
    
    num_layers = cache.num_layers
    num_kv_heads = cache.num_kv_heads
    original_size = total_tokens * num_layers * num_kv_heads
    compression_ratio = total_chunks / original_size if original_size > 0 else 0.0
    
    avg_chunk_size = sum(all_chunk_sizes) / len(all_chunk_sizes) if all_chunk_sizes else 0.0
    
    # Total entries = total tokens across all chunks
    total_entries = sum(all_chunk_sizes)
    
    # Verify: avg_chunk_size * total_chunks should equal total_entries
    # And: compression_ratio should equal total_chunks / original_size
    # So: avg_chunk_size should equal total_entries / total_chunks = original_size / total_chunks * compression_ratio
    #     = 1 / compression_ratio (if total_entries == original_size, which it should)
    
    return {
        "epsilon": epsilon,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "total_tokens": total_tokens,
        "total_chunks": total_chunks,
        "total_entries": total_entries,
        "original_size": original_size,
        "compression_ratio": compression_ratio,
        "avg_chunk_size": avg_chunk_size,
        "max_chunk_size": max(all_chunk_sizes) if all_chunk_sizes else 0,
        "min_chunk_size": min(all_chunk_sizes) if all_chunk_sizes else 0,
        "perplexity": perplexity,
    }


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
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
    
    # Load prompts
    prompts = load_prompts(args.prompts_path, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts")
    
    # Test each epsilon
    results = []
    for epsilon in args.epsilons:
        print(f"\nTesting epsilon={epsilon}...")
        
        epsilon_metrics = []
        
        for prompt_idx, prompt in enumerate(prompts):
            try:
                metrics = run_inference(
                    model, tokenizer, device, prompt,
                    epsilon, args.stats_path, args.max_new_tokens
                )
                epsilon_metrics.append(metrics)
                
                print(f"  Prompt {prompt_idx + 1}/{len(prompts)}: "
                      f"tokens={metrics['total_tokens']}, "
                      f"compression={metrics['compression_ratio']:.1%}, "
                      f"ppl={metrics['perplexity']:.4f}")
            except Exception as e:
                print(f"  ERROR on prompt {prompt_idx + 1}: {e}")
                continue
        
        # Compute averages for this epsilon
        if epsilon_metrics:
            avg_result = {
                "epsilon": epsilon,
                "num_prompts": len(epsilon_metrics),
                "avg_tokens": sum(m['total_tokens'] for m in epsilon_metrics) / len(epsilon_metrics),
                "avg_compression": sum(m['compression_ratio'] for m in epsilon_metrics) / len(epsilon_metrics),
                "avg_chunk_size": sum(m['avg_chunk_size'] for m in epsilon_metrics) / len(epsilon_metrics),
                "avg_perplexity": sum(m['perplexity'] for m in epsilon_metrics) / len(epsilon_metrics),
                "min_perplexity": min(m['perplexity'] for m in epsilon_metrics),
                "max_perplexity": max(m['perplexity'] for m in epsilon_metrics),
                "all_results": epsilon_metrics,
            }
            results.append(avg_result)
            print(f"  Average: compression={avg_result['avg_compression']:.1%}, ppl={avg_result['avg_perplexity']:.4f}")
    
    # Save results
    output_path = Path("analysis/lse_epsilon_sweep.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Summary (Averaged over {args.num_prompts} prompts) ===")
    print(f"{'Epsilon':<10} | {'Prompts':<8} | {'Avg Tokens':<11} | {'Compression':<12} | {'Avg Chunk':<10} | {'Avg PPL':<12}")
    print("-" * 90)
    for r in results:
        print(f"{r['epsilon']:<10.1f} | {r['num_prompts']:<8} | {r['avg_tokens']:<11.1f} | "
              f"{r['avg_compression']:>10.1%} | {r['avg_chunk_size']:>10.2f} | {r['avg_perplexity']:>12.4f}")
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
