#!/usr/bin/env python
"""Run a single method - clean implementation with fresh model per run"""
import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.whitened_chunk_cache import WhitenedChunkCache
from baselines.kvmerger_cache import KVMergerCache


def load_qfilters_cache():
    """Load QFiltersCache from qfilters/src"""
    QFILTERS_SRC = ROOT / "qfilters" / "src"
    
    utils_spec = importlib.util.spec_from_file_location("qfilters_utils", QFILTERS_SRC / "utils.py")
    utils_module = importlib.util.module_from_spec(utils_spec)
    utils_spec.loader.exec_module(utils_module)
    
    src_module = types.ModuleType("src")
    src_module.__path__ = [str(QFILTERS_SRC)]
    src_module.utils = utils_module
    sys.modules["src"] = src_module
    sys.modules["src.utils"] = utils_module
    
    cache_spec = importlib.util.spec_from_file_location("qfilters_hf_cache", QFILTERS_SRC / "hf_cache.py")
    cache_module = importlib.util.module_from_spec(cache_spec)
    cache_spec.loader.exec_module(cache_module)
    return cache_module.QFiltersCache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=35.0)
    parser.add_argument("--rank", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Config
    problem_indices = [5, 11, 17, 23, 29]  # Problems 6, 12, 18, 24, 30
    max_new_tokens = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=" * 80)
    print(f"Method: {args.method}")
    if args.method == "lse":
        print(f"Epsilon: {args.epsilon}")
    if args.rank:
        print(f"Rank: {args.rank}")
    print(f"Problems: {[i+1 for i in problem_indices]}")
    print(f"=" * 80)
    
    # LOAD FRESH MODEL
    print("Loading fresh model...")
    attn_impl = "eager" if "kvmerger" in args.method else "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompts
    all_prompts = json.loads(Path("analysis/aime_prompts.json").read_text())
    prompts = [all_prompts[i] for i in problem_indices]
    
    # Generate
    print("\nGenerating...")
    generations = []
    
    for idx, prompt in enumerate(prompts):
        print(f"  Problem {problem_indices[idx]+1} ({idx+1}/5)...", end=" ", flush=True)
        
        # Create cache for this problem
        cache = None
        if args.method == "baseline":
            pass
        elif args.method == "kvmerger_cosine":
            cache = KVMergerCache(
                num_layers=32, num_kv_heads=8, num_query_heads=32, head_dim=128,
                merge_interval=4, merge_window=8, cosine_threshold=0.709, sigma=1.0,
            )
        elif args.method == "kvmerger_l2":
            cache = KVMergerCache(
                num_layers=32, num_kv_heads=8, num_query_heads=32, head_dim=128,
                merge_interval=4, merge_window=8, l2_threshold=14.85,
                use_whitening=True, stats_path="analysis/qk_stats.pt", sigma=1.0,
            )
        elif args.method == "lse":
            cache = WhitenedChunkCache(
                epsilon=args.epsilon, num_layers=32, num_kv_heads=8,
                num_query_heads=32, head_dim=128,
                stats_path="analysis/qk_stats.pt", rank=args.rank,
                query_cache_path="dumps/deepseek_r1_qkv",
                query_bank_size=16384,
                solver_query_count=1024,
                seed=0,
            )
        elif args.method == "qfilter":
            QFiltersCache = load_qfilters_cache()
            cache = QFiltersCache(
                max_length=128,
                window_length=64,
                model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            )
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            if cache is None:
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                           do_sample=False, pad_token_id=tokenizer.pad_token_id)
            else:
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                           do_sample=False, past_key_values=cache,
                                           pad_token_id=tokenizer.pad_token_id)
        
        if hasattr(cache, 'finalize_all_chunks'):
            cache.finalize_all_chunks()
        
        generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
        gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generations.append(gen_text)
        print(f"{len(generated_ids)} tokens", flush=True)
    
    # Compute perplexities in batch on uncompressed model
    print("\nComputing perplexities...")
    perplexities = []
    for i, (prompt, gen) in enumerate(zip(prompts, generations)):
        full_text = prompt + " " + gen
        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            ppl = torch.exp(outputs.loss).item()
        
        perplexities.append(ppl)
        print(f"  Problem {problem_indices[i]+1}: ppl={ppl:.4f}")
    
    avg_ppl = sum(perplexities) / len(perplexities)
    print(f"\nAverage perplexity: {avg_ppl:.4f}")
    
    # Save results
    method_name = args.method
    if args.rank:
        method_name += f"_r{args.rank}"
    
    results = {
        "method": args.method,
        "epsilon": args.epsilon if args.method == "lse" else None,
        "rank": args.rank,
        "problem_indices": problem_indices,
        "perplexities": perplexities,
        "avg_perplexity": avg_ppl,
        "generations": generations,
    }
    
    output_path = Path(f"analysis/final_{method_name}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
