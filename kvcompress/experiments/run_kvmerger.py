#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.kvmerger_cache import KVMergerCache


def parse_args():
    parser = argparse.ArgumentParser("Run KVMerger inference experiments")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--prompts-path", type=Path, default=Path("analysis/aime_prompts.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/kvmerger"))
    parser.add_argument("--use-whitening", action="store_true", help="Use whitened keys")
    parser.add_argument("--stats-path", type=Path, default=Path("analysis/qk_stats.pt"))
    parser.add_argument("--num-prompts", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--merge-interval", type=int, default=4)
    parser.add_argument("--merge-window", type=int, default=8)
    parser.add_argument("--cosine-threshold", type=float, default=None, help="Cosine similarity threshold (min similarity to merge)")
    parser.add_argument("--l2-threshold", type=float, default=None, help="L2 distance threshold (max distance to merge)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian kernel bandwidth for KVMerger")
    parser.add_argument("--time-limit", type=float, default=None)
    return parser.parse_args()


def load_prompts(path: Path, limit: int) -> List[str]:
    prompts = json.loads(path.read_text())
    return prompts[:limit]


def setup_attention_hooks(model: AutoModelForCausalLM, cache: KVMergerCache) -> List:
    """Setup forward hooks to capture attention matrices and pass to cache"""
    hooks = []
    
    def attention_hook(module, input, output):
        # output is typically (hidden_states, attention_weights, past_key_values)
        if len(output) >= 2 and isinstance(output[1], torch.Tensor):
            attention_weights = output[1]  # Shape: [batch, heads, seq_len, seq_len]
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # Extract layer index from module name
            layer_idx = int(module.__class__.__name__.split('_')[-1]) if 'layer' in str(module) else 0
            
            # Process each head
            for head_idx in range(num_heads):
                # Get attention matrix for this head
                attn_matrix = attention_weights[0, head_idx, :, :]  # [seq_len, seq_len]
                
                # Update cache attention scores
                cache.update_attention_scores(layer_idx, head_idx, attn_matrix)
    
    # Register hooks on attention layers
    for name, module in model.named_modules():
        if 'attention' in name.lower() and hasattr(module, 'forward'):
            hook = module.register_forward_hook(attention_hook)
            hooks.append(hook)
    
    return hooks


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompts: List[str],
    generations: List[str],
) -> float:
    total_neg_log_lik = 0.0
    total_tokens = 0
    for prompt, gen in zip(prompts, generations):
        if not gen:
            continue
        text = prompt + gen
        inputs = tokenizer(text, return_tensors="pt").to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]
        labels = inputs.input_ids.clone()
        labels[:, :prompt_len] = -100
        with torch.no_grad():
            output = model(input_ids=inputs.input_ids, labels=labels)
        num_tokens = (labels != -100).sum().item()
        total_neg_log_lik += output.loss.item() * num_tokens
        total_tokens += num_tokens
    
    if total_tokens == 0:
        return float('inf')
    avg_neg_log_lik = total_neg_log_lik / total_tokens
    return float(torch.exp(torch.tensor(avg_neg_log_lik)).item())


def main():
    args = parse_args()
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",  # Disable flash attention to materialize attention matrix
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompts
    prompts = load_prompts(args.prompts_path, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts")
    
    # Run inference
    mode = "whitened" if args.use_whitening else "original"
    print(f"Running KVMerger ({mode} mode)...")
    
    generations = []
    
    for prompt_idx, prompt in enumerate(prompts):
        if time.time() - start_time > args.time_limit:
            break
        
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
        
        # Setup attention hooks to capture attention matrices
        hooks = setup_attention_hooks(model, cache)
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    past_key_values=cache,
                    pad_token_id=tokenizer.pad_token_id,
                )
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generations.append(generated_text)
        
        print(f"  Completed {prompt_idx + 1}/{len(prompts)}")
    
    # Compute metrics
    stats = cache.get_stats()
    perplexity = compute_perplexity(model, tokenizer, device, prompts[:len(generations)], generations)
    
    metrics = {
        "mode": mode,
        "merge_interval": args.merge_interval,
        "merge_window": args.merge_window,
        "cosine_threshold": args.cosine_threshold,
        "l2_threshold": args.l2_threshold,
        "num_generations": len(generations),
        "decode_tokens": stats["decode_tokens"],
        "total_entries": stats["total_entries"],
        "total_merged": stats["total_merged"],
        "compression_ratio": stats["compression_ratio"],
        "perplexity": perplexity,
    }
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"metrics_{mode}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    gens_path = args.output_dir / f"generations_{mode}.json"
    with open(gens_path, "w") as f:
        json.dump({"mode": mode, "generations": generations}, f, indent=2)
    
    print(f"\n=== Results ===")
    print(f"Mode: {mode}")
    print(f"Decode tokens: {stats['decode_tokens']}")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Compression ratio: {stats['compression_ratio']:.2%}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()

