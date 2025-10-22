#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lse_cache_correct import LSECacheCorrect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run LSE cache (correct) decoding experiments")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--prompts-path", type=Path, default=Path("analysis/aime_prompts.json"))
    parser.add_argument("--stats-path", type=Path, default=Path("analysis/qk_stats.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/lse_correct"))
    parser.add_argument("--dump-dir", type=Path, default=Path("dumps/lse_correct"))
    parser.add_argument("--epsilons", nargs="*", type=float, default=None)
    parser.add_argument("--num-prompts", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--projection-rank", type=int, default=8)
    parser.add_argument("--projection-cache-dir", type=Path, default=Path("analysis/projections"))
    parser.add_argument("--query-cache-dir", type=Path, default=Path("dumps/deepseek_r1_qkv"))
    parser.add_argument("--query-bank-size", type=int, default=16384)
    parser.add_argument("--solver-query-count", type=int, default=1024)
    parser.add_argument("--max-prefill-queries", type=int, default=1024)
    parser.add_argument("--max-decode-queries", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-4)
    return parser.parse_args()


def load_prompts(path: Path, limit: int) -> List[str]:
    prompts = json.loads(path.read_text())
    return prompts[:limit]


def assign_epsilons(eps_list: List[float], rank: int, world_size: int) -> List[float]:
    return [eps for idx, eps in enumerate(sorted(eps_list)) if idx % world_size == rank]


def format_epsilon(epsilon: float) -> str:
    return f"{epsilon:.2e}"


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
        return float("inf")
    return math.exp(total_neg_log_lik / total_tokens)


def main() -> None:
    args = parse_args()
    start_time = time.time()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    prompts = load_prompts(args.prompts_path, args.num_prompts)

    eps_list = args.epsilons if args.epsilons else [0.01, 0.1, 1.0, 10.0, 100.0]
    work_eps = assign_epsilons(eps_list, rank, world_size)
    if not work_eps:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            device_map={"": device},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.dump_dir.mkdir(parents=True, exist_ok=True)

    for epsilon in work_eps:
        if args.time_limit and (time.time() - start_time) > args.time_limit:
            break

        eps_str = format_epsilon(epsilon)
        run_output_dir = args.output_dir / f"epsilon_{eps_str}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        run_dump_dir = args.dump_dir / f"epsilon_{eps_str}"
        run_dump_dir.mkdir(parents=True, exist_ok=True)

        prompt_generations: List[str] = []
        accumulated_tokens = 0
        accumulated_chunks = 0
        chunk_records: List[Dict[str, Dict[str, List[int]]]] = []
        last_head_lengths: Dict[int, List[int]] | None = None

        for prompt_idx, prompt in enumerate(prompts):
            if args.time_limit and (time.time() - start_time) > args.time_limit:
                break

            cache = LSECacheCorrect(
                epsilon=epsilon,
                num_layers=model.config.num_hidden_layers,
                num_kv_heads=model.config.num_key_value_heads,
                num_query_heads=model.config.num_attention_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
                stats_path=args.stats_path,
                projection_cache_path=args.projection_cache_dir,
                query_cache_path=args.query_cache_dir,
                query_bank_size=args.query_bank_size,
                solver_query_count=args.solver_query_count,
                max_prefill_queries=args.max_prefill_queries,
                max_decode_queries=args.max_decode_queries,
                ridge=args.ridge,
                rank=args.projection_rank,
                seed=args.seed + rank,
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

            cache.finalize_all_chunks()
            stats = cache.get_chunk_stats()
            accumulated_tokens += stats.get("decode_tokens", 0)
            accumulated_chunks += stats.get("total_chunks", 0)
            chunk_records.append(stats.get("chunks_per_layer", {}))
            head_lengths = stats.get("head_lengths")
            if isinstance(head_lengths, dict):
                last_head_lengths = head_lengths

            generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            prompt_generations.append(generated_text)

            dump_path = run_dump_dir / f"rank{rank}_prompt{prompt_idx:02d}.json"
            dump_payload = {
                "prompt_index": prompt_idx,
                "prompt": prompt,
                "generation": generated_text,
                "stats": stats,
            }
            dump_path.write_text(json.dumps(dump_payload, indent=2))

            log_path = run_dump_dir / f"rank{rank}_prompt{prompt_idx:02d}_log.txt"
            with log_path.open("w") as log_f:
                log_f.write(f"Epsilon: {epsilon}\n")
                log_f.write(f"Prompt index: {prompt_idx}\n")
                log_f.write(f"Decode tokens: {stats.get('decode_tokens', 0)}\n")
                log_f.write(f"Total chunks: {stats.get('total_chunks', 0)}\n")
                log_f.write(f"Avg chunk size: {stats.get('avg_chunk_size', 0.0):.4f}\n")
                log_f.write(f"Compression ratio: {stats.get('compression_ratio', 0.0):.6f}\n")
                log_f.write("Chunks per layer:\n")
                chunks_per_layer = stats.get("chunks_per_layer", {})
                for layer_id, chunk_count in chunks_per_layer.items():
                    log_f.write(f"  Layer {layer_id}: {chunk_count}\n")

        if not prompt_generations:
            continue

        perplexity = compute_perplexity(
            model,
            tokenizer,
            device,
            prompts[: len(prompt_generations)],
            prompt_generations,
        )
        compression_ratio = (
            accumulated_chunks / max(1, accumulated_tokens) if accumulated_tokens > 0 else 1.0
        )

        aggregate_chunks: Dict[int, int] = {}
        for record in chunk_records:
            for layer_id, chunk_count in record.items():
                aggregate_chunks[layer_id] = aggregate_chunks.get(layer_id, 0) + chunk_count

        metrics = {
            "epsilon": epsilon,
            "rank": rank,
            "world_size": world_size,
            "num_generations": len(prompt_generations),
            "decode_tokens": accumulated_tokens,
            "decode_chunks": accumulated_chunks,
            "compression_ratio": compression_ratio,
            "perplexity": perplexity,
            "chunk_records": chunk_records,
            "aggregate_chunks_per_layer": aggregate_chunks,
            "final_head_lengths": last_head_lengths,
        }

        metrics_path = run_output_dir / f"metrics_rank{rank}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))

        gens_path = run_output_dir / f"generations_rank{rank}.json"
        gens_payload = {
            "epsilon": epsilon,
            "rank": rank,
            "generations": prompt_generations,
        }
        gens_path.write_text(json.dumps(gens_payload, indent=2))

        log_root = Path("logs/lse_correct")
        log_root.mkdir(parents=True, exist_ok=True)
        summary_log = log_root / f"epsilon_{eps_str}_rank{rank}.log"
        with summary_log.open("w") as lf:
            lf.write(f"epsilon: {epsilon}\n")
            lf.write(f"rank: {args.projection_rank}\n")
            lf.write(f"num_prompts: {len(prompt_generations)}\n")
            lf.write(f"decode_tokens: {accumulated_tokens}\n")
            lf.write(f"decode_chunks: {accumulated_chunks}\n")
            lf.write(f"compression_ratio: {compression_ratio:.6f}\n")
            lf.write(f"perplexity: {perplexity:.6f}\n")
            lf.write("chunks_per_layer:\n")
            for layer_id in sorted(aggregate_chunks.keys()):
                lf.write(f"  layer_{layer_id}: {aggregate_chunks[layer_id]}\n")
            if last_head_lengths:
                lf.write("final_head_lengths:\n")
                for layer_id in sorted(last_head_lengths.keys()):
                    lengths = ",".join(str(x) for x in last_head_lengths[layer_id])
                    lf.write(f"  layer_{layer_id}: [{lengths}]\n")


if __name__ == "__main__":
    main()
