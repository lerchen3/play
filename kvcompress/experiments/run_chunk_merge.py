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

from experiments.chunk_merge_cache import ChunkMergeCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run chunk-merge decoding experiments")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--prompts-path", type=Path, default=Path("analysis/aime_prompts.json"))
    parser.add_argument("--stats-path", type=Path, default=Path("analysis/chunk_merge_stats.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/chunk_merge"))
    parser.add_argument("--dump-dir", type=Path, default=Path("dumps/chunk_merge"))
    parser.add_argument("--epsilons", nargs="*", type=float, default=None)
    parser.add_argument("--num-prompts", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--max-queries", type=int, default=1024)
    return parser.parse_args()


def derive_epsilons(stats_path: Path) -> List[float]:
    data = json.loads(stats_path.read_text())
    median = data.get("median", 1.0)
    base = float(median if math.isfinite(median) and median > 0 else 1.0)
    epsilons = [base / 100.0, base / 10.0, base, base * 10.0, base * 100.0]
    return epsilons


def assign_epsilons(eps_list: List[float], rank: int, world_size: int) -> List[float]:
    return [eps for idx, eps in enumerate(eps_list) if idx % world_size == rank]


def load_prompts(path: Path, limit: int) -> List[str]:
    prompts = json.loads(path.read_text())
    return prompts[:limit]


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
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    prompts = load_prompts(args.prompts_path, args.num_prompts)

    if args.epsilons:
        eps_list = args.epsilons
    else:
        eps_list = derive_epsilons(args.stats_path)

    eps_list = sorted(eps_list)
    work_eps = assign_epsilons(eps_list, rank, world_size)

    if not work_eps:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map={"": device},
    )
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
        error_records: List[Dict[str, Dict[str, float]]] = []
        debug_records: List[Dict[str, Dict[str, List[float]]]] = []
        query_length_records: List[Dict[str, Dict[str, List[int]]]] = []

        for prompt_idx, prompt in enumerate(prompts):
            if args.time_limit and (time.time() - start_time) > args.time_limit:
                break

            cache = ChunkMergeCache(
                epsilon=epsilon,
                num_layers=model.config.num_hidden_layers,
                num_kv_heads=model.config.num_key_value_heads,
                num_query_heads=model.config.num_attention_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
                max_queries=args.max_queries,
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

            cache.flush()
            summary = cache.summary()
            accumulated_tokens += summary["decode_tokens"]
            accumulated_chunks += summary["decode_chunks"]
            chunk_records.append(summary["chunk_sizes"])
            error_records.append(summary["avg_errors"])
            debug_records.append(summary["debug_errors"])
            query_length_records.append(summary["chunk_query_lengths"])

            generated_ids = output_ids[0, inputs.input_ids.shape[1] :]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            prompt_generations.append(generated_text)

            dump_path = run_dump_dir / f"rank{rank}_prompt{prompt_idx:02d}.json"
            dump_payload = {
                "prompt_index": prompt_idx,
                "prompt": prompt,
                "generation": generated_text,
                "summary": summary,
            }
            dump_path.write_text(json.dumps(dump_payload, indent=2))

        if not prompt_generations:
            continue

        perplexity = compute_perplexity(model, tokenizer, device, prompts[: len(prompt_generations)], prompt_generations)
        compression_ratio = (accumulated_chunks / accumulated_tokens) if accumulated_tokens > 0 else 0.0

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
            "avg_errors": error_records,
            "debug_errors": debug_records,
            "chunk_query_lengths": query_length_records,
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


if __name__ == "__main__":
    main()
