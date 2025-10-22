#!/usr/bin/env python
import argparse
import importlib.util
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
QFILTERS_SRC = ROOT / "qfilters" / "src"


def load_qfilters_cache_class():
    utils_spec = importlib.util.spec_from_file_location("qfilters_utils", QFILTERS_SRC / "utils.py")
    utils_module = importlib.util.module_from_spec(utils_spec)
    assert utils_spec.loader is not None
    utils_spec.loader.exec_module(utils_module)

    src_module = types.ModuleType("src")
    src_module.__path__ = [str(QFILTERS_SRC)]
    src_module.utils = utils_module
    sys.modules["src"] = src_module
    sys.modules["src.utils"] = utils_module

    cache_spec = importlib.util.spec_from_file_location("qfilters_hf_cache", QFILTERS_SRC / "hf_cache.py")
    cache_module = importlib.util.module_from_spec(cache_spec)
    assert cache_spec.loader is not None
    cache_spec.loader.exec_module(cache_module)
    return cache_module.QFiltersCache


QFiltersCache = load_qfilters_cache_class()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run QFilters decoding experiment")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--prompts-path", type=Path, default=Path("analysis/aime_prompts.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/qfilter"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.manual_seed(args.seed + rank)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    prompts = json.loads(args.prompts_path.read_text())[: args.num_prompts]
    prompts_assigned = [prompt for idx, prompt in enumerate(prompts) if idx % world_size == rank]
    if not prompts_assigned:
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

    metrics = {
        "rank": rank,
        "world_size": world_size,
        "max_length": args.max_length,
        "window_length": args.window_length,
        "original_entries": 0,
        "kept_entries": 0,
        "total_prompts": len(prompts_assigned),
    }

    generations: List[str] = []

    for prompt_idx, prompt in enumerate(prompts_assigned):
        if args.time_limit and (time.time() - start_time) > args.time_limit:
            break

        cache = QFiltersCache(
            max_length=args.max_length,
            window_length=args.window_length,
            model_name=args.model_name,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                past_key_values=cache,
            )

        generated_len = output_ids.shape[1] - prompt_len
        generated_text = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
        generations.append(generated_text)

        num_heads = cache.key_cache[0].shape[1] if cache.key_cache else model.config.num_key_value_heads
        layer_entries = 0
        for layer_cache in cache.key_cache:
            layer_entries += layer_cache.shape[2] * num_heads

        original_per_layer = (prompt_len + generated_len) * num_heads
        total_layers = len(cache.key_cache)

        metrics["original_entries"] += original_per_layer * total_layers
        metrics["kept_entries"] += layer_entries

    metrics_path = args.output_dir / f"metrics_rank{rank}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    generations_path = args.output_dir / f"generations_rank{rank}.json"
    generations_path.write_text(json.dumps({"generations": generations, "rank": rank}, indent=2))


if __name__ == "__main__":
    main()
