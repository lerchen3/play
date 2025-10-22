#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Aggregate QFilter metrics")
    parser.add_argument("--results-dir", type=Path, default=Path("analysis/qfilter"))
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--prompts-path", type=Path, default=Path("analysis/aime_prompts.json"))
    parser.add_argument("--output", type=Path, default=Path("analysis/qfilter_summary.json"))
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompts: List[str],
    generations: List[str],
) -> float:
    total_neg_log_lik = 0.0
    total_tokens = 0
    for prompt, generation in zip(prompts, generations):
        if not generation:
            continue
        combined = prompt + generation
        inputs = tokenizer(combined, return_tensors="pt").to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]
        labels = inputs.input_ids.clone()
        labels[:, :prompt_len] = -100
        with torch.no_grad():
            outputs = model(input_ids=inputs.input_ids, labels=labels)
        token_count = (labels != -100).sum().item()
        total_neg_log_lik += outputs.loss.item() * token_count
        total_tokens += token_count
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_neg_log_lik / total_tokens)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
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

    prompts = json.loads(args.prompts_path.read_text())

    total_original = 0
    total_kept = 0
    generations: List[str] = []

    for metrics_path in sorted(args.results_dir.glob("metrics_rank*.json")):
        payload = json.loads(metrics_path.read_text())
        total_original += payload.get("original_entries", 0)
        total_kept += payload.get("kept_entries", 0)

    for gens_path in sorted(args.results_dir.glob("generations_rank*.json")):
        payload = json.loads(gens_path.read_text())
        generations.extend(payload.get("generations", []))

    generation_prompts = prompts[: len(generations)]
    perplexity = compute_perplexity(model, tokenizer, device, generation_prompts, generations)
    compression_ratio = (total_kept / total_original) if total_original else 0.0

    summary = {
        "compression_ratio": compression_ratio,
        "perplexity": perplexity,
        "original_entries": total_original,
        "kept_entries": total_kept,
        "num_generations": len(generations),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
