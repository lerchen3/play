#!/usr/bin/env python
"""
Run greedy decoding with the full cache and report perplexity metrics.
"""
import argparse
import json
import math
import time
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Baseline greedy decoding evaluation")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--prompts-path", type=Path, default=Path("analysis/aime_prompts.json"))
    parser.add_argument("--output-path", type=Path, default=Path("analysis/baseline_metrics.json"))
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_prompts(path: Path, limit: int) -> List[str]:
    prompts = json.loads(path.read_text())
    return prompts[:limit]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()

    prompts = load_prompts(args.prompts_path, args.num_prompts)

    generations: List[str] = []
    neg_log_lik_total = 0.0
    token_count_total = 0

    start_time = time.time()
    for idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = output_ids[0, inputs.input_ids.shape[1] :]
        generation = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generations.append(generation)

        text = prompt + generation
        perplexity_inputs = tokenizer(text, return_tensors="pt").to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        labels = perplexity_inputs.input_ids.clone()
        labels[:, :prompt_len] = -100
        with torch.no_grad():
            outputs = model(input_ids=perplexity_inputs.input_ids, labels=labels)

        num_tokens = (labels != -100).sum().item()
        neg_log_lik = outputs.loss.item() * num_tokens
        neg_log_lik_total += neg_log_lik
        token_count_total += num_tokens

        print(
            f"[BASELINE_PROMPT] idx={idx} tokens={num_tokens} "
            f"loss={outputs.loss.item():.6f} neg_log_lik={neg_log_lik:.4f}",
            flush=True,
        )

    perplexity = math.exp(neg_log_lik_total / token_count_total) if token_count_total > 0 else float("inf")
    elapsed = time.time() - start_time

    payload = {
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "total_tokens": token_count_total,
        "total_neg_log_lik": neg_log_lik_total,
        "perplexity": perplexity,
        "elapsed_seconds": elapsed,
        "generations": generations,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2))
    print(f"[BASELINE_DONE] perplexity={perplexity:.6f} tokens={token_count_total}")


if __name__ == "__main__":
    main()
