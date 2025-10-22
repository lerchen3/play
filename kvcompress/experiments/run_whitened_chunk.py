#!/usr/bin/env python
"""
Run greedy decoding with the compressed cache and report compression/perplexity metrics.
"""
import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.whitened_noquery_cache import WhitenedNoQueryCache


class Tee(io.TextIOBase):
    def __init__(self, stream: io.TextIOBase, log_file: io.TextIOBase) -> None:
        self._stream = stream
        self._log = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run chunk-merge decoding experiments")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--prompts-path", type=Path, default=Path("analysis/aime_prompts.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/whitened_chunk"))
    parser.add_argument("--dump-dir", type=Path, default=Path("dumps/whitened_chunk"))
    parser.add_argument("--epsilons", nargs="+", type=float, required=True)
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument("--projection-rank", type=int, default=None)
    parser.add_argument("--log-cache-lengths", action="store_true", default=False)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    return parser.parse_args()


def load_prompts(path: Path, limit: int) -> List[str]:
    prompts = json.loads(path.read_text())
    return prompts[:limit]


def format_epsilon(epsilon: float) -> str:
    return f"{epsilon:.2f}".replace(".", "p")


def compute_prompt_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompt: str,
    generation: str,
) -> Dict[str, float]:
    text = prompt + generation
    inputs = tokenizer(text, return_tensors="pt").to(device)

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    labels = inputs.input_ids.clone()
    labels[:, :prompt_len] = -100
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, labels=labels)

    token_count = (labels != -100).sum().item()
    neg_log_lik = outputs.loss.item() * token_count
    return {
        "tokens": token_count,
        "neg_log_lik": neg_log_lik,
        "loss": outputs.loss.item(),
        "perplexity": math.exp(neg_log_lik / token_count) if token_count > 0 else float("inf"),
    }


def run_experiment(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompts: List[str],
    epsilon: float,
    projection_rank: Optional[int],
    max_new_tokens: int,
    output_dir: Path,
    dump_dir: Path,
    log_cache_lengths: bool,
    log_dir: Path,
) -> None:
    eps_tag = format_epsilon(epsilon)
    rank_tag = f"_rank{projection_rank}" if projection_rank is not None else ""
    run_dir = output_dir / f"epsilon_{eps_tag}{rank_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_subdir = dump_dir / f"epsilon_{eps_tag}{rank_tag}"
    dump_subdir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"epsilon_{eps_tag}{rank_tag}.log"

    total_tokens = 0
    total_neg_log_lik = 0.0
    prompt_generations: List[str] = []
    prompt_metrics: List[Dict[str, object]] = []
    chunk_records: List[Dict[str, Dict[str, int]]] = []
    layer_lengths_all: List[List[float]] = []
    with open(log_path, "w", encoding="utf-8") as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)
        try:
            for prompt_idx, prompt in enumerate(prompts):
                cache = WhitenedNoQueryCache(
                    epsilon=epsilon,
                    num_layers=model.config.num_hidden_layers,
                    num_kv_heads=model.config.num_key_value_heads,
                    num_query_heads=model.config.num_attention_heads,
                    head_dim=model.config.hidden_size // model.config.num_attention_heads,
                    stats_path="analysis/qk_stats.pt",
                    projection_rank=projection_rank,
                )
                cache.log_kv_lengths = log_cache_lengths

                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        past_key_values=cache,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                cache.finalize_all_chunks()

                generated_ids = output_ids[0, inputs.input_ids.shape[1] :]
                generation = tokenizer.decode(generated_ids, skip_special_tokens=True)
                prompt_generations.append(generation)

                ppl_stats = compute_prompt_perplexity(model, tokenizer, device, prompt, generation)
                total_tokens += ppl_stats["tokens"]
                total_neg_log_lik += ppl_stats["neg_log_lik"]

                layer_lengths = [
                    cache.get_seq_length(layer_idx) for layer_idx in range(model.config.num_hidden_layers)
                ]
                avg_len = sum(layer_lengths) / len(layer_lengths)
                baseline_len = inputs.input_ids.shape[1] + generated_ids.shape[0]
                compression_ratio = avg_len / baseline_len if baseline_len > 0 else 0.0
                layer_lengths_all.append(layer_lengths)

                stats = cache.get_chunk_stats()
                chunk_records.append(stats.get("chunks_per_layer", {}))

                prompt_metrics.append(
                    {
                        "prompt_index": prompt_idx,
                        "prefill_tokens": int(inputs.input_ids.shape[1]),
                        "decode_tokens": int(generated_ids.shape[0]),
                        "avg_sequence_length": avg_len,
                        "baseline_sequence_length": baseline_len,
                        "compression_ratio": compression_ratio,
                        "perplexity": ppl_stats["perplexity"],
                        "loss": ppl_stats["loss"],
                        "neg_log_lik": ppl_stats["neg_log_lik"],
                        "tokens": ppl_stats["tokens"],
                        "layer_lengths": layer_lengths,
                    }
                )

                dump_payload = {
                    "prompt_index": prompt_idx,
                    "prompt": prompt,
                    "generation": generation,
                    "metrics": prompt_metrics[-1],
                    "chunks_per_layer": stats.get("chunks_per_layer", {}),
                    "projection_rank": projection_rank,
                }
                (dump_subdir / f"prompt{prompt_idx:02d}.json").write_text(json.dumps(dump_payload, indent=2))
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = old_stdout, old_stderr

    overall_ppl = math.exp(total_neg_log_lik / total_tokens) if total_tokens > 0 else float("inf")
    metrics = {
        "epsilon": epsilon,
        "projection_rank": projection_rank,
        "num_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "total_tokens": total_tokens,
        "total_neg_log_lik": total_neg_log_lik,
        "perplexity": overall_ppl,
        "prompts": prompt_metrics,
        "chunks": chunk_records,
        "layer_lengths": layer_lengths_all,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    generations_path = run_dir / "generations.json"
    generations_path.write_text(json.dumps({"generations": prompt_generations}, indent=2))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + int(os.environ.get("RANK", "0")))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="flash_attention_2",
    )
    model.eval()

    prompts = load_prompts(args.prompts_path, args.num_prompts)

    start_time = time.time()
    for epsilon in args.epsilons:
        run_experiment(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            epsilon=epsilon,
            projection_rank=args.projection_rank,
            max_new_tokens=args.max_new_tokens,
            output_dir=args.output_dir,
            dump_dir=args.dump_dir,
            log_cache_lengths=args.log_cache_lengths,
            log_dir=args.log_dir,
        )
    elapsed = time.time() - start_time
    print(f"[RUN_COMPLETE] total_time={elapsed:.1f}s")


if __name__ == "__main__":
    main()
