"""Lightweight inference helper for the Qwen3-Next reference implementation."""

from __future__ import annotations

import argparse
import os

import torch
import yaml

from model.qwen3next import Qwen3NextConfig, Qwen3NextForCausalLM
from train.utils import flatten_config, load_tokenizer

DEFAULT_TOKENIZER_PATH = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run greedy/sampled decoding with Qwen3-Next.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to weights saved by train.py")
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--prompt", type=str, default="", help="Prompt text to feed the model")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config overriding CLI defaults")
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--n_q_heads", type=int, default=16)
    parser.add_argument("--n_kv_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_experts", type=int, default=0)
    parser.add_argument("--num_experts_per_tok", type=int, default=1)
    parser.add_argument("--decoder_sparse_step", type=int, default=1)

    args = parser.parse_args()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config_data = yaml.safe_load(handle) or {}
        if not isinstance(config_data, dict):
            raise ValueError("Configuration file must contain a mapping of keys to values.")
        overrides = flatten_config(config_data)
        for key, value in overrides.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"[qwen3next_inference] Ignoring unknown config key '{key}'")
    return args


def _build_config(args: argparse.Namespace, tokenizer) -> Qwen3NextConfig:
    return Qwen3NextConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.d_model,
        intermediate_size=args.d_ff,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.n_q_heads,
        num_key_value_heads=args.n_kv_heads,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        decoder_sparse_step=args.decoder_sparse_step,
    )


@torch.no_grad()
def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    if top_k > 0:
        values, indices = torch.topk(probs, top_k)
        probs = torch.zeros_like(probs).scatter_(0, indices, values)
        probs /= probs.sum()
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        mask = cumulative <= top_p
        mask[..., 0] = True
        filtered = sorted_probs * mask
        filtered /= filtered.sum()
        choice = torch.multinomial(filtered, 1)
        return sorted_indices[choice].item()
    return torch.multinomial(probs, 1).item()


@torch.no_grad()
def generate(
    model: Qwen3NextForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else encoded.input_ids
    input_ids = input_ids.to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    generated = input_ids
    for _ in range(max_new_tokens):
        attention_mask = (generated != pad_id).long()
        outputs = model(input_ids=generated, attention_mask=attention_mask)
        next_token_logits = outputs.logits[:, -1, :].squeeze(0)
        next_token = _sample_next_token(next_token_logits, temperature, top_k, top_p)
        next_token_tensor = torch.tensor([[next_token]], device=device, dtype=generated.dtype)
        generated = torch.cat([generated, next_token_tensor], dim=1)
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated.squeeze(0), skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.tokenizer_path)
    config = _build_config(args, tokenizer)
    model = Qwen3NextForCausalLM(config).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Weights not found at {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[qwen3next_inference] Missing keys: {missing[:8]} ...")
    if unexpected:
        print(f"[qwen3next_inference] Unexpected keys: {unexpected[:8]} ...")
    model.eval()

    output = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        max(args.temperature, 1e-5),
        args.top_k,
        args.top_p,
        device,
    )
    print(output)


if __name__ == "__main__":
    main()
