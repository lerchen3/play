#!/usr/bin/env python
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from modeling.modeling_llama import LlamaFlashAttention2, LlamaForCausalLM, apply_rotary_pos_emb
from transformers.modeling_flash_attention_utils import _flash_attention_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Dump per-layer Q/K/V/O tensors.")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--dataset", type=str, default="PatrickHaller/fineweb-1B")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--total-batches", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=Path("dumps/qkv_attn"))
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank
    return 0, 1, 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def yield_token_chunks(
    dataset_iter: Iterable[Dict[str, str]],
    tokenizer: AutoTokenizer,
    seq_len: int,
) -> Iterator[List[int]]:
    buffer: List[int] = []
    for sample in dataset_iter:
        text = sample.get("text")
        if not text:
            continue
        input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not input_ids:
            continue
        buffer.extend(input_ids)
        while len(buffer) >= seq_len:
            chunk = buffer[:seq_len]
            buffer = buffer[seq_len:]
            yield chunk


def yield_batches(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    tokenizer: AutoTokenizer,
    seq_len: int,
    batch_size: int,
    streaming: bool,
    world_size: int,
    rank: int,
) -> Iterator[torch.LongTensor]:
    if streaming:
        stream = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        stream = stream.shuffle(buffer_size=10_000, seed=rank + 17)

        def shard_iter() -> Iterator[Dict[str, str]]:
            for global_idx, sample in enumerate(stream):
                if global_idx % world_size == rank:
                    yield sample

        chunk_iter = yield_token_chunks(shard_iter(), tokenizer, seq_len)
    else:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=False)

        def shard_iter() -> Iterator[Dict[str, str]]:
            total = len(dataset)
            for idx in range(rank, total, world_size):
                yield dataset[idx]

        chunk_iter = yield_token_chunks(shard_iter(), tokenizer, seq_len)
    batch: List[List[int]] = []
    for chunk in chunk_iter:
        batch.append(chunk)
        if len(batch) < batch_size:
            continue
        tensor = torch.tensor(batch, dtype=torch.long)
        batch.clear()
        yield tensor


class AttentionRecorder:
    def __init__(self, model: LlamaForCausalLM, dtype: torch.dtype, device: torch.device, out_dir: Path) -> None:
        self.model = model
        self.dtype = dtype
        self.device = device
        self.out_dir = out_dir
        self.storage: Dict[int, Dict[str, torch.Tensor]] = {}
        self._patch()

    def _patch(self) -> None:
        for layer_idx, block in enumerate(self.model.model.layers):
            attn = block.self_attn
            if isinstance(attn, LlamaFlashAttention2):
                self._wrap_flash(attn, layer_idx)
            else:
                raise RuntimeError("Flash attention is required.")

    def _wrap_flash(self, attn: LlamaFlashAttention2, layer_idx: int) -> None:
        original_forward = attn.forward

        def patched(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings=None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)
            value_states = attn.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

            if position_embeddings is None:
                cos, sin = attn.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                query_states, key_states = past_key_value.update(query_states, key_states, attn.layer_idx, cache_kwargs)

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            query_states_cpu = query_states.detach().to(dtype=self.dtype, device="cpu", non_blocking=True)
            key_states_cpu = key_states.detach().to(dtype=self.dtype, device="cpu", non_blocking=True)
            value_states_cpu = value_states.detach().to(dtype=self.dtype, device="cpu", non_blocking=True)

            dropout_rate = attn.attention_dropout if attn.training else 0.0

            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                position_ids=position_ids,
                dropout=dropout_rate,
                sliding_window=getattr(attn, "sliding_window", None),
                use_top_left_mask=attn._flash_attn_uses_top_left_mask,
                is_causal=attn.is_causal,
                **kwargs,
            )

            attn_output_cpu = attn_output.detach().to(dtype=self.dtype, device="cpu", non_blocking=True)
            self.storage[layer_idx] = {
                "q": query_states_cpu,
                "k": key_states_cpu,
                "v": value_states_cpu,
                "o": attn_output_cpu,
            }

            attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
            attn_output = attn.o_proj(attn_output)

            if output_attentions:
                attn_weights = torch.empty(
                    attn_output.shape[0], attn.num_heads, q_len, q_len, device=attn_output.device, dtype=attn_output.dtype
                )
            else:
                attn_weights = None
            return attn_output, attn_weights, past_key_value

        attn.forward = patched  # type: ignore[assignment]

    def save(self, batch_idx: int) -> None:
        for layer_idx, payload in self.storage.items():
            layer_dir = self.out_dir / f"layer_{layer_idx:02d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            path = layer_dir / f"batch_{batch_idx:04d}.pt"
            torch.save(payload, path)
        self.storage.clear()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    set_seed(args.seed + rank)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    model.to(device=device)
    model.eval()

    recorder = AttentionRecorder(model, dtype=dtype, device=device, out_dir=args.output_dir)

    batches_per_rank = args.total_batches // world_size
    remainder = args.total_batches % world_size
    extra = 1 if rank < remainder else 0
    start_idx = rank * batches_per_rank + min(rank, remainder)
    total_local = batches_per_rank + extra

    stream_iter = yield_batches(
        args.dataset,
        args.dataset_config,
        args.split,
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.streaming,
        world_size,
        rank,
    )

    with torch.no_grad():
        for local_idx in range(total_local):
            try:
                input_ids = next(stream_iter)
            except StopIteration:
                break
            attn_mask = torch.ones_like(input_ids)
            input_ids = input_ids.to(device=device, non_blocking=True)
            attn_mask = attn_mask.to(device=device, non_blocking=True)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=False,
                output_hidden_states=False,
            )
            del outputs
            torch.cuda.synchronize(device)
            recorder.save(start_idx + local_idx)

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        meta = {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "total_batches": args.total_batches,
            "dtype": args.dtype,
            "world_size": world_size,
        }
        torch.save(meta, args.output_dir / "metadata.pt")


if __name__ == "__main__":
    main()
