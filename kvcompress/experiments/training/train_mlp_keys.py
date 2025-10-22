#!/usr/bin/env python
import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train per-head MLPs for key projection bottlenecks.")
    parser.add_argument("--dump-root", type=Path, default=Path("dumps/deepseek_r1_qkv"))
    parser.add_argument("--stats-path", type=Path, default=Path("analysis/qk_stats.pt"))
    parser.add_argument("--init-path", type=Path, default=Path("analysis/projection_init.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/mlp"))
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--rank", type=int, default=None, help="Run a single rank.")
    parser.add_argument("--rank-map", type=str, default=None, help="Comma separated rank list aligned with LOCAL_RANK.")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=600, help="Max seconds to spend before aborting.")
    parser.add_argument("--micro-batches", type=int, default=8, help="Number of batch files concatenated per step.")
    return parser.parse_args()


def load_tensor(path: Path) -> Dict[str, torch.Tensor]:
    return torch.load(path)


def list_batches(layer_dir: Path) -> List[Path]:
    return sorted(layer_dir.glob("batch_*.pt"))


def load_batch(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return {k: v.to(device=device, dtype=torch.bfloat16, non_blocking=True) for k, v in payload.items()}


def load_micro_batch(paths: List[Path], device: torch.device) -> Dict[str, torch.Tensor]:
    merged: Dict[str, List[torch.Tensor]] = {}
    for path in paths:
        payload = load_batch(path, device)
        for key, tensor in payload.items():
            merged.setdefault(key, []).append(tensor)
    return {key: torch.cat(tensors, dim=0) for key, tensors in merged.items()}


class KeyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LayerMLP(nn.Module):
    def __init__(self, num_heads: int, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [KeyMLP(input_dim, hidden_dim, out_dim) for _ in range(num_heads)]
        )
        self.input_dim = input_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, num_heads, dim = x.shape
        outputs = []
        for idx, head in enumerate(self.heads):
            inp = x[:, :, idx, :].reshape(-1, dim).to(torch.float32)
            out = head(inp).reshape(batch, seq_len, self.out_dim)
            outputs.append(out)
        stacked = torch.stack(outputs, dim=2)
        return stacked.to(dtype=x.dtype)


def project_queries(q: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
    bases = bases.to(device=q.device, dtype=q.dtype)
    projected = torch.einsum("bsnd,hdr->bsnr", q, bases)
    return projected


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    return x.repeat_interleave(groups, dim=2)


def scaled_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    dim = q.shape[-1]
    scale = dim ** -0.5
    logits = torch.matmul(q.to(torch.float32), k.transpose(-1, -2).to(torch.float32)) * scale
    weights = torch.softmax(logits, dim=-1).to(v.dtype)
    return torch.matmul(weights, v)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.to(torch.float32), target.to(torch.float32))


def evaluate(
    mlp: LayerMLP,
    eval_batches: List[Dict[str, torch.Tensor]],
    query_bases: torch.Tensor,
    k_mean: torch.Tensor,
    group_size: int,
) -> float:
    total = 0.0
    with torch.no_grad():
        for payload in eval_batches:
            q = payload["q"]
            k = payload["k"]
            v = payload["v"]
            o = payload["o"]

            k_centered = (k - k_mean)
            k_proj = mlp(k_centered)
            q_proj = project_queries(q, query_bases)
            k_rep = repeat_kv(k_proj, group_size)
            q_rep = q_proj
            v_rep = repeat_kv(v, group_size)
            pred = scaled_attention(q_rep, k_rep, v_rep)
            total += mse_loss(pred, o).item()
    return total / max(1, len(eval_batches))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    stats = load_tensor(args.stats_path)
    k_mean_full = stats["k_mean"].to(device=device, dtype=torch.bfloat16)

    init = load_tensor(args.init_path)
    eigenvectors = init["eigenvectors"].to(device=device, dtype=torch.float32)

    num_layers, num_kv_heads, head_dim, _ = eigenvectors.shape
    num_heads_total = stats["sigma_q"].shape[1]
    group_size = num_heads_total // num_kv_heads

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rank is not None:
        target_ranks = [args.rank]
    elif args.rank_map and local_rank >= 0:
        rank_entries = [int(tok.strip()) for tok in args.rank_map.split(",") if tok.strip()]
        if not rank_entries or local_rank >= len(rank_entries):
            raise ValueError("Rank map does not align with LOCAL_RANK.")
        target_ranks = [rank_entries[local_rank]]
    else:
        target_ranks = args.ranks

    start_time = time.time()

    for rank in target_ranks:
        if args.time_limit and (time.time() - start_time) >= args.time_limit:
            break
        eval_losses = torch.zeros((num_layers, num_kv_heads), dtype=torch.float32)
        layer_states: List[Dict[str, torch.Tensor]] = []

        query_basis = torch.zeros((num_layers, num_heads_total, head_dim, rank), dtype=torch.float32, device=device)
        for layer in range(num_layers):
            for kv in range(num_kv_heads):
                basis = eigenvectors[layer, kv, :, :rank]
                for offset in range(group_size):
                    q_idx = kv * group_size + offset
                    query_basis[layer, q_idx] = basis

        for layer_idx in range(num_layers):
            if args.time_limit and (time.time() - start_time) >= args.time_limit:
                break
            layer_dir = args.dump_root / f"layer_{layer_idx:02d}"
            batch_paths = list_batches(layer_dir)
            train_cutoff = max(1, int(len(batch_paths) * args.train_fraction))
            train_paths = batch_paths[:train_cutoff]
            eval_paths = batch_paths[train_cutoff:] or batch_paths[-args.eval_batches :]
            eval_data = [load_batch(p, device) for p in eval_paths[: args.eval_batches]]

            mlp = LayerMLP(num_kv_heads, head_dim, args.hidden, rank).to(device)
            optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)

            best_loss = float("inf")
            best_state = None
            no_improve = 0
            steps = 0

            k_mean_layer = k_mean_full[layer_idx].unsqueeze(0).unsqueeze(0)
            query_bases_layer = query_basis[layer_idx]

            while steps < args.max_steps:
                if args.time_limit and (time.time() - start_time) >= args.time_limit:
                    break

                start_idx = (steps * args.micro_batches) % len(train_paths)
                micro_paths = [
                    train_paths[(start_idx + offset) % len(train_paths)]
                    for offset in range(args.micro_batches)
                ]
                payload = load_micro_batch(micro_paths, device)

                q = payload["q"]
                k = payload["k"]
                v = payload["v"]
                o = payload["o"]

                k_centered = (k - k_mean_layer)
                k_proj = mlp(k_centered)
                q_proj = project_queries(q, query_bases_layer)
                k_rep = repeat_kv(k_proj, group_size)
                q_rep = q_proj
                v_rep = repeat_kv(v, group_size)

                pred = scaled_attention(q_rep, k_rep, v_rep)
                loss = mse_loss(pred, o)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                steps += 1

                eval_loss = evaluate(mlp, eval_data, query_bases_layer, k_mean_layer, group_size)
                if eval_loss + 1e-6 < best_loss:
                    best_loss = eval_loss
                    best_state = {k: v.detach().cpu() for k, v in mlp.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= 3:
                    break

                del payload, q, k, v, o, k_centered, k_proj, q_proj, k_rep, q_rep, v_rep, pred, loss
                torch.cuda.empty_cache()

            if best_state is None:
                best_state = {k: v.detach().cpu() for k, v in mlp.state_dict().items()}
                best_loss = evaluate(mlp, eval_data, query_bases_layer, k_mean_layer, group_size)

            eval_losses[layer_idx].fill_(best_loss)
            layer_states.append(best_state)

            del eval_data
            torch.cuda.empty_cache()

        if args.time_limit and (time.time() - start_time) >= args.time_limit:
            break

        torch.save(
            {
                "rank": rank,
                "eval_loss": eval_losses,
                "states": layer_states,
                "hidden": args.hidden,
            },
            output_dir / f"mlp_rank_{rank}.pt",
        )


if __name__ == "__main__":
    main()
