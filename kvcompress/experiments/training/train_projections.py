#!/usr/bin/env python
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Optimize projection matrices for key subspaces.")
    parser.add_argument("--dump-root", type=Path, default=Path("dumps/deepseek_r1_qkv"))
    parser.add_argument("--stats-path", type=Path, default=Path("analysis/qk_stats.pt"))
    parser.add_argument("--init-path", type=Path, default=Path("analysis/projection_init.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/projections"))
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=600, help="Max seconds per rank")
    parser.add_argument("--rank", type=int, default=None, help="Train single rank")
    return parser.parse_args()


def load_stats(stats_path: Path) -> Dict[str, torch.Tensor]:
    stats = torch.load(stats_path)
    return {
        "sigma_q": stats["sigma_q"],
        "cov_k": stats["cov_k"],
        "k_mean": stats["k_mean"],
    }


def load_init(init_path: Path) -> Dict[str, torch.Tensor]:
    bundle = torch.load(init_path)
    return {
        "eigenvectors": bundle["eigenvectors"],
        "var_dir": bundle["var_dir"],
    }


def layer_batch_paths(dump_root: Path, layer_idx: int) -> List[Path]:
    layer_dir = dump_root / f"layer_{layer_idx:02d}"
    return sorted(layer_dir.glob("batch_*.pt"))


def load_batch(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return {
        key: tensor.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        for key, tensor in payload.items()
    }


def repeat_kv(k: torch.Tensor, groups: int) -> torch.Tensor:
    return k.repeat_interleave(groups, dim=2)


def project_keys(
    k_centered: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    basis = basis.to(dtype=k_centered.dtype)
    coeff = torch.einsum("bsnd,ndr->bsnr", k_centered, basis)
    projected = torch.einsum("bsnr,ndr->bsnd", coeff, basis)
    return projected.to(k_centered.dtype)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        causal=True,
        return_attn_probs=False,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.to(torch.float32), target.to(torch.float32))


def reorthonormalize(basis: torch.Tensor) -> None:
    with torch.no_grad():
        for idx in range(basis.shape[0]):
            q, _ = torch.linalg.qr(basis[idx])
            basis[idx].copy_(q)


def evaluate(
    basis: torch.Tensor,
    eval_batches: List[Dict[str, torch.Tensor]],
    k_mean: torch.Tensor,
    group_size: int,
) -> float:
    device = basis.device
    total_loss = 0.0
    with torch.no_grad():
        for payload in eval_batches:
            q = payload["q"]
            k = payload["k"]
            v = payload["v"]
            o = payload["o"]

            k_centered = (k - k_mean)
            projected = project_keys(k_centered, basis)
            k_new = projected
            k_rep = repeat_kv(k_new, group_size)
            v_rep = repeat_kv(v, group_size)

            pred = flash_attention(q, k_rep, v_rep)
            loss = mse_loss(pred, o)
            total_loss += loss.item()
    return total_loss / max(1, len(eval_batches))


def main() -> None:
    import time
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    stats = load_stats(args.stats_path)
    k_mean = stats["k_mean"].to(device=device, dtype=torch.bfloat16)

    init_bundle = load_init(args.init_path)
    eigenvectors = init_bundle["eigenvectors"].to(device=device, dtype=torch.bfloat16)

    num_layers, num_kv_heads, head_dim, _ = eigenvectors.shape
    total_heads = stats["sigma_q"].shape[1]
    group_size = total_heads // num_kv_heads

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ranks_to_train = [args.rank] if args.rank is not None else args.ranks
    start_time = time.time()

    for rank in ranks_to_train:
        if time.time() - start_time >= args.time_limit:
            print(f"Time limit {args.time_limit}s reached, stopping")
            break
        basis_store = torch.zeros((num_layers, num_kv_heads, head_dim, rank), dtype=torch.float32)
        eval_store = torch.zeros((num_layers, num_kv_heads), dtype=torch.float32)

        for layer_idx in range(num_layers):
            batch_paths = layer_batch_paths(args.dump_root, layer_idx)
            train_cutoff = int(len(batch_paths) * args.train_fraction)
            train_paths = batch_paths[:train_cutoff]
            eval_paths = batch_paths[train_cutoff:]
            if not eval_paths:
                eval_paths = batch_paths[-args.eval_batches :]

            eval_data = []
            for path in eval_paths[: args.eval_batches]:
                payload = load_batch(path, device)
                eval_data.append(payload)

            basis_init = eigenvectors[layer_idx, :, :, :rank].to(torch.float32)
            basis_param = nn.Parameter(basis_init.clone().to(device))
            optimizer = torch.optim.Adam([basis_param], lr=args.lr)

            best_loss = float("inf")
            best_state = None
            no_improve = 0
            steps = 0

            while steps < args.max_steps:
                train_path = train_paths[steps % len(train_paths)]
                payload = load_batch(train_path, device)

                q = payload["q"]
                k = payload["k"]
                v = payload["v"]
                o = payload["o"]

                k_centered = (k - k_mean[layer_idx].unsqueeze(0).unsqueeze(0))
                projected = project_keys(k_centered, basis_param)
                k_new = projected

                k_rep = repeat_kv(k_new, group_size)
                v_rep = repeat_kv(v, group_size)

                pred = flash_attention(q, k_rep, v_rep)
                loss = mse_loss(pred, o)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                reorthonormalize(basis_param)

                steps += 1

                eval_loss = evaluate(basis_param, eval_data, k_mean[layer_idx].unsqueeze(0).unsqueeze(0), group_size)
                if eval_loss + 1e-6 < best_loss:
                    best_loss = eval_loss
                    best_state = basis_param.detach().cpu().to(torch.float32)
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= 3:
                    break

                del payload, q, k, v, o, pred, loss, k_centered, projected, k_new, k_rep, v_rep
                torch.cuda.empty_cache()

            if best_state is None:
                best_state = basis_param.detach().cpu().to(torch.float32)
                best_loss = evaluate(basis_param, eval_data, k_mean[layer_idx].unsqueeze(0).unsqueeze(0), group_size)

            basis_store[layer_idx] = best_state
            eval_store[layer_idx] = best_loss

            del eval_data
            torch.cuda.empty_cache()

        torch.save(
            {
                "basis": basis_store,
                "eval_loss": eval_store,
                "rank": rank,
            },
            output_dir / f"projection_rank_{rank}.pt",
        )


if __name__ == "__main__":
    main()
