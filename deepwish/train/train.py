from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Ensure project root is importable when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml

from train.utils import ensure_dir, ensure_parent_dir, flatten_config, load_tokenizer, set_seed

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None

DEFAULT_TOKENIZER_PATH = "Qwen/Qwen3-0.6B"
SUPPORTED_PRECISIONS: Dict[str, Optional[torch.dtype]] = {
    "fp32": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def get_model_class(architecture: str):
    """Import and return the appropriate model class based on architecture."""
    if architecture == "qwen3next":
        from model.qwen3next.training import Qwen3NextTrainModel

        return Qwen3NextTrainModel
    if architecture == "qwen3":
        from model.qwen3.training import Qwen3TrainModel

        return Qwen3TrainModel
    if architecture == "deepseekv3":
        from model.deepseekv3 import DeepSeekV3Model

        return DeepSeekV3Model
    raise ValueError(f"Unknown architecture: {architecture}")


class ChatDataset(Dataset[Any]):
    """Dataset that tokenizes user/assistant turns on the fly."""

    def __init__(self, users, assistants, tokenizer, seq_len: int, mtp_depth: int = 0):
        assert len(users) == len(assistants)
        self.users = users
        self.assistants = assistants
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mtp_depth = mtp_depth
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.use_chat_template = callable(getattr(tokenizer, "apply_chat_template", None))

    def __len__(self) -> int:
        return len(self.users)

    def _encode_messages(self, messages):
        if self.use_chat_template:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.seq_len,
            )
        else:
            # Fallback: simple conversation formatting
            conversation = "\n\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in messages
            )
            encoded = self.tokenizer(
                conversation,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.seq_len,
            )
        if isinstance(encoded, dict):
            input_ids = encoded.get("input_ids")
        else:
            input_ids = getattr(encoded, "input_ids", encoded)
        if isinstance(input_ids, torch.Tensor):
            tensor = input_ids
        else:
            tensor = torch.tensor(input_ids, dtype=torch.long)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.squeeze(0)

    def __getitem__(self, idx: int):
        user = "" if pd.isna(self.users[idx]) else str(self.users[idx])
        assistant = "" if pd.isna(self.assistants[idx]) else str(self.assistants[idx])

        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]

        input_ids = self._encode_messages(messages)

        if self.mtp_depth > 0:
            tgt_matrix = torch.full((self.seq_len, self.mtp_depth), self.pad_id, dtype=torch.long)
            for i in range(self.seq_len - 1):
                max_j = min(self.mtp_depth, self.seq_len - i - 1)
                if max_j <= 0:
                    break
                tgt_matrix[i, :max_j] = input_ids[i + 1 : i + 1 + max_j]
            return input_ids, tgt_matrix
        return input_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train language models with DeepSeek-V3 or Qwen3 architectures.")
    parser.add_argument('--config', type=str, default=None, help='Path to a YAML config file (CLI keys).')

    # Core selection
    parser.add_argument('--architecture', type=str, choices=['qwen3', 'qwen3next', 'deepseekv3'], default=None,
                        help='Model architecture to use.')
    parser.add_argument('--tokenizer_path', type=str, default=DEFAULT_TOKENIZER_PATH,
                        help='Tokenizer identifier or local path.')

    # Data arguments
    parser.add_argument('--data_path', type=str, default=None, help='Path to CSV with chat data.')
    parser.add_argument('--user_column', type=str, default='user', help='User column name in CSV.')
    parser.add_argument('--assistant_column', type=str, default='assistant', help='Assistant column name in CSV.')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split fraction.')

    # Training arguments
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per device.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='AdamW weight decay.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--deterministic', action='store_true', help='Force deterministic CuDNN behaviour.')
    parser.add_argument('--precision', type=str, choices=list(SUPPORTED_PRECISIONS.keys()), default='fp32',
                        help='Training precision for autocast.')

    # Model arguments
    parser.add_argument('--d_model', type=int, help='Model hidden dimension (default varies by arch).')
    parser.add_argument('--num_layers', type=int, help='Number of transformer layers.')
    parser.add_argument('--rmsnorm_eps', type=float, default=1e-6, help='RMSNorm epsilon.')

    # Qwen3-specific
    parser.add_argument('--n_q_heads', type=int, help='Number of query heads (Qwen3).')
    parser.add_argument('--n_kv_heads', type=int, help='Number of key-value heads (Qwen3).')
    parser.add_argument('--d_ff', type=int, help='Feed-forward dimension (Qwen3).')
    parser.add_argument('--window_size', type=int, help='Sliding-window size for NSA/Qwen3.')
    parser.add_argument('--nsa_cmp_blk_size', type=int, help='NSA compressed block size (Qwen3).')
    parser.add_argument('--nsa_cmp_stride', type=int, help='NSA compressed block stride (Qwen3).')
    parser.add_argument('--nsa_slc_top_n', type=int, help='NSA top-N selected blocks per group (Qwen3).')

    # DeepSeek-V3 specific
    parser.add_argument('--n_heads', type=int, help='Number of attention heads (DeepSeek-V3).')
    parser.add_argument('--dc_kv', type=int, help='Key-value compression dim (DeepSeek-V3 MLA).')
    parser.add_argument('--dc_q', type=int, help='Query compression dim (DeepSeek-V3 MLA).')
    parser.add_argument('--d_head', type=int, help='Attention head dimension override.')
    parser.add_argument('--n_shared_experts', type=int, help='Number of shared experts (DeepSeek-V3 MoE).')
    parser.add_argument('--n_routed_experts', type=int, help='Number of routed experts (DeepSeek-V3 MoE).')
    parser.add_argument('--k_routed_experts', type=int, help='Number of active routed experts (DeepSeek-V3 MoE).')
    parser.add_argument('--d_ff_expert_mult', type=int, help='Expert FFN dimension multiplier (DeepSeek-V3).')
    parser.add_argument('--moe_balance_factor', type=float, help='MoE load balancing factor.')
    parser.add_argument('--bias_update_speed', type=float, help='MoE bias update speed.')
    parser.add_argument('--mtp_depth', type=int, default=0, help='Multi-token prediction depth (DeepSeek-V3).')
    parser.add_argument('--mtp_weight', type=float, default=0.5, help='Weight for MTP auxiliary loss.')

    # Optimiser arguments
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1.')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam beta2.')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='Adam epsilon.')

    # Learning rate schedule
    parser.add_argument('--lr_warmup_steps', type=int, default=1000, help='LR warmup steps.')
    parser.add_argument('--lr_decay_steps', type=int, default=10000, help='LR decay steps.')

    # Checkpointing and evaluation
    parser.add_argument('--model_checkpoint', type=str, default=None, help='Model checkpoint path for loading weights.')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Full training checkpoint path.')
    parser.add_argument('--checkpoint_save_path', type=str, default=None, help='Path to save periodic checkpoints.')
    parser.add_argument('--model_save_path', type=str, default=None, help='Path to save final model weights.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for checkpoints and metrics.')
    parser.add_argument('--metrics_path', type=str, default=None, help='CSV file for logging metrics.')
    parser.add_argument('--log_dir', type=str, default=None, help='TensorBoard log directory.')
    parser.add_argument('--log_every', type=int, default=10, help='Steps between console logs.')
    parser.add_argument('--save_every', type=int, default=100, help='Steps between checkpoint saves.')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval in steps.')
    parser.add_argument('--tensorboard_flush_secs', type=int, default=30, help='TensorBoard flush seconds.')

    # Dataloader
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers.')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Prefetch factor for DataLoader.')
    parser.add_argument('--pin_memory', action='store_true', help='Pin DataLoader memory.')
    parser.add_argument('--persistent_workers', action='store_true', help='Use persistent workers in DataLoader.')
    parser.add_argument('--limit_train_batches', type=int, default=None, help='Optional cap on train batches per epoch.')
    parser.add_argument('--limit_eval_batches', type=int, default=None, help='Optional cap on eval batches.')

    # Misc
    parser.add_argument('--time_limit', type=int, default=11 * 3600 + 30 * 60, help='Training time limit in seconds.')
    parser.add_argument('--ema_decay', type=float, default=0.9, help='EMA decay for loss tracking.')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as handle:
            config_data = yaml.safe_load(handle) or {}
        if not isinstance(config_data, dict):
            raise ValueError('Config file must contain a mapping of CLI keys to values.')
        flat_config = flatten_config(config_data)
        for key, value in flat_config.items():
            if not hasattr(args, key):
                print(f"[parse_args] Warning: ignoring unknown config key '{key}'")
                continue
            default_value = parser.get_default(key)
            current_value = getattr(args, key)
            if current_value == default_value:
                setattr(args, key, value)

    if args.architecture is None:
        parser.error('`--architecture` must be supplied either via CLI or config file.')

    if args.architecture == "qwen3next":
        args.d_model = args.d_model or 1024
        args.num_layers = args.num_layers or 28
        args.n_q_heads = args.n_q_heads or 16
        args.n_kv_heads = args.n_kv_heads or 8
        args.d_ff = args.d_ff or 3072
    elif args.architecture == "deepseekv3":
        args.d_model = args.d_model or 512
        args.num_layers = args.num_layers or 6
        args.n_heads = args.n_heads or 8
        args.dc_kv = args.dc_kv or 32
        args.dc_q = args.dc_q or 32
        args.n_shared_experts = args.n_shared_experts or 1
        args.n_routed_experts = args.n_routed_experts or 4
        args.k_routed_experts = args.k_routed_experts or 1
        args.d_ff_expert_mult = args.d_ff_expert_mult or 2
        args.moe_balance_factor = args.moe_balance_factor or 0.01
        args.bias_update_speed = args.bias_update_speed or 0.001
        args.d_ff_expert = args.d_model * args.d_ff_expert_mult
        if not hasattr(args, 'use_nsa'):
            args.use_nsa = False
        if not hasattr(args, 'num_kv_heads'):
            args.num_kv_heads = args.n_heads
        if not hasattr(args, 'window_size') or args.window_size <= 0:
            args.window_size = None
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")

    if args.metrics_path is None:
        args.metrics_path = os.path.join(args.checkpoint_dir, 'metrics.csv')

    if args.num_workers <= 0:
        args.prefetch_factor = None
        args.persistent_workers = False

    return args


def evaluate(model, loader, device, args, limit_batches=None, autocast_ctx=None) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    if loader is None:
        return float('nan')

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            if args.architecture in {"deepseekv3", "qwen3next"} and args.mtp_depth > 0 and isinstance(batch, (tuple, list)):
                input_ids, tgt_matrix = batch
                input_ids = input_ids.to(device)
                tgt_matrix = tgt_matrix.to(device)
            else:
                input_ids = batch[0] if isinstance(batch, (tuple, list)) else batch
                input_ids = input_ids.to(device)
                tgt_matrix = None

            real_model = model.module if hasattr(model, 'module') else model
            target_main = torch.roll(input_ids, shifts=-1, dims=1)
            target_main[:, -1] = args.pad_token_id

            ctx = autocast_ctx() if autocast_ctx is not None else nullcontext()
            with ctx:
                outputs = real_model(input_ids, target_main=target_main, tgt_matrix=tgt_matrix, is_training=False)
            if isinstance(outputs, torch.Tensor) and outputs.numel() >= 2:
                loss_main, loss_mtp = outputs[0], outputs[1]
                loss = loss_main + args.mtp_weight * loss_mtp
            else:
                loss = outputs if isinstance(outputs, torch.Tensor) else torch.tensor(outputs, device=device)

            total_loss += loss.item()
            count += 1

    model.train()
    return total_loss / max(1, count)


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    print(f"[RANK {rank}] Starting training on device: {device}", flush=True)

    try:
        args = parse_args()
        set_seed(args.seed, args.deterministic)

        if args.precision in ('fp16', 'bf16') and device.type != 'cuda':
            print('[train] Requested mixed precision but CUDA is unavailable; falling back to fp32.')
            args.precision = 'fp32'
        autocast_dtype = SUPPORTED_PRECISIONS[args.precision]
        amp_enabled = device.type == 'cuda' and autocast_dtype is not None

        def autocast_ctx():
            return torch.cuda.amp.autocast(dtype=autocast_dtype) if amp_enabled else nullcontext()

        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and args.precision == 'fp16')

        tokenizer = load_tokenizer(args.tokenizer_path)
        if tokenizer.pad_token_id is None:
            raise ValueError('Tokenizer must define a pad token.')

        args.vocab_size = len(tokenizer)
        args.pad_token_id = tokenizer.pad_token_id
        args.device = device

        def print_rank0(*msg):
            if rank == 0:
                print(*msg)

        print_rank0(f"Using tokenizer: {args.tokenizer_path} (vocab size={args.vocab_size})")

        if args.data_path and not os.path.isfile(args.data_path):
            raise FileNotFoundError(f"Data file not found at {args.data_path}")

        if args.data_path:
            df = pd.read_csv(args.data_path)
            if args.user_column not in df.columns or args.assistant_column not in df.columns:
                raise KeyError(f"Dataset must contain columns '{args.user_column}' and '{args.assistant_column}'")
            users = df[args.user_column].tolist()
            assistants = df[args.assistant_column].tolist()
        else:
            print_rank0('No data path provided; using synthetic fallback data for smoke testing.')
            users = ["Hello", "How are you?", "What's the weather like?", "Tell me a joke!"] * 32
            assistants = [
                "Hi there!",
                "Doing great, thanks!",
                "It is sunny today.",
                "Why did the tensor go to therapy? It had too many unresolved gradients!",
            ] * 32

        split_idx = int(len(users) * (1.0 - args.val_split)) if len(users) else 0
        train_users, val_users = users[:split_idx], users[split_idx:]
        train_assistants, val_assistants = assistants[:split_idx], assistants[split_idx:]

        mtp_depth = args.mtp_depth if args.architecture in {"deepseekv3", "qwen3next"} else 0
        train_dataset = ChatDataset(train_users, train_assistants, tokenizer, args.seq_len, mtp_depth)
        val_dataset = ChatDataset(val_users, val_assistants, tokenizer, args.seq_len, mtp_depth) if val_users else None

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if val_dataset and world_size > 1 else None

        loader_kwargs: Dict[str, Any] = {
            'batch_size': args.batch_size,
            'drop_last': True,
            'num_workers': args.num_workers,
            'pin_memory': args.pin_memory,
        }
        if args.num_workers > 0 and args.prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = args.prefetch_factor
        if args.num_workers > 0:
            loader_kwargs['persistent_workers'] = args.persistent_workers

        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            **loader_kwargs,
        )

        val_loader = None
        if val_dataset is not None:
            val_kwargs = dict(loader_kwargs)
            val_kwargs['drop_last'] = False
            val_loader = DataLoader(
                val_dataset,
                sampler=val_sampler,
                shuffle=False,
                **val_kwargs,
            )

        print_rank0(f"Dataset split: {len(train_dataset)} train / {len(val_dataset) if val_dataset else 0} val")

        ModelClass = get_model_class(args.architecture)
        model = ModelClass(args).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print_rank0(f"Model created with {total_params:,} parameters ({total_params / 1e6:.2f}M)")

        resume_state = None
        if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
            resume_state = torch.load(args.resume_checkpoint, map_location='cpu')
            state_dict = resume_state.get('model_state_dict')
            if state_dict:
                model.load_state_dict(state_dict, strict=False)
                print_rank0(f"Resumed model weights from {args.resume_checkpoint}")
        elif args.model_checkpoint and os.path.isfile(args.model_checkpoint):
            ckpt = torch.load(args.model_checkpoint, map_location='cpu')
            state_dict = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state_dict, strict=False)
            print_rank0(f"Loaded model weights from {args.model_checkpoint}")

        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                find_unused_parameters=True,
            )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )

        def lr_lambda(step: int) -> float:
            if step < args.lr_warmup_steps:
                return float(step) / float(max(1, args.lr_warmup_steps))
            decay_steps = max(1, args.lr_decay_steps - args.lr_warmup_steps)
            return max(0.0, float(args.lr_decay_steps - step) / decay_steps)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        global_step = 0
        start_epoch = 0
        ema_loss = None
        cum_loss = 0.0
        metrics = []

        if resume_state:
            if 'optimizer_state_dict' in resume_state:
                optimizer.load_state_dict(resume_state['optimizer_state_dict'])
            if 'scheduler_state_dict' in resume_state:
                scheduler.load_state_dict(resume_state['scheduler_state_dict'])
            global_step = resume_state.get('global_step', 0)
            start_epoch = resume_state.get('epoch', 0)
            ema_loss = resume_state.get('ema_loss', None)
            cum_loss = resume_state.get('cum_loss', 0.0)
            metrics = resume_state.get('metrics', []) if resume_state.get('metrics') else []
            print_rank0(f"Resumed training state from step {global_step}, epoch {start_epoch}")

        ensure_dir(args.checkpoint_dir)
        ensure_parent_dir(args.metrics_path)
        ensure_parent_dir(args.checkpoint_save_path)
        ensure_parent_dir(args.model_save_path)
        writer = None
        if args.log_dir:
            if SummaryWriter is None:
                print_rank0('TensorBoard not available; install tensorboard to enable logging.')
            else:
                ensure_dir(args.log_dir)
                writer = SummaryWriter(log_dir=args.log_dir, flush_secs=args.tensorboard_flush_secs)
                writer.add_text('config/summary', str(vars(args)))

        start_time = time.time()
        print_rank0('Starting training...')

        for epoch in range(start_epoch, args.epochs):
            print_rank0(f"=== Epoch {epoch + 1}/{args.epochs} ===")
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            updates_this_epoch = 0

            for batch_idx, batch in enumerate(train_loader):
                if args.limit_train_batches is not None and batch_idx >= args.limit_train_batches:
                    break

                if args.time_limit and (time.time() - start_time) > args.time_limit:
                    print_rank0('Time limit reached; saving checkpoint and exiting.')
                    if rank == 0 and args.checkpoint_save_path:
                        real_model = model.module if hasattr(model, 'module') else model
                        ckpt = {
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': real_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'ema_loss': ema_loss,
                            'cum_loss': cum_loss,
                            'metrics': metrics,
                        }
                        torch.save(ckpt, args.checkpoint_save_path)
                    return

                if args.architecture in {"deepseekv3", "qwen3next"} and args.mtp_depth > 0 and isinstance(batch, (tuple, list)):
                    input_ids, tgt_matrix = batch
                    input_ids = input_ids.to(device)
                    tgt_matrix = tgt_matrix.to(device)
                else:
                    input_ids = batch[0] if isinstance(batch, (tuple, list)) else batch
                    input_ids = input_ids.to(device)
                    tgt_matrix = None

                real_model = model.module if hasattr(model, 'module') else model
                target_main = torch.roll(input_ids, shifts=-1, dims=1)
                target_main[:, -1] = args.pad_token_id

                ctx = autocast_ctx()
                with ctx:
                    outputs = real_model(input_ids, target_main=target_main, tgt_matrix=tgt_matrix, is_training=True)
                    if isinstance(outputs, torch.Tensor) and outputs.numel() >= 2:
                        loss_main, loss_mtp = outputs[0], outputs[1]
                        loss_tensor = loss_main + args.mtp_weight * loss_mtp
                    else:
                        loss_tensor = outputs if isinstance(outputs, torch.Tensor) else torch.tensor(outputs, device=device)

                loss = loss_tensor / args.grad_accum_steps
                loss_value = loss.detach().float().item()

                if grad_scaler.is_enabled():
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % args.grad_accum_steps == 0:
                    if grad_scaler.is_enabled():
                        grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if grad_scaler.is_enabled():
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    updates_this_epoch += 1

                    step_loss = loss_value * args.grad_accum_steps
                    epoch_loss += step_loss
                    cum_loss += step_loss

                    if ema_loss is None:
                        ema_loss = step_loss
                    else:
                        ema_loss = args.ema_decay * ema_loss + (1 - args.ema_decay) * step_loss

                    if rank == 0:
                        metrics_entry = {
                            'step': global_step,
                            'epoch': epoch + 1,
                            'loss': step_loss,
                            'ema_loss': ema_loss,
                            'lr': scheduler.get_last_lr()[0],
                        }
                        metrics.append(metrics_entry)
                        if writer:
                            writer.add_scalar('train/loss', step_loss, global_step)
                            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

                    if global_step % args.log_every == 0:
                        avg_loss = cum_loss / max(1, global_step)
                        print_rank0(f"Epoch {epoch + 1} Step {global_step}: loss={step_loss:.4f}, EMA={ema_loss:.4f}, avg={avg_loss:.4f}")

                    if args.eval_interval and global_step % args.eval_interval == 0 and val_loader is not None:
                        eval_loss = evaluate(model, val_loader, device, args, limit_batches=args.limit_eval_batches,
                                             autocast_ctx=autocast_ctx)
                        print_rank0(f"Validation loss at step {global_step}: {eval_loss:.4f}")
                        if rank == 0:
                            metrics[-1]['eval_loss'] = eval_loss
                            if writer:
                                writer.add_scalar('val/loss', eval_loss, global_step)

                    if args.save_every and global_step % args.save_every == 0 and rank == 0 and args.checkpoint_save_path:
                        real_model = model.module if hasattr(model, 'module') else model
                        ckpt = {
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': real_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'ema_loss': ema_loss,
                            'cum_loss': cum_loss,
                            'metrics': metrics,
                        }
                        torch.save(ckpt, args.checkpoint_save_path)
                        if args.metrics_path and metrics:
                            pd.DataFrame(metrics).to_csv(args.metrics_path, index=False)

            if updates_this_epoch > 0:
                avg_epoch_loss = epoch_loss / updates_this_epoch
                print_rank0(f"End of epoch {epoch + 1}: average loss {avg_epoch_loss:.4f}")

        if rank == 0 and args.model_save_path:
            real_model = model.module if hasattr(model, 'module') else model
            ckpt = {'model_state_dict': real_model.state_dict()}
            torch.save(ckpt, args.model_save_path)
            print_rank0(f"Saved final model to {args.model_save_path}")
            if args.metrics_path and metrics:
                pd.DataFrame(metrics).to_csv(args.metrics_path, index=False)

        if writer:
            writer.flush()
            writer.close()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
