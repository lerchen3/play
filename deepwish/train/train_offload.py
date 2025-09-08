import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import math
import time
import sys
import torch.distributed as dist
import os

# Add project root to Python path for all imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer
from train.model import DeepSeekV3Model
from train.offload_step import run_offload_deepseek_step

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


class ChatDataset(Dataset):
    """Chat dataset that tokenizes user/assistant pairs via chat template on the fly."""
    def __init__(self, users, assistants, tokenizer, seq_len, mtp_depth=0):
        assert len(users) == len(assistants)
        self.users = users
        self.assistants = assistants
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mtp_depth = mtp_depth
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = "" if pd.isna(self.users[idx]) else str(self.users[idx])
        assistant = "" if pd.isna(self.assistants[idx]) else str(self.assistants[idx])

        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]

        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.seq_len,
        )
        if isinstance(encoded, dict) or hasattr(encoded, 'input_ids'):
            input_ids = (encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids).squeeze(0)
        else:
            input_ids = encoded.squeeze(0)

        tgt_matrix = torch.zeros(self.seq_len, self.mtp_depth, dtype=torch.long)
        for i in range(self.seq_len - 1):
            for j in range(self.mtp_depth):
                tgt_matrix[i, j] = input_ids[i + j + 1] if (i + j + 1) < self.seq_len else self.pad_id
        return input_ids, tgt_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DeepSeek-V3-like model with offloading.")
    parser.add_argument('--data_path', type=str, default=None, help='Path to CSV with chat data')
    parser.add_argument('--user_column', type=str, default='user', help='User column name in CSV')
    parser.add_argument('--assistant_column', type=str, default='assistant', help='Assistant column name in CSV')
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--num_kv_heads', type=int, default=8, help='KV heads for GQA/NSA; must divide n_heads')
    parser.add_argument('--use_nsa', action='store_true', help='Enable NSA (uses GQA)')
    parser.add_argument('--window_size', type=int, default=0, help='Sliding-window size for attention kernels')
    parser.add_argument('--dc_kv', type=int, default=32)
    parser.add_argument('--dc_q', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--rmsnorm_eps', type=float, default=1e-6)
    parser.add_argument('--n_shared_experts', type=int, default=1)
    parser.add_argument('--n_routed_experts', type=int, default=4)
    parser.add_argument('--k_routed_experts', type=int, default=1)
    parser.add_argument('--d_ff_expert_mult', type=int, default=2)
    parser.add_argument('--moe_balance_factor', type=float, default=0.01)
    parser.add_argument('--bias_update_speed', type=float, default=0.001)
    parser.add_argument('--mtp_depth', type=int, default=0, help='Number of future tokens per position')
    parser.add_argument('--mtp_weight', type=float, default=0.5, help='Weight for MTP loss')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                        help='Path to model checkpoint for resuming training (model weights only)')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to full training checkpoint for resuming (includes optimizer, scheduler, metrics)')
    parser.add_argument('--checkpoint_save_path', type=str, default=None,
                        help='Path to save training checkpoint (model + optimizer + epoch)')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='Path to save final model (for inference.py)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of dataset for validation')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Run evaluation every N training steps')
    parser.add_argument('--d_head', type=int, default=None,
                        help='Override the attention head dimension (default d_model/n_heads)')
    parser.add_argument('--time_limit', type=int, default=11*3600+30*60,
                        help='Hard training time limit in seconds before checkpoint+exit')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW')
    parser.add_argument('--lr_warmup_steps', type=int, default=1000,
                        help='Steps to linearly warm up the learning rate')
    parser.add_argument('--lr_decay_steps', type=int, default=10000,
                        help='Total steps for linear LR decay after warmup')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save step checkpoints/metrics')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--ema_decay', type=float, default=0.9, help='Decay factor for exponential moving average of losses')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='Adam epsilon')
    args = parser.parse_args()
    args.d_ff_expert = args.d_model * args.d_ff_expert_mult
    return args


def evaluate(model, loader, device, args):
    model.eval()
    total = 0.0
    count = 0
    if loader is None:
        return float('nan')
    with torch.no_grad():
        for inp, tgt_matrix in loader:
            inp = inp.to(device)
            tgt_matrix = tgt_matrix.to(device)
            # unwrap DDP to access the real model
            real_model = model.module if hasattr(model, 'module') else model
            # prepare next-token targets
            target_main = torch.roll(inp, shifts=-1, dims=1)
            target_main[:, -1] = args.pad_token_id
            # forward with loss computation
            losses = real_model(
                inp,
                target_main=target_main,
                tgt_matrix=tgt_matrix,
                is_training=False
            )
            loss_main, loss_mtp = losses[0], losses[1]
            # ensure tensors
            if loss_main is None:
                loss_main = torch.tensor(0.0, device=device)
            if loss_mtp is None:
                loss_mtp = torch.tensor(0.0, device=device)
            loss = loss_main + args.mtp_weight * loss_mtp
            total += loss.item()
            count += 1
    model.train()
    avg_loss = total / max(1, count)
    return avg_loss


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU mode
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    # Initialize distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    print(f"[RANK {rank}] Starting main function...", flush=True)
    
    try:
        args = parse_args()
        print(f"[RANK {rank}] Args parsed successfully", flush=True)
        
        # Initialize tokenizer (local path)
        print(f"[RANK {rank}] Loading tokenizer from local Qwen path...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/qwen-3/transformers/0.6b/1")

        # Ensure pad token exists
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Use total length, not vocab_size
        args.vocab_size = len(tokenizer)
        args.pad_token_id = tokenizer.pad_token_id
        args.device = device
        # NSA/GQA
        if not hasattr(args, 'use_nsa'):
            args.use_nsa = False
        if not hasattr(args, 'num_kv_heads'):
            args.num_kv_heads = args.n_heads
        if not hasattr(args, 'window_size'):
            args.window_size = None

        # Only print from rank 0
        def print_rank0(*msg):
            if rank == 0:
                print(*msg)

        print_rank0(f"Starting training with {world_size} GPUs")
        print_rank0(f"Data path: {args.data_path}")
        print_rank0(f"Vocab size: {args.vocab_size}")
        print_rank0(f"Pad token ID: {args.pad_token_id}")

        # Load data
        print_rank0("Loading dataset...")
        df = pd.read_csv(args.data_path)
        users = df[args.user_column].tolist() if args.user_column in df.columns else [""] * len(df)
        assistants = df[args.assistant_column].tolist() if args.assistant_column in df.columns else [""] * len(df)
        # Split data
        split_idx = int(len(users) * (1.0 - args.val_split))
        train_users, val_users = users[:split_idx], users[split_idx:]
        train_assistants, val_assistants = assistants[:split_idx], assistants[split_idx:]
        print_rank0(f"Dataset split: {len(train_users)} train, {len(val_users)} val")

        # Create datasets
        train_dataset = ChatDataset(train_users, train_assistants, tokenizer, args.seq_len, args.mtp_depth)
        val_dataset = ChatDataset(val_users, val_assistants, tokenizer, args.seq_len, args.mtp_depth) if val_users else None

        # Create DataLoaders
        print_rank0("Creating data loaders...")
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False) if val_dataset else None
        print_rank0("Data loaders created")

        # Initialize training state variables
        start_epoch = 0
        start_global_step = 1
        resume_ema_loss = None
        resume_ema_loss_main = None 
        resume_ema_loss_mtp = None
        resume_ema_bal_loss = None
        resume_ema_param_norm = None
        resume_ema_grad_norm = None
        resume_ema_tokens_per_sec = None
        resume_ema_eval_loss = None
        resume_cum_loss = 0.0
        resume_cum_loss_main = 0.0
        resume_cum_loss_mtp = 0.0

        os.makedirs(args.checkpoint_dir, exist_ok=True)

        # Only keep one checkpoint and one metrics file on disk
        ckpt_path = args.checkpoint_save_path or os.path.join(args.checkpoint_dir, 'latest_checkpoint.pt')
        metrics_csv_path = os.path.join(args.checkpoint_dir, 'metrics.csv')

        metrics = []
        global_step = start_global_step
        cum_loss = resume_cum_loss
        cum_loss_main = resume_cum_loss_main
        cum_loss_mtp = resume_cum_loss_mtp
        ema_loss = resume_ema_loss
        ema_loss_main = resume_ema_loss_main
        ema_loss_mtp = resume_ema_loss_mtp
        ema_bal_loss = resume_ema_bal_loss
        ema_param_norm = resume_ema_param_norm
        ema_grad_norm = resume_ema_grad_norm
        ema_tokens_per_sec = resume_ema_tokens_per_sec
        ema_eval_loss = resume_ema_eval_loss
        decay = args.ema_decay

        print_rank0("Creating model...")
        model = DeepSeekV3Model(args).to(device)
        print_rank0("Model created and moved to device")

        # Load model-only checkpoint if requested (just weights)
        if args.model_checkpoint and os.path.isfile(args.model_checkpoint):
            print_rank0(f"Loading model weights from {args.model_checkpoint}")
            ckpt = torch.load(args.model_checkpoint, map_location=device)
            state_dict = ckpt.get('model_state_dict', ckpt)
            # Allow seq_len changes: drop RoPE buffers and load non-strict
            keys_to_drop = [k for k in list(state_dict.keys()) if (
                '.attn.rope_k.cos' in k or '.attn.rope_k.sin' in k or '.attn.rope_q.cos' in k or '.attn.rope_q.sin' in k
            )]
            for k in keys_to_drop:
                state_dict.pop(k, None)
            res = model.load_state_dict(state_dict, strict=False)
            print_rank0(f"Model weights loaded (non-strict). Missing: {len(res.missing_keys)}, Unexpected: {len(res.unexpected_keys)}")

        # Wrap model with DistributedDataParallel if multi-GPU
        if world_size > 1:
            print_rank0("Wrapping model with DistributedDataParallel...")
            model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[local_rank],
                find_unused_parameters=True
            )
            print_rank0("DDP wrapper applied")

        # Create scheduler
        def lr_lambda(step):
            if step < args.lr_warmup_steps:
                return float(step) / float(max(1, args.lr_warmup_steps))
            decay_steps = max(1, args.lr_decay_steps - args.lr_warmup_steps)
            return max(0.0, float(args.lr_decay_steps - step) / decay_steps)
        
        # Dummy optimizer for scheduler
        dummy_optimizer = optim.SGD([torch.zeros(1)], lr=args.lr)
        scheduler = optim.lr_scheduler.LambdaLR(dummy_optimizer, lr_lambda)

        # For offloading, we manage optimizer state manually
        print_rank0("Setting up Adam states for offloading...")
        adam_states = {
            p: {'m': torch.zeros_like(p, device='cpu').pin_memory(),
                 'v': torch.zeros_like(p, device='cpu').pin_memory()}
            for p in model.parameters()
        }
        print_rank0("Adam states initialized")

        # Load full training checkpoint if requested (complete state)
        if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
            print_rank0(f"Resuming training from full checkpoint: {args.resume_checkpoint}")
            ckpt = torch.load(args.resume_checkpoint, map_location=device)
            
            # Load model state (allow seq_len changes by skipping RoPE buffers)
            model_state_dict = ckpt['model_state_dict']
            keys_to_drop = [k for k in list(model_state_dict.keys()) if (
                '.attn.rope_k.cos' in k or '.attn.rope_k.sin' in k or '.attn.rope_q.cos' in k or '.attn.rope_q.sin' in k
            )]
            for k in keys_to_drop:
                model_state_dict.pop(k, None)
            target_model = model.module if hasattr(model, 'module') else model
            res = target_model.load_state_dict(model_state_dict, strict=False)
            print_rank0(f"Model state loaded (non-strict). Missing: {len(res.missing_keys)}, Unexpected: {len(res.unexpected_keys)}")
            
            # Load scheduler state
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                print_rank0("Scheduler state loaded from checkpoint")
            
            # Load training progress
            start_global_step = ckpt.get('global_step', 1)
            start_epoch = ckpt.get('epoch', 0)
            
            # Load EMA values
            resume_ema_loss = ckpt.get('ema_loss')
            resume_ema_loss_main = ckpt.get('ema_loss_main')
            resume_ema_loss_mtp = ckpt.get('ema_loss_mtp')
            resume_ema_bal_loss = ckpt.get('ema_bal_loss')
            resume_ema_param_norm = ckpt.get('ema_param_norm')
            resume_ema_grad_norm = ckpt.get('ema_grad_norm')
            resume_ema_tokens_per_sec = ckpt.get('ema_tokens_per_sec')
            resume_ema_eval_loss = ckpt.get('ema_eval_loss')
            
            # Load cumulative losses
            resume_cum_loss = ckpt.get('cum_loss', 0.0)
            resume_cum_loss_main = ckpt.get('cum_loss_main', 0.0)
            resume_cum_loss_mtp = ckpt.get('cum_loss_mtp', 0.0)
            
            # Load adam_states for offload mode if available
            if 'adam_states' in ckpt:
                print_rank0("Loading Adam states for offload mode...")
                saved_adam_states = ckpt['adam_states']
                # Map saved states to current model parameters
                param_mapping = {}
                for i, p in enumerate(model.parameters()):
                    param_mapping[f'param_{i}'] = p
                
                for param_key, state in saved_adam_states.items():
                    if param_key in param_mapping:
                        adam_states[param_mapping[param_key]] = state
                print_rank0("Adam states loaded from checkpoint")
            
            print_rank0(f"Resumed from epoch {start_epoch}, global step {start_global_step}")

        # Hard time limit
        TIME_LIMIT = args.time_limit
        start_time = time.time()
        time_exceeded = False

        print_rank0(f"Starting training epochs from epoch {start_epoch}...")
        for epoch in range(start_epoch, args.epochs):
            print_rank0(f"=== EPOCH {epoch+1}/{args.epochs} ===")
            model.train()
            total_loss = 0.0
            
            print_rank0(f"Starting iteration over data loader...")
            batch_count = 0
            for batch_idx, (inp, tgt_matrix) in enumerate(loader):
                batch_count += 1
                if batch_count == 1:
                    print_rank0(f"Processing first batch: {inp.shape}")
                elif batch_count % 10 == 0:
                    print_rank0(f"Processed {batch_count} batches")
                    
                step_start = time.time()
                # Check for time expiry
                if time.time() - start_time > TIME_LIMIT:
                    print_rank0(f"Time limit reached at Epoch {epoch+1}, Step {batch_idx+1}. Saving checkpoint.")
                    if rank == 0:  # Only save from rank 0
                        ckpt = {
                            'epoch': epoch + 1,
                            'batch_idx': batch_idx + 1,
                            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                            'global_step': global_step,
                            'scheduler_state_dict': scheduler.state_dict(),
                            'ema_loss': ema_loss,
                            'ema_loss_main': ema_loss_main,
                            'ema_loss_mtp': ema_loss_mtp,
                            'ema_bal_loss': ema_bal_loss,
                            'ema_param_norm': ema_param_norm,
                            'ema_grad_norm': ema_grad_norm,
                            'ema_tokens_per_sec': ema_tokens_per_sec,
                            'ema_eval_loss': ema_eval_loss,
                            'cum_loss': cum_loss,
                            'cum_loss_main': cum_loss_main,
                            'cum_loss_mtp': cum_loss_mtp
                        }
                        
                        # Add adam_states for offload mode
                        if adam_states is not None:
                            # Create serializable adam_states mapping
                            serializable_adam_states = {}
                            for i, p in enumerate(model.parameters()):
                                if p in adam_states:
                                    serializable_adam_states[f'param_{i}'] = adam_states[p]
                            ckpt['adam_states'] = serializable_adam_states
                        torch.save(ckpt, ckpt_path)
                        print_rank0(f"Checkpoint saved to {ckpt_path}")
                    time_exceeded = True
                    break

                inp = inp.to(device)
                tgt_matrix = tgt_matrix.to(device)

                real_model = model.module if hasattr(model, 'module') else model
                target_main = torch.roll(inp, shifts=-1, dims=1)
                target_main[:, -1] = args.pad_token_id

                # The offload step handles its own forward, backward, and update.
                loss_main_val, loss_mtp_val = run_offload_deepseek_step(
                    real_model, inp, target_main, tgt_matrix, args, adam_states, global_step
                )
                step_loss = torch.tensor(loss_main_val + args.mtp_weight * loss_mtp_val, device=device)

                # Since offload is a full update, we step and update biases right away.
                global_step += 1
                scheduler.step()  # Update learning rate schedule

                for layer in real_model.layers:
                    if hasattr(layer, 'moe'):
                        layer.moe.update_biases(args.bias_update_speed)
                loss_main, loss_mtp = torch.tensor(loss_main_val), torch.tensor(loss_mtp_val)

                # Update counters
                step_time = time.time() - step_start
                tokens_per_sec = inp.numel() / step_time if step_time > 0 else float('inf')

                cum_loss += step_loss.item()
                cum_loss_main += loss_main.item()
                cum_loss_mtp += (loss_mtp.item() if isinstance(loss_mtp, torch.Tensor) else float(loss_mtp))

                # Calculate expert usage balance loss
                all_counts = torch.zeros(args.n_routed_experts, device=device)
                # Handle DistributedDataParallel by accessing underlying module if wrapped
                real_model = model.module if hasattr(model, 'module') else model
                for layer in real_model.layers:
                    # Use the accumulated expert usage from MoE call
                    if layer.total_expert_usage is not None:
                        all_counts += layer.total_expert_usage.to(device).float()
                uniform_count = all_counts.sum() / args.n_routed_experts
                bal_loss = torch.abs(all_counts - uniform_count).mean()
                    
                # Calculate parameter norm (use unwrapped model for parameters)
                param_norm = 0.0
                for p in real_model.parameters():
                    param_norm += p.data.pow(2).sum().item()
                param_norm = math.sqrt(param_norm)

                # Calculate gradient norm - for offload mode, gradients are handled internally
                grad_norm = 0.0  # Placeholder since gradients are cleared in offload_step

                # Update EMAs
                if ema_loss is None:
                    ema_loss = step_loss.item()
                    ema_loss_main = loss_main.item()
                    ema_loss_mtp = loss_mtp.item() if isinstance(loss_mtp, torch.Tensor) else float(loss_mtp)
                    ema_bal_loss = bal_loss.item()
                    ema_param_norm = param_norm
                    ema_grad_norm = grad_norm
                    ema_tokens_per_sec = tokens_per_sec
                else:
                    ema_loss = decay * ema_loss + (1 - decay) * step_loss.item()
                    ema_loss_main = decay * ema_loss_main + (1 - decay) * loss_main.item()
                    ema_loss_mtp = decay * ema_loss_mtp + (1 - decay) * (loss_mtp.item() if isinstance(loss_mtp, torch.Tensor) else float(loss_mtp))
                    ema_bal_loss = decay * ema_bal_loss + (1 - decay) * bal_loss.item()
                    ema_param_norm = decay * ema_param_norm + (1 - decay) * param_norm
                    ema_grad_norm = decay * ema_grad_norm + (1 - decay) * grad_norm
                    ema_tokens_per_sec = decay * ema_tokens_per_sec + (1 - decay) * tokens_per_sec

                # Record metrics (only on rank 0)
                if rank == 0:
                    metrics.append({
                        'step': global_step,
                        'loss': step_loss.item(),
                        'loss_main': loss_main.item(),
                        'loss_mtp': loss_mtp.item() if isinstance(loss_mtp, torch.Tensor) else float(loss_mtp),
                        'bal_loss': bal_loss.item(),
                        'avg_loss': cum_loss / global_step,
                        'avg_loss_main': cum_loss_main / global_step,
                        'avg_loss_mtp': cum_loss_mtp / global_step,
                        'ema_loss': ema_loss,
                        'ema_loss_main': ema_loss_main,
                        'ema_loss_mtp': ema_loss_mtp,
                        'ema_bal_loss': ema_bal_loss,
                        'param_norm': param_norm,
                        'grad_norm': grad_norm,
                        'lr': scheduler.get_last_lr()[0],
                        'step_time': step_time,
                        'tokens_per_sec': tokens_per_sec,
                        'eval_loss': None,
                        'ema_param_norm': ema_param_norm,
                        'ema_grad_norm': ema_grad_norm,
                        'ema_tokens_per_sec': ema_tokens_per_sec,
                        'ema_eval_loss': ema_eval_loss,
                    })

                total_loss += step_loss.item()
                if global_step % 10 == 0:
                    avg = total_loss / global_step
                    print_rank0(f"Epoch {epoch+1}/{args.epochs}, Step {global_step}, Avg Loss: {avg:.4f}")

                    # Print peak GPU memory after first 10 steps
                    if global_step == 10:
                        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)  # Convert to GB
                        print_rank0(f"Peak GPU Memory after 10 steps: {peak_memory:.2f} GB")
                    
                    if global_step % args.eval_interval == 0 and val_loader is not None:
                        eval_loss = evaluate(model, val_loader, device, args)
                        if rank == 0:
                            metrics[-1]['eval_loss'] = eval_loss
                            # Update EMA for eval_loss
                            if ema_eval_loss is None:
                                ema_eval_loss = eval_loss
                            else:
                                ema_eval_loss = decay * ema_eval_loss + (1 - decay) * eval_loss
                            metrics[-1]['ema_eval_loss'] = ema_eval_loss
                        print_rank0(f"Eval loss after {global_step} steps: {eval_loss:.4f}")

                    if global_step % 100 == 0 and rank == 0:
                        ckpt = {
                            'step': global_step,
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'global_step': global_step,
                            'ema_loss': ema_loss,
                            'ema_loss_main': ema_loss_main,
                            'ema_loss_mtp': ema_loss_mtp,
                            'ema_bal_loss': ema_bal_loss,
                            'ema_param_norm': ema_param_norm,
                            'ema_grad_norm': ema_grad_norm,
                            'ema_tokens_per_sec': ema_tokens_per_sec,
                            'ema_eval_loss': ema_eval_loss,
                            'cum_loss': cum_loss,
                            'cum_loss_main': cum_loss_main,
                            'cum_loss_mtp': cum_loss_mtp
                        }
                        
                        # Add adam_states for offload mode
                    if adam_states is not None:
                            # Create serializable adam_states mapping
                            serializable_adam_states = {}
                            for i, p in enumerate(model.parameters()):
                                if p in adam_states:
                                    serializable_adam_states[f'param_{i}'] = adam_states[p]
                            ckpt['adam_states'] = serializable_adam_states

                    # overwrite single checkpoint
                    torch.save(ckpt, ckpt_path)
                    # overwrite metrics.csv
                    pd.DataFrame(metrics).to_csv(metrics_csv_path, index=False)

                    # Create individual plots for each metric
                    if MATPLOTLIB_AVAILABLE:
                        steps = [m['step'] for m in metrics]

                        # Get all metric keys except 'step' (which is our x-axis)
                        metric_keys = set()
                        for metric_dict in metrics:
                            metric_keys.update(metric_dict.keys())
                        metric_keys.discard('step')  # Remove 'step' since it's the x-axis

                        # Create individual plot for each metric
                        for metric_name in sorted(metric_keys):
                            # Check if this metric exists in all entries (some metrics like eval_loss are sparse)
                            metric_values = []
                            metric_steps = []
                            for i, m in enumerate(metrics):
                                if metric_name in m and m[metric_name] is not None:
                                    metric_values.append(m[metric_name])
                                    metric_steps.append(m['step'])

                            if metric_values:  # Only plot if we have data
                                plt.figure(figsize=(10, 6))
                                plt.plot(metric_steps, metric_values, label=metric_name, marker='o' if len(metric_values) < 50 else None)
                                plt.xlabel('step')
                                plt.ylabel(metric_name)
                                plt.title(f'{metric_name.replace("_", " ").title()} vs Step')
                                plt.grid(True, alpha=0.3)
                                plt.legend()
                                plt.tight_layout()

                                # Save each plot with the metric name
                                plot_path = os.path.join(args.checkpoint_dir, f'{metric_name}.png')
                                plt.savefig(plot_path)
                                plt.close()

            if time_exceeded:
                break

            # End of epoch logging
            print_rank0(f"End Epoch {epoch+1}, Avg Loss: {total_loss/(global_step):.4f}")
            if val_loader is not None:
                epoch_eval = evaluate(model, val_loader, device, args)
                print_rank0(f"Epoch {epoch+1} Validation Loss: {epoch_eval:.4f}")

            # overwrite single checkpoint at end of epoch (only rank 0)
            if rank == 0:
                ckpt = {
                    'epoch': epoch+1,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'global_step': global_step,
                    'scheduler_state_dict': scheduler.state_dict(),
                    'ema_loss': ema_loss,
                    'ema_loss_main': ema_loss_main,
                    'ema_loss_mtp': ema_loss_mtp,
                    'ema_bal_loss': ema_bal_loss,
                    'ema_param_norm': ema_param_norm,
                    'ema_grad_norm': ema_grad_norm,
                    'ema_tokens_per_sec': ema_tokens_per_sec,
                    'ema_eval_loss': ema_eval_loss,
                    'cum_loss': cum_loss,
                    'cum_loss_main': cum_loss_main,
                    'cum_loss_mtp': cum_loss_mtp
                }
                
                # Add adam_states for offload mode
                if adam_states is not None:
                    # Create serializable adam_states mapping
                    serializable_adam_states = {}
                    for i, p in enumerate(model.parameters()):
                        if p in adam_states:
                            serializable_adam_states[f'param_{i}'] = adam_states[p]
                    ckpt['adam_states'] = serializable_adam_states
                
                torch.save(ckpt, ckpt_path)
                print_rank0(f"Checkpoint saved to {ckpt_path}")

        # If we broke early due to time limit, exit without doing final save
        if time_exceeded:
            sys.exit(0)

        # after all epochs (only rank 0 saves)
        if args.model_save_path and rank == 0:
            final_ckpt = {
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            }
            torch.save(final_ckpt, args.model_save_path)
            print_rank0(f"Final model saved to {args.model_save_path}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
