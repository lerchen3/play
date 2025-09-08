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

# Add project root to Python path for all imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


def get_model_class(architecture):
    """Import and return the appropriate model class based on architecture"""
    if architecture == "qwen3":
        from train.qwen3_model import Qwen3Model
        return Qwen3Model
    elif architecture == "deepseekv3":
        from train.model import DeepSeekV3Model
        return DeepSeekV3Model
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


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
        
        # Handle different tokenizer return types
        if isinstance(encoded, dict) or hasattr(encoded, 'input_ids'):
            input_ids = (encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids).squeeze(0)
        else:
            input_ids = encoded.squeeze(0)

        # For MTP (Multi-Token Prediction) if using DeepSeek-V3
        if self.mtp_depth > 0:
            tgt_matrix = torch.zeros(self.seq_len, self.mtp_depth, dtype=torch.long)
            for i in range(self.seq_len - 1):
                for j in range(self.mtp_depth):
                    tgt_matrix[i, j] = input_ids[i + j + 1] if (i + j + 1) < self.seq_len else self.pad_id
            return input_ids, tgt_matrix
        else:
            return input_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Train models with different architectures.")
    
    # Architecture selection
    parser.add_argument('--architecture', type=str, choices=['qwen3', 'deepseekv3'], required=True,
                       help='Model architecture to use')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None, help='Path to CSV with chat data')
    parser.add_argument('--user_column', type=str, default='user', help='User column name in CSV')
    parser.add_argument('--assistant_column', type=str, default='assistant', help='Assistant column name in CSV')
    
    # Training arguments
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    
    # Common model arguments
    parser.add_argument('--d_model', type=int, help='Model hidden dimension (default varies by arch)')
    parser.add_argument('--num_layers', type=int, help='Number of transformer layers (default varies by arch)')
    parser.add_argument('--rmsnorm_eps', type=float, default=1e-6, help='RMSNorm epsilon')
    
    # Qwen3-specific arguments
    parser.add_argument('--n_q_heads', type=int, help='Number of query heads (Qwen3)')
    parser.add_argument('--n_kv_heads', type=int, help='Number of key-value heads (Qwen3, GQA)')
    parser.add_argument('--d_ff', type=int, help='Feed-forward dimension (Qwen3)')
    
    # DeepSeek-V3 specific arguments  
    parser.add_argument('--n_heads', type=int, help='Number of attention heads (DeepSeekV3)')
    parser.add_argument('--dc_kv', type=int, help='Key-value compression dimension (DeepSeekV3 MLA)')
    parser.add_argument('--dc_q', type=int, help='Query compression dimension (DeepSeekV3 MLA)')
    parser.add_argument('--d_head', type=int, help='Attention head dimension override')
    parser.add_argument('--n_shared_experts', type=int, help='Number of shared experts (DeepSeekV3 MoE)')
    parser.add_argument('--n_routed_experts', type=int, help='Number of routed experts (DeepSeekV3 MoE)')
    parser.add_argument('--k_routed_experts', type=int, help='Number of active routed experts (DeepSeekV3 MoE)')
    parser.add_argument('--d_ff_expert_mult', type=int, help='Expert FFN dimension multiplier (DeepSeekV3)')
    parser.add_argument('--moe_balance_factor', type=float, help='MoE load balancing factor (DeepSeekV3)')
    parser.add_argument('--bias_update_speed', type=float, help='MoE bias update speed (DeepSeekV3)')
    parser.add_argument('--mtp_depth', type=int, default=0, help='Multi-token prediction depth (DeepSeekV3)')
    parser.add_argument('--mtp_weight', type=float, default=0.5, help='Weight for MTP loss (DeepSeekV3)')
    
    # Optimizer arguments
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='Adam epsilon')
    
    # Learning rate schedule
    parser.add_argument('--lr_warmup_steps', type=int, default=1000, help='LR warmup steps')
    parser.add_argument('--lr_decay_steps', type=int, default=10000, help='LR decay steps')
    
    # Checkpointing and evaluation
    parser.add_argument('--model_checkpoint', type=str, default=None, help='Model checkpoint path for resuming')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Full training checkpoint path')
    parser.add_argument('--checkpoint_save_path', type=str, default=None, help='Path to save checkpoints')
    parser.add_argument('--model_save_path', type=str, default=None, help='Path to save final model')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split fraction')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval in steps')
    
    # Misc
    parser.add_argument('--time_limit', type=int, default=11*3600+30*60, help='Training time limit in seconds')
    parser.add_argument('--ema_decay', type=float, default=0.9, help='EMA decay for loss tracking')
    
    args = parser.parse_args()
    
    # Set architecture-specific defaults
    if args.architecture == "qwen3":
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
    
    return args


def evaluate(model, loader, device, args):
    """Evaluate the model on validation data"""
    model.eval()
    total_loss = 0.0
    count = 0
    
    if loader is None:
        return float('nan')
        
    with torch.no_grad():
        for batch in loader:
            if args.architecture == "deepseekv3" and args.mtp_depth > 0:
                input_ids, tgt_matrix = batch
                input_ids = input_ids.to(device)
                tgt_matrix = tgt_matrix.to(device)
                
                # Get model (unwrap DDP if needed)
                real_model = model.module if hasattr(model, 'module') else model
                target_main = torch.roll(input_ids, shifts=-1, dims=1)
                target_main[:, -1] = args.pad_token_id
                
                losses = real_model(input_ids, target_main=target_main, tgt_matrix=tgt_matrix, is_training=False)
                loss_main, loss_mtp = losses[0], losses[1]
                loss = loss_main + args.mtp_weight * loss_mtp
            else:
                input_ids = batch if not isinstance(batch, (list, tuple)) else batch[0]
                input_ids = input_ids.to(device)
                target_main = torch.roll(input_ids, shifts=-1, dims=1)
                target_main[:, -1] = args.pad_token_id
                
                loss = model(input_ids, target_main=target_main, is_training=True)
                
            total_loss += loss.item()
            count += 1
            
    model.train()
    return total_loss / max(1, count)


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    # Initialize distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    print(f"[RANK {rank}] Starting training on device: {device}", flush=True)
    
    try:
        args = parse_args()
        
        # Initialize tokenizer
        print(f"[RANK {rank}] Loading tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/qwen-3/transformers/0.6b/1")

        # Ensure pad token exists
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Set model parameters
        args.vocab_size = len(tokenizer)
        args.pad_token_id = tokenizer.pad_token_id
        args.device = device
        
        def print_rank0(*msg):
            if rank == 0:
                print(*msg)

        print_rank0(f"Starting {args.architecture} training with {world_size} GPUs")
        print_rank0(f"Vocab size: {args.vocab_size}")
        print_rank0(f"Sequence length: {args.seq_len}")

        # Load and split data
        if args.data_path:
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
            mtp_depth = args.mtp_depth if args.architecture == "deepseekv3" else 0
            train_dataset = ChatDataset(train_users, train_assistants, tokenizer, args.seq_len, mtp_depth)
            val_dataset = ChatDataset(val_users, val_assistants, tokenizer, args.seq_len, mtp_depth) if val_users else None
            
            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None
        else:
            print_rank0("No data path provided - creating dummy data for testing")
            dummy_users = ["Hello", "How are you?", "What's the weather?"] * 20
            dummy_assistants = ["Hi there!", "I'm doing well!", "It's sunny today!"] * 20
            mtp_depth = args.mtp_depth if args.architecture == "deepseekv3" else 0
            train_dataset = ChatDataset(dummy_users, dummy_assistants, tokenizer, args.seq_len, mtp_depth)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = None

        # Create model
        print_rank0(f"Creating {args.architecture} model...")
        ModelClass = get_model_class(args.architecture)
        model = ModelClass(args).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print_rank0(f"Model created with {total_params:,} parameters ({total_params/1e6:.1f}M)")

        # Load model checkpoint if provided
        if args.model_checkpoint and os.path.isfile(args.model_checkpoint):
            print_rank0(f"Loading model weights from {args.model_checkpoint}")
            ckpt = torch.load(args.model_checkpoint, map_location=device)
            state_dict = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state_dict, strict=False)
            print_rank0("Model weights loaded")

        # Wrap model with DDP if multi-GPU
        if world_size > 1:
            print_rank0("Wrapping model with DistributedDataParallel...")
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                find_unused_parameters=True
            )

        # Create optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2), 
            eps=args.adam_eps
        )
        
        def lr_lambda(step):
            if step < args.lr_warmup_steps:
                return float(step) / float(max(1, args.lr_warmup_steps))
            decay_steps = max(1, args.lr_decay_steps - args.lr_warmup_steps)
            return max(0.0, float(args.lr_decay_steps - step) / decay_steps)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Training state
        global_step = 1
        start_time = time.time()
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Metrics tracking
        metrics = []
        ema_loss = None
        cum_loss = 0.0
        
        print_rank0("Starting training...")
        for epoch in range(args.epochs):
            print_rank0(f"=== EPOCH {epoch+1}/{args.epochs} ===")
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                # Check time limit
                if time.time() - start_time > args.time_limit:
                    print_rank0(f"Time limit reached. Saving checkpoint...")
                    if rank == 0 and args.checkpoint_save_path:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'global_step': global_step,
                        }, args.checkpoint_save_path)
                        print_rank0(f"Checkpoint saved")
                    return
                
                # Handle different architectures
                if args.architecture == "deepseekv3" and args.mtp_depth > 0:
                    input_ids, tgt_matrix = batch
                    input_ids = input_ids.to(device)
                    tgt_matrix = tgt_matrix.to(device)
                    
                    # Get model (unwrap DDP if needed)
                    real_model = model.module if hasattr(model, 'module') else model
                    target_main = torch.roll(input_ids, shifts=-1, dims=1)
                    target_main[:, -1] = args.pad_token_id
                    
                    losses = real_model(input_ids, target_main=target_main, tgt_matrix=tgt_matrix, is_training=True)
                    loss_main, loss_mtp = losses[0], losses[1]
                    loss = loss_main + args.mtp_weight * loss_mtp
                else:
                    input_ids = batch if not isinstance(batch, (list, tuple)) else batch[0]
                    input_ids = input_ids.to(device)
                    target_main = torch.roll(input_ids, shifts=-1, dims=1)
                    target_main[:, -1] = args.pad_token_id
                    
                    real_model = model.module if hasattr(model, 'module') else model
                    loss = real_model(input_ids, target_main=target_main, is_training=True)
                
                loss = loss / args.grad_accum_steps
                loss.backward()
                
                # Update on accumulation steps
                if (batch_idx + 1) % args.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Update routing biases for DeepSeekV3 MoE
                    if args.architecture == "deepseekv3":
                        real_model = model.module if hasattr(model, 'module') else model
                        for layer in real_model.layers:
                            if hasattr(layer, 'moe'):
                                layer.moe.update_biases(args.bias_update_speed)
                    
                    # Update metrics
                    step_loss = loss.item() * args.grad_accum_steps
                    epoch_loss += step_loss
                    cum_loss += step_loss
                    
                    if ema_loss is None:
                        ema_loss = step_loss
                    else:
                        ema_loss = args.ema_decay * ema_loss + (1 - args.ema_decay) * step_loss
                    
                    # Record metrics
                    if rank == 0:
                        metrics.append({
                            'step': global_step,
                            'epoch': epoch + 1,
                            'loss': step_loss,
                            'avg_loss': cum_loss / global_step,
                            'ema_loss': ema_loss,
                            'lr': scheduler.get_last_lr()[0],
                        })
                    
                    # Logging
                    if global_step % 10 == 0:
                        avg_loss = cum_loss / global_step
                        print_rank0(f"Epoch {epoch+1}, Step {global_step}, Loss: {step_loss:.4f}, Avg: {avg_loss:.4f}, EMA: {ema_loss:.4f}")
                    
                    # Evaluation
                    if global_step % args.eval_interval == 0 and val_loader is not None:
                        eval_loss = evaluate(model, val_loader, device, args)
                        print_rank0(f"Validation loss at step {global_step}: {eval_loss:.4f}")
                        if rank == 0 and metrics:
                            metrics[-1]['eval_loss'] = eval_loss
                    
                    # Save checkpoint periodically
                    if global_step % 100 == 0 and rank == 0 and args.checkpoint_save_path:
                        ckpt = {
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'global_step': global_step,
                            'ema_loss': ema_loss,
                            'cum_loss': cum_loss,
                        }
                        torch.save(ckpt, args.checkpoint_save_path)
                        
                        # Save metrics
                        if metrics:
                            metrics_path = os.path.join(args.checkpoint_dir, 'metrics.csv')
                            pd.DataFrame(metrics).to_csv(metrics_path, index=False)

            # End of epoch
            avg_epoch_loss = epoch_loss / (len(train_loader) // args.grad_accum_steps)
            print_rank0(f"End of epoch {epoch+1}, Average loss: {avg_epoch_loss:.4f}")

        # Save final model
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
