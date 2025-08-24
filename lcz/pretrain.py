import os
import argparse
import pandas as pd
import numpy as np
import torch
# Disable cuDNN to avoid its workspace allocations leading to OOM
os.environ["CUDNN_WORKSPACE_LIMIT_IN_MB"] = "4096"
import time
import random
import re
from torch.utils.data import IterableDataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from model import LeelaChessZeroNet
from move_encoding import MOVE_INDEX
from cpp_interface import ChessGame, FenGame
import config
import matplotlib.pyplot as plt

class StreamingCSVGameDataset(IterableDataset):
    """Streaming dataset that processes chess games one at a time from CSV."""
    def __init__(self, csv_path, accelerator=None, cutoff=None, start_row=None, chunksize=1, eval_fraction=0.0, mode='train', game_cls=FenGame):
        self.csv_path = csv_path
        self.accelerator = accelerator
        self.cutoff = cutoff
        self.start_row = start_row
        self.chunksize = chunksize
        self.mode = mode
        self.game_cls = game_cls
    def _process_row(self, row, idx, cutoff, rank, world_size):
        """Helper to process a single CSV row into training samples."""
        samples = []
        if cutoff is not None and idx >= cutoff:
            return samples
        if(idx % 1000000 == 0):
            print(f"idx {idx} btw")
        if idx % world_size != rank:
            return samples
        if self.start_row is not None and idx < self.start_row:
            return samples
        result_str = str(row.get("Result", ""))
        if result_str == "1-0":
            result_val = 1
        elif result_str == "0-1":
            result_val = -1
        elif result_str == "1/2-1/2":
            result_val = 0
        raw_moves = row["Moves"]
        moves = re.findall(r"'(.*?)'", str(raw_moves))
        game = self.game_cls()
        states, actions, players = [], [], []
        valid = True
        for mv in moves:
            if mv not in MOVE_INDEX:
                raise ValueError(f"mv {mv} not in MOVE_INDEX")
            states.append(game.get_state().numpy())
            players.append(game.current_player())
            actions.append(MOVE_INDEX[mv])
            if not game.move(mv):
                valid = False
                break
        if not valid:
            return []
        for st, act, pl in zip(states, actions, players):
            pi_target = np.zeros(len(MOVE_INDEX), dtype=np.float32)
            pi_target[act] = 1.0
            z_target = result_val * pl
            samples.append((
                torch.from_numpy(st),
                torch.from_numpy(pi_target),
                torch.tensor(z_target, dtype=torch.float32)
            ))
        return samples
    def __iter__(self):
        # DDP sharding
        if self.accelerator:
            rank = self.accelerator.process_index
            world_size = self.accelerator.num_processes
        else:
            rank, world_size = 0, 1

        # DataLoader worker sharding
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1

        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunksize):
            # apply DDP sharding
            if world_size > 1:
                chunk = chunk.iloc[rank::world_size]
            # apply DataLoader sharding
            if num_workers > 1:
                chunk = chunk.iloc[worker_id::num_workers]

            for idx, row in chunk.iterrows():  # use the original DataFrame index directly
                samples = self._process_row(row, idx, self.cutoff, rank, world_size)
                for st, pi, z in samples:
                    yield st, pi, z


def main():
    parser = argparse.ArgumentParser(description="Supervised pretraining from CSV games")
    parser.add_argument("--data_csv_path", required=True)
    parser.add_argument("--output_model", default=os.path.join(config.MODEL_DIR, "pretrain_latest.pt"))
    parser.add_argument("--input_model", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--time_limit", type=int, default=41400, help="Time limit in seconds (default: 11h 30min)")
    parser.add_argument("--cutoff", type=int, default=None, help="number of rows of csv to use")
    parser.add_argument("--start_row", type=int, default=None, help="index of first row to process")
    parser.add_argument("--eval_fraction", type=float, default=0.02, help="Fraction of data to use for evaluation (train fraction is 1 - eval_fraction)")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Number of training steps between evaluations")
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of training steps between logging training metrics")
    parser.add_argument("--use_fen_game", action="store_true", help="Use FenGame backend instead of ChessGame")
    parser.add_argument("--chunksize", type=int, default=1000,
                        help="Number of rows to read per chunk. Larger is more efficient.")
    parser.add_argument("--in_channels", type=int, default=config.IN_CHANNELS,
                        help="Number of input feature channels")
    parser.add_argument("--num_res_blocks", type=int, default=config.NUM_RES_BLOCKS,
                        help="Number of residual blocks")
    parser.add_argument("--num_filters", type=int, default=config.NUM_FILTERS,
                        help="Number of convolutional filters")
    args = parser.parse_args()
    # Keep config aligned with CLI overrides so state representation matches
    config.IN_CHANNELS = args.in_channels
    config.NUM_RES_BLOCKS = args.num_res_blocks
    config.NUM_FILTERS = args.num_filters
    start_time = time.time()
    os.makedirs("training_plots", exist_ok=True)
    # Set up DDP
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    game_cls = FenGame if args.use_fen_game else ChessGame
    full_df = pd.read_csv(args.data_csv_path)
    total_rows = len(full_df)
    print(f"Total rows in dataset: {total_rows}")

    train_frac = 1.0 - args.eval_fraction
    train_df = full_df.sample(frac=train_frac, random_state=42)
    eval_df = full_df.drop(train_df.index)

    train_csv = "train.csv"
    eval_csv = "eval.csv"
    train_df.to_csv(train_csv, index=False)
    eval_df.to_csv(eval_csv, index=False)

    accelerator.print(f"Saved train CSV to {train_csv} ({len(train_df)} rows), eval CSV to {eval_csv} ({len(eval_df)} rows)")

    # Create streaming datasets and loaders
    train_ds = StreamingCSVGameDataset(
        train_csv,
        accelerator,
        args.cutoff,
        args.start_row,
        chunksize=args.chunksize,
        mode='train',
        game_cls=game_cls,
    )
    eval_ds = StreamingCSVGameDataset(
        eval_csv,
        accelerator,
        args.cutoff,
        None, # no start row.
        chunksize=args.chunksize,
        mode='eval',
        game_cls=game_cls,
    )

    # Use 8 workers for data loading
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Model, optimizer setup
    device = accelerator.device
    net = LeelaChessZeroNet(
        in_channels=args.in_channels,
        num_res_blocks=args.num_res_blocks,
        num_filters=args.num_filters,
    ).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate,
                                 weight_decay=1e-4)

    # Load checkpoint if provided
    # however, just start over @ epoch 1, step 1.
    start_epoch, start_step = 1, 0
    if args.input_model and os.path.exists(args.input_model):
        ckpt = torch.load(args.input_model, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            net.load_state_dict(ckpt['model_state_dict'])
            accelerator.print(f"Resumed from {args.input_model} at epoch {start_epoch}, step {start_step}")
        else:
            net.load_state_dict(ckpt)
            accelerator.print(f"Loaded model from {args.input_model}")

    net, optimizer, train_loader, eval_loader = accelerator.prepare(
        net, optimizer, train_loader, eval_loader)
    # Load optimizer state after prepare if resuming
    if 'ckpt' in locals() and isinstance(ckpt, dict) and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # Initialize metrics tracking
    train_steps = []
    train_losses = []
    train_ema_losses = []
    train_loss_ema = None
    eval_steps = []
    eval_losses = []
    eval_ema_losses = []
    eval_loss_ema = None
    # Track best evaluation model
    best_eval_loss = float('inf')
    best_eval_model_path = os.path.join(config.MODEL_DIR, "best_eval_model.pt")
    best_eval_epoch = None
    best_eval_step = None

    for epoch in range(start_epoch, args.epochs + 1):
        net.train()
        step_num = 0
        # skip steps if resuming
        skip_steps = start_step if epoch == start_epoch else 0
        total_loss = 0
        total_loss_pi = 0
        total_loss_v = 0
        num_batches = 0
        
        for s, pi, z in train_loader:
            # skip previously processed steps when resuming
            if skip_steps > 0:
                skip_steps -= 1
                step_num += 1
                continue
            # Check time limit
            if time.time() - start_time > args.time_limit:
                accelerator.print(f"Time limit of {args.time_limit} seconds reached. Stopping training.")
                torch.save({
                    'epoch': epoch,
                    'step': step_num,
                    'model_state_dict': accelerator.unwrap_model(net).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, args.output_model)
                accelerator.print(f"Saved final checkpoint to {args.output_model}")
                # Plot before exiting due to time limit
                plt.figure()
                plt.plot(train_steps, train_losses, label='Train Loss')
                plt.plot(train_steps, train_ema_losses, label='Train Loss EMA')
                if eval_losses:
                    plt.plot(eval_steps, eval_losses, label='Eval Loss')
                    plt.plot(eval_steps, eval_ema_losses, label='Eval Loss EMA')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(os.path.join("training_plots", "loss_plot.png"))
                plt.close()
                return
            
            s = s.to(device)
            pi = pi.to(device)
            z = z.to(device)
            log_pi, v = net(s)
            loss_pi = -(pi * log_pi).sum(dim=1).mean()
            loss_v = ((z - v.view(-1)) ** 2).mean()
            loss = loss_pi + loss_v
            
            # Accumulate losses for averaging
            total_loss += loss.item()
            total_loss_pi += loss_pi.item()
            total_loss_v += loss_v.item()
            num_batches += 1
            
            # Log every logging_steps
            if step_num % args.logging_steps == 0 and step_num > 0:
                avg_loss = total_loss / num_batches
                avg_loss_pi = total_loss_pi / num_batches
                avg_loss_v = total_loss_v / num_batches
                accelerator.print(f"Epoch {epoch}, Step {step_num}, Avg Loss: {avg_loss:.4f} (policy: {avg_loss_pi:.4f}, value: {avg_loss_v:.4f})")
                # Record training metrics
                train_steps.append(step_num)
                train_losses.append(avg_loss)
                if train_loss_ema is None:
                    train_loss_ema = avg_loss
                else:
                    train_loss_ema = train_loss_ema * 0.99 + avg_loss * 0.01
                train_ema_losses.append(train_loss_ema)
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            step_num += 1
            # Evaluate every eval_steps (after parameter update)
            if step_num % args.eval_steps == 0 and step_num > 0:
                torch.cuda.empty_cache()
                net.eval()
                eval_loss_pi = 0.0
                eval_loss_v = 0.0
                eval_batches = 0
                with torch.no_grad():
                    for s_eval, pi_eval, z_eval in eval_loader:
                        s_eval = s_eval.to(device)
                        pi_eval = pi_eval.to(device)
                        z_eval = z_eval.to(device)
                        log_pi_eval, v_eval = net(s_eval)
                        loss_pi_eval = -(pi_eval * log_pi_eval).sum(dim=1).mean()
                        loss_v_eval = ((z_eval - v_eval.view(-1)) ** 2).mean()
                        eval_loss_pi += loss_pi_eval.item()
                        eval_loss_v += loss_v_eval.item()
                        eval_batches += 1
                avg_eval_pi = eval_loss_pi / eval_batches
                avg_eval_v = eval_loss_v / eval_batches
                accelerator.print(f"Epoch {epoch}, Step {step_num}, Eval Loss: {(avg_eval_pi + avg_eval_v):.4f} (policy: {avg_eval_pi:.4f}, value: {avg_eval_v:.4f})")
                # Record evaluation metrics
                eval_steps.append(step_num)
                eval_loss_val = avg_eval_pi + avg_eval_v
                eval_losses.append(eval_loss_val)
                if eval_loss_ema is None:
                    eval_loss_ema = eval_loss_val
                else:
                    eval_loss_ema = eval_loss_ema * 0.99 + eval_loss_val * 0.01
                eval_ema_losses.append(eval_loss_ema)
                # Save best eval model
                if eval_loss_val < best_eval_loss:
                    best_eval_loss = eval_loss_val
                    best_eval_epoch = epoch
                    best_eval_step = step_num
                    torch.save({
                        'epoch': epoch,
                        'step': step_num,
                        'model_state_dict': accelerator.unwrap_model(net).state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, best_eval_model_path)
                    accelerator.print(f"New best eval model saved with loss {best_eval_loss:.4f} at epoch {best_eval_epoch}, step {best_eval_step} to {best_eval_model_path}")
                net.train()
        accelerator.print(f"Completed epoch {epoch}/{args.epochs}")
        torch.save({
            'epoch': epoch,
            'step': step_num,
            'model_state_dict': accelerator.unwrap_model(net).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, args.output_model)
        accelerator.print(f"Saved checkpoint to {args.output_model}")

    # Plot after training completes
    plt.figure()
    plt.plot(train_steps, train_losses, label='Train Loss')
    plt.plot(train_steps, train_ema_losses, label='Train Loss EMA')
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label='Eval Loss')
        plt.plot(eval_steps, eval_ema_losses, label='Eval Loss EMA')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join("training_plots", "loss_plot.png"))
    plt.close()

if __name__ == "__main__":
    main()
