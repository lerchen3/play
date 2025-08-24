import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import random
import gc
import shutil
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from model import LeelaChessZeroNet
from self_play import SelfPlayWorker
from cpp_interface import ChessGame, FenGame, set_mcts_params
import config
import argparse
# parse command-line arguments to override defaults
parser = argparse.ArgumentParser(description="Distributed Chess Zero training parameters")
parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE)
parser.add_argument("--model_dir", type=str, default=config.MODEL_DIR)
parser.add_argument("--num_iterations", type=int, default=config.NUM_ITERATIONS)
parser.add_argument("--games_per_iter", type=int, default=config.GAMES_PER_ITER)
parser.add_argument("--epochs_per_iter", type=int, default=config.EPOCHS_PER_ITER)
parser.add_argument("--replay_buffer_size", type=int, default=config.REPLAY_BUFFER_SIZE)
parser.add_argument("--entropy_bonus", type=float, default=config.ENTROPY_BONUS)
parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
parser.add_argument("--lr_step_size", type=int, default=config.LR_STEP_SIZE)
parser.add_argument("--lr_gamma", type=float, default=config.LR_GAMMA)
parser.add_argument("--use_fen_game", action="store_true", help="Use FenGame backend for self-play")
parser.add_argument("--load_checkpoint_path", type=str, default=None, help="Path to a checkpoint to load.")
parser.add_argument("--reset_iterations", type=int, default=0, help="Number of iterations between restarts.")
parser.add_argument("--max_moves", type=int, default=200, help="Maximum plies per self-play game")
parser.add_argument("--in_channels", type=int, default=config.IN_CHANNELS)
parser.add_argument("--num_res_blocks", type=int, default=config.NUM_RES_BLOCKS)
parser.add_argument("--num_filters", type=int, default=config.NUM_FILTERS)
parser.add_argument("--num_mcts_sims", type=int, default=config.NUM_MCTS_SIMS)
parser.add_argument("--cpuct", type=float, default=config.CPUCT)
parser.add_argument("--batch_mcts_size", type=int, default=config.BATCH_MCTS_SIZE)
parser.add_argument("--root_noise_eps", type=float, default=config.ROOT_NOISE_EPS)
parser.add_argument("--dirichlet_alpha", type=float, default=config.DIRICHLET_ALPHA)
parser.add_argument("--virtual_loss", type=float, default=config.VIRTUAL_LOSS)
args = parser.parse_args()

# Keep config consistent with CLI overrides so MCTS and data helpers
# produce states with the expected channel count.
config.IN_CHANNELS = args.in_channels
config.NUM_RES_BLOCKS = args.num_res_blocks
config.NUM_FILTERS = args.num_filters

# override config values with CLI args
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
MODEL_DIR = args.model_dir
NUM_ITERATIONS = args.num_iterations
GAMES_PER_ITER = args.games_per_iter
EPOCHS_PER_ITER = args.epochs_per_iter
REPLAY_BUFFER_SIZE = args.replay_buffer_size
ENTROPY_BONUS = args.entropy_bonus
WEIGHT_DECAY = args.weight_decay
LR_STEP_SIZE = args.lr_step_size
LR_GAMMA = args.lr_gamma
RESET_ITERATIONS = args.reset_iterations
USE_FEN_GAME = args.use_fen_game
LOAD_CHECKPOINT_PATH = args.load_checkpoint_path
MAX_MOVES = args.max_moves
IN_CHANNELS = args.in_channels
NUM_RES_BLOCKS = args.num_res_blocks
NUM_FILTERS = args.num_filters
NUM_MCTS_SIMS = args.num_mcts_sims
CPUCT = args.cpuct
BATCH_MCTS_SIZE = args.batch_mcts_size
ROOT_NOISE_EPS = args.root_noise_eps
DIRICHLET_ALPHA = args.dirichlet_alpha
VIRTUAL_LOSS = args.virtual_loss
REPLAY_BUFFER_FILE = os.path.join(MODEL_DIR, 'replay_buffer.pt')
os.makedirs(MODEL_DIR, exist_ok=True)
set_mcts_params(NUM_MCTS_SIMS, CPUCT, BATCH_MCTS_SIZE,
                ROOT_NOISE_EPS, DIRICHLET_ALPHA, VIRTUAL_LOSS)
if LOAD_CHECKPOINT_PATH:
    dst_ckpt = os.path.join(MODEL_DIR, 'model_latest.pt')
    shutil.copy2(LOAD_CHECKPOINT_PATH, dst_ckpt)
    LOAD_CHECKPOINT_PATH = dst_ckpt
from collections import deque
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import datetime

class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, pi, z = self.data[idx]
        return torch.from_numpy(s), torch.from_numpy(pi), torch.tensor(z, dtype=torch.float32)

def train(rank, world_size, num_iters, start_iter, total_iters):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # not redirecting stdout/stderr so training logs appear in notebook console
    print(f"[Rank {rank}] Initializing training process (world_size={world_size})")
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    # create CPU-based GLOO group with extended timeout for self-play sync
    gloo_group = dist.new_group(ranks=list(range(world_size)), backend='gloo', timeout=datetime.timedelta(seconds=1000000000))
    # pick actual CUDA device or fall back to CPU
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print(f"[Rank {rank}] Using device {device}")
    net = LeelaChessZeroNet(
        in_channels=IN_CHANNELS,
        num_res_blocks=NUM_RES_BLOCKS,
        num_filters=NUM_FILTERS,
    ).to(device)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[rank],
        bucket_cap_mb=50,
        find_unused_parameters=True
    )
    optimizer = torch.optim.Adam(
        net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    # replay buffer for self-play data
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    if os.path.exists(REPLAY_BUFFER_FILE):
        try:
            stored = torch.load(REPLAY_BUFFER_FILE, weights_only=False)
            replay_buffer.extend(stored)
            if rank == 0:
                print(f"[Rank {rank}] Loaded replay buffer with {len(replay_buffer)} samples")
        except Exception as e:
            if rank == 0:
                print(f"[Rank {rank}] Failed to load replay buffer: {e}")

    # resume from latest checkpoint if available
    checkpoint_to_load = None
    if LOAD_CHECKPOINT_PATH:
        checkpoint_to_load = LOAD_CHECKPOINT_PATH
    else:
        latest_checkpoint = os.path.join(MODEL_DIR, 'model_latest.pt')
        if os.path.exists(latest_checkpoint):
            checkpoint_to_load = latest_checkpoint

    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        map_loc = {'cuda:0': f'cuda:{rank}'}
        checkpoint = torch.load(checkpoint_to_load, map_location=map_loc)
        
        state_dict_to_load = checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict_to_load = checkpoint['model_state_dict']
        
        # remove `module.` prefix if present
        unwrapped_state_dict = {k.replace('module.', ''): v for k, v in state_dict_to_load.items()}
        
        net.module.load_state_dict(unwrapped_state_dict)
        if rank == 0:
            print(f"Resumed training from {checkpoint_to_load}")

    # iterative self-play & training
    for iteration in range(start_iter, start_iter + num_iters):
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[Rank {rank}] Iteration {iteration}/{total_iters}")
        random.seed(iteration)
        remove_n = int(len(replay_buffer) * 0.2)
        if remove_n > 0:
            keep = set(range(len(replay_buffer))) - set(random.sample(range(len(replay_buffer)), remove_n))
            replay_buffer = deque([x for i, x in enumerate(replay_buffer) if i in keep], maxlen=REPLAY_BUFFER_SIZE)

        # --- Self-play data generation ---
        print(f"[Rank {rank}] Generating {GAMES_PER_ITER} self-play games")
        net.eval()  # switch to evaluation mode for self-play
        game_cls = FenGame if USE_FEN_GAME else ChessGame
        sp = SelfPlayWorker(net, device=device, num_games=GAMES_PER_ITER, game_cls=game_cls, max_moves=MAX_MOVES)
        new_data = sp.self_play()
        # gather new data across all ranks - variable self-play duration; split evenly between ranks
        print(f"[Rank {rank}] Collected {len(new_data)} samples (not aggregated)")
        all_new_data = [None] * world_size
        dist.all_gather_object(all_new_data, new_data, group=gloo_group)
        # flatten aggregated data
        new_data = [sample for sublist in all_new_data for sample in sublist]
        print(f"[Rank {rank}] Collected {len(new_data)} samples (aggregated across ranks)")
        replay_buffer.extend(new_data)

        # synchronize ranks before training to avoid DDP deadlock due to variable self-play durations
        print(f"[Rank {rank}] Waiting for all ranks before starting training")
        dist.barrier(group=gloo_group)

        # --- Training on replay buffer ---
        net.train()  # switch back to training mode
        dataset = ChessDataset(list(replay_buffer))
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
        
        for epoch in range(1, EPOCHS_PER_ITER + 1):
            print(f"[Rank {rank}] Training epoch {epoch}/{EPOCHS_PER_ITER}")
            sampler.set_epoch(epoch)
            for batch_idx, (s, pi, z) in enumerate(loader, start=1):
                s = s.to(device)
                pi = pi.to(device)
                z = z.to(device)
                pred_log_pi, pred_v = net(s)
                # policy loss
                loss_pi = -torch.mean(torch.sum(pi * pred_log_pi, dim=1))
                # value loss
                pred_v_flat = pred_v.view(-1)
                loss_v = torch.mean((z - pred_v_flat) ** 2)
                # entropy regularization
                prob_pi = pred_log_pi.exp()
                entropy = -torch.mean(torch.sum(prob_pi * pred_log_pi, dim=1))
                loss = loss_pi + loss_v + ENTROPY_BONUS * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if rank == 0:
                print(f"[Rank {rank}] Completed training epoch {epoch}")
        # LR scheduler step
        scheduler.step() # every LR_STEP_SIZE * EPOCHS_PER_ITER
        # save checkpoint and replay buffer
        if rank == 0:
            torch.save(net.module.state_dict(), os.path.join(MODEL_DIR, 'model_latest.pt'))
            torch.save(list(replay_buffer), REPLAY_BUFFER_FILE)
        dist.barrier(group=gloo_group)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        gc.collect()
    dist.destroy_process_group()
    print(f"[Rank {rank}] Training process exiting")

def main():
    world_size = torch.cuda.device_count()
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')

    total_remaining = NUM_ITERATIONS
    start_iter = 1
    reset = RESET_ITERATIONS if RESET_ITERATIONS > 0 else NUM_ITERATIONS

    while total_remaining > 0:
        run_iters = min(reset, total_remaining)
        mp.spawn(train, args=(world_size, run_iters, start_iter, NUM_ITERATIONS), nprocs=world_size)
        total_remaining -= run_iters
        start_iter += run_iters
        # after first loop, ensure we load from latest checkpoint
        globals()['LOAD_CHECKPOINT_PATH'] = None

if __name__ == '__main__':
    main() 
