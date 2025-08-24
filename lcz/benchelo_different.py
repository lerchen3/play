#!/usr/bin/env python
"""
Enhanced ELO Benchmarking Script

Supports:
- JSON configurations passed directly from CLI
- Pure network mode (no MCTS, just direct network evaluation)
- Reset iterations functionality (similar to train.py)
- Randomized matchmaking instead of sequential
- Time-based limits instead of game count limits
"""

import argparse
import itertools
import json
import os
import config
import random
import csv
import numpy as np
import torch
import time
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from model import LeelaChessZeroNet
from cpp_interface import ChessGame, MCTS, set_mcts_params
from move_encoding import INDEX_MOVE, MOVE_INDEX

DEFAULT_STATS_FILE = 'elo_stats_different.json'
INITIAL_ELO = 1200
K_FACTOR = 32
DEFAULT_TIME_LIMIT = 41400  # 11 hours 30 minutes

@dataclass
class ModelConfig:
    """Configuration for a model including MCTS parameters"""
    path: str
    name: str = ""
    
    # Model architecture
    in_channels: int = config.IN_CHANNELS
    num_res_blocks: int = config.NUM_RES_BLOCKS  
    num_filters: int = config.NUM_FILTERS
    
    # MCTS parameters
    num_mcts_sims: int = config.NUM_MCTS_SIMS
    cpuct: float = config.CPUCT
    batch_mcts_size: int = config.BATCH_MCTS_SIZE
    root_noise_eps: float = config.ROOT_NOISE_EPS
    dirichlet_alpha: float = config.DIRICHLET_ALPHA
    virtual_loss: float = config.VIRTUAL_LOSS
    
    # Pure network mode (no MCTS)
    pure_network: bool = False

    def __post_init__(self):
        if not self.name:
            self.name = Path(self.path).stem

def load_net(path, device, in_channels, num_res_blocks, num_filters):
    """Load a standard neural network"""
    net = LeelaChessZeroNet(
        in_channels=in_channels,
        num_res_blocks=num_res_blocks, 
        num_filters=num_filters,
    ).to(device)
    checkpoint = torch.load(path, map_location=device)
    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    net.load_state_dict(checkpoint)
    net.eval()
    return net

def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(r_a, r_b, score_a, score_b):
    exp_a = expected_score(r_a, r_b)
    exp_b = expected_score(r_b, r_a)
    new_a = r_a + K_FACTOR * (score_a - exp_a)
    new_b = r_b + K_FACTOR * (score_b - exp_b)
    return new_a, new_b

def get_pure_network_move(net, game, device):
    """Get move using pure network evaluation (no MCTS)"""
    state = game.get_state()
    state_tensor = state.unsqueeze(0).to(device)
    
    with torch.no_grad():
        log_pi, _ = net(state_tensor)
        policy = torch.exp(log_pi).cpu().numpy().flatten()
    
    # Get legal moves as UCI strings and convert to indices
    legal_moves_uci = game.legal_moves()
    legal_indices = []
    for move_uci in legal_moves_uci:
        if move_uci in MOVE_INDEX:
            legal_indices.append(MOVE_INDEX[move_uci])
    
    # Mask illegal moves
    legal_policy = np.zeros_like(policy)
    for move_idx in legal_indices:
        legal_policy[move_idx] = policy[move_idx]
    
    # Normalize
    if legal_policy.sum() > 0:
        legal_policy = legal_policy / legal_policy.sum()
    else:
        print("wtf legal policy sum is 0")
    
    return legal_policy

def play_game(config_w: ModelConfig, config_b: ModelConfig, 
              net_w, net_b, device: str, fen=None, 
              record_game=False, max_moves=200):
    """Play a game between two models with different configurations"""
    game = ChessGame(fen=fen) if fen else ChessGame()
    
    pgn_moves = []
    move_count = 0

    while not game.is_game_over() and move_count < max_moves:
        if game.current_player() == 1:
            # White's turn
            config_current = config_w
            net_current = net_w
        else:
            # Black's turn  
            config_current = config_b
            net_current = net_b
        
        # Get policy based on configuration
        if config_current.pure_network:
            # Pure network mode - direct evaluation
            policy = get_pure_network_move(net_current, game, device)
        else:
            # Standard MCTS
            set_mcts_params(
                config_current.num_mcts_sims, config_current.cpuct, config_current.batch_mcts_size,
                config_current.root_noise_eps, config_current.dirichlet_alpha, config_current.virtual_loss
            )
            mcts = MCTS(net_current, device=device)
            policy = mcts.search(game)
            
        move_idx = np.random.choice(len(policy), p=policy)
        uci = INDEX_MOVE[move_idx]
        
        if record_game:
            pgn_moves.append(uci)  # Use UCI notation instead of algebraic

        game.move(uci)
        move_count += 1

    if move_count >= max_moves and not game.is_game_over():
        result = 0
    else:
        result = game.result()

    pgn = ""
    if record_game:
        m_no = 1
        for i, m in enumerate(pgn_moves):
            if i % 2 == 0:
                pgn += f"{m_no}. {m} "
                m_no += 1
            else:
                pgn += f"{m} "
        pgn = pgn.strip()

    if result == 1:
        score_w, score_b = 1.0, 0.0
        if record_game: pgn += " 1-0"
    elif result == -1:
        score_w, score_b = 0.0, 1.0
        if record_game: pgn += " 0-1"
    else:
        score_w, score_b = 0.5, 0.5
        if record_game: pgn += " 1/2-1/2"
    
    if record_game:
        return score_w, score_b, pgn
    return score_w, score_b, None

def load_model_configs_from_json(json_string: str) -> List[ModelConfig]:
    """Load model configurations from JSON string"""
    configs_data = json.loads(json_string)
    
    configs = []
    for config_data in configs_data:
        config_obj = ModelConfig(**config_data)
        configs.append(config_obj)
    
    return configs

def create_default_config_string(models: List[str]) -> str:
    """Create a default configuration JSON string for the given models"""
    configs = []
    for i, model_path in enumerate(models):
        config_data = {
            "path": model_path,
            "name": f"model_{i}",
            "in_channels": config.IN_CHANNELS,
            "num_res_blocks": config.NUM_RES_BLOCKS,
            "num_filters": config.NUM_FILTERS,
            "num_mcts_sims": config.NUM_MCTS_SIMS,
            "cpuct": config.CPUCT,
            "batch_mcts_size": config.BATCH_MCTS_SIZE,
            "root_noise_eps": config.ROOT_NOISE_EPS,
            "dirichlet_alpha": config.DIRICHLET_ALPHA,
            "virtual_loss": config.VIRTUAL_LOSS,
            "pure_network": False
        }
        configs.append(config_data)
    
    return json.dumps(configs, indent=2)

def load_stats(configs: List[ModelConfig], stats_file: str):
    """Load ELO statistics"""
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = {}
    
    for config_obj in configs:
        key = config_obj.name
        stats.setdefault(key, {
            'elo': INITIAL_ELO,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'games': 0,
            'config': config_obj.__dict__
        })
    return stats

def save_stats(stats, stats_file):
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

def reset_environment(stats_file: str, games_csv_file: str):
    """Reset the environment by clearing stats and games files"""
    if os.path.exists(stats_file):
        os.remove(stats_file)
        print(f"Removed {stats_file}")
    
    if os.path.exists(games_csv_file):
        os.remove(games_csv_file)
        print(f"Removed {games_csv_file}")

def run_benchmark_iteration(model_configs: List[ModelConfig], 
                          stats_file: str, 
                          device: str, 
                          fens: List[str],
                          start_time: float,
                          time_limit: int,
                          use_fen_game: bool,
                          max_moves: int) -> bool:
    """Run one iteration of benchmarking. Returns True if should continue."""
    
    # Load networks
    nets = {}
    for config_obj in model_configs:
        nets[config_obj.name] = load_net(
            config_obj.path, device,
            config_obj.in_channels,
            config_obj.num_res_blocks,
            config_obj.num_filters
        )

    stats = load_stats(model_configs, stats_file)
    games_to_log = []
    
    # Create all possible matchups and randomize them
    all_matchups = []
    for config_a, config_b in itertools.combinations(model_configs, 2):
        # Add both color combinations
        all_matchups.append((config_a, config_b, 'normal'))  # config_a plays white
        all_matchups.append((config_b, config_a, 'swapped'))  # config_b plays white
    
    # Randomize the matchups
    random.shuffle(all_matchups)
    
    game_count = 0
    for config_w, config_b_game, color_variant in all_matchups:
        # Check time limit
        if time.time() - start_time > time_limit:
            print(f"Time limit of {time_limit} seconds reached. Stopping.")
            return False
        
        net_w, net_b = nets[config_w.name], nets[config_b_game.name]
        
        fen = random.choice(fens) if fens else None
        record_game = use_fen_game and game_count < 10
        
        print(f"Game {game_count + 1}: {config_w.name} (W) vs {config_b_game.name} (B) - {color_variant}")
        if config_w.pure_network:
            print(f"  {config_w.name} using Pure Network")
        if config_b_game.pure_network:
            print(f"  {config_b_game.name} using Pure Network")
        
        score_w, score_b_game, pgn = play_game(
            config_w, config_b_game, net_w, net_b, device, 
            fen=fen, record_game=record_game, max_moves=max_moves
        )
        
        if record_game:
            games_to_log.append({
                'model_a': config_w.name,
                'model_b': config_b_game.name,
                'game_idx': game_count,
                'color_variant': color_variant,
                'start_fen': fen or "startpos",
                'pgn': pgn,
                'config_a_pure': config_w.pure_network,
                'config_b_pure': config_b_game.pure_network
            })

        # Update stats
        if score_w == 1.0:
            stats[config_w.name]['wins'] += 1
            stats[config_b_game.name]['losses'] += 1
        elif score_w == 0.0:
            stats[config_w.name]['losses'] += 1
            stats[config_b_game.name]['wins'] += 1
        else:
            stats[config_w.name]['draws'] += 1
            stats[config_b_game.name]['draws'] += 1

        # Determine original config names for ELO updates
        config_a_name = config_w.name if color_variant == 'normal' else config_b_game.name
        config_b_name = config_b_game.name if color_variant == 'normal' else config_w.name
        
        stats[config_a_name]['games'] += 1
        stats[config_b_name]['games'] += 1
        
        # Update ELO ratings
        elo_a = stats[config_a_name]['elo']
        elo_b = stats[config_b_name]['elo']
        score_a = score_w if color_variant == 'normal' else score_b_game

        new_a, new_b = update_elo(elo_a, elo_b, score_a, 1 - score_a)
        stats[config_a_name]['elo'] = new_a
        stats[config_b_name]['elo'] = new_b

        print(f'  Result: {score_w}-{score_b_game}')
        print(f'  Current ELOs: {config_a_name}={stats[config_a_name]["elo"]:.1f}, '
              f'{config_b_name}={stats[config_b_name]["elo"]:.1f}')
        
        game_count += 1

    if use_fen_game and games_to_log:
        with open('games_different.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'model_a', 'model_b', 'game_idx', 'color_variant', 'start_fen', 'pgn',
                'config_a_pure', 'config_b_pure'
            ])
            if f.tell() == 0:  # Write header if file is empty
                writer.writeheader()
            writer.writerows(games_to_log)
        print(f"Logged {len(games_to_log)} games to games_different.csv")

    save_stats(stats, stats_file)
    return True

def main():
    parser = argparse.ArgumentParser(description='Advanced ELO benchmarking with JSON configs from CLI')
    parser.add_argument('--time_limit', type=int, default=DEFAULT_TIME_LIMIT,
                        help='Time limit in seconds (default: 41400 = 11.5 hours)')
    parser.add_argument('--stats_file', default=DEFAULT_STATS_FILE,
                        help='Path to save updated elo statistics')
    parser.add_argument('--config_json', type=str, default=None,
                        help='JSON string with model configurations')
    parser.add_argument('--create_config', action='store_true',
                        help='Create default config JSON and exit')
    parser.add_argument('--use_fen_game', action='store_true',
                        help='Use FEN games and save game data to csv (starts from standard position if no FEN file provided)')
    parser.add_argument('--fen_file', type=str, default=None,
                        help='Path to a file with one FEN per line (optional, defaults to standard starting position)')
    parser.add_argument('--max_moves', type=int, default=200,
                        help='Maximum plies per benchmark game')
    parser.add_argument('--reset_iterations', type=int, default=0,
                        help='Number of iterations between environment resets (0 = no reset)')
    # MCTS hyperparameter flags
    parser.add_argument('--num_mcts_sims', type=int, default=config.NUM_MCTS_SIMS,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--cpuct', type=float, default=config.CPUCT,
                        help='PUCT exploration constant')
    parser.add_argument('--batch_mcts_size', type=int, default=config.BATCH_MCTS_SIZE,
                        help='Batch size for batched inference in MCTS')
    parser.add_argument('--root_noise_eps', type=float, default=config.ROOT_NOISE_EPS,
                        help='Fraction of Dirichlet noise to mix at root')
    parser.add_argument('--dirichlet_alpha', type=float, default=config.DIRICHLET_ALPHA,
                        help='Dirichlet alpha for root noise')
    parser.add_argument('--virtual_loss', type=float, default=config.VIRTUAL_LOSS,
                        help='Virtual loss penalty for tree-parallel MCTS')
    parser.add_argument('checkpoints', nargs='+', help='Model checkpoint paths')
    args = parser.parse_args()

    # Override MCTS defaults in config module from CLI flags
    config.NUM_MCTS_SIMS = args.num_mcts_sims
    config.CPUCT = args.cpuct
    config.BATCH_MCTS_SIZE = args.batch_mcts_size
    config.ROOT_NOISE_EPS = args.root_noise_eps
    config.DIRICHLET_ALPHA = args.dirichlet_alpha
    config.VIRTUAL_LOSS = args.virtual_loss

    # Create default config JSON if requested
    if args.create_config:
        config_json = create_default_config_string(args.checkpoints)
        print("Default configuration JSON:")
        print(config_json)
        return

    # Load model configurations
    if args.config_json:
        model_configs = load_model_configs_from_json(args.config_json)
        # Filter to only include models in checkpoints
        model_configs = [c for c in model_configs if c.path in args.checkpoints]
    else:
        # Create default configs for all models
        model_configs = []
        for model_path in args.checkpoints:
            model_configs.append(ModelConfig(
                path=model_path,
                name=Path(model_path).stem
            ))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fens = []
    if args.use_fen_game:
        if args.fen_file and os.path.exists(args.fen_file):
            with open(args.fen_file, 'r') as f:
                fens = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(fens)} FEN positions from {args.fen_file}")
        else:
            # Default to standard starting position if no FEN file provided
            fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
            print("Using standard chess starting position")

    start_time = time.time()
    iteration = 0
    
    print(f"Starting benchmark with {len(model_configs)} models")
    print(f"Time limit: {args.time_limit} seconds ({args.time_limit / 3600:.1f} hours)")
    print(f"Reset iterations: {args.reset_iterations}")
    print(f"Models: {[c.name for c in model_configs]}")
    
    while True:
        iteration += 1
        print(f"\n=== Iteration {iteration} ===")
        
        # Reset environment if needed
        if args.reset_iterations > 0 and iteration > 1 and (iteration - 1) % args.reset_iterations == 0:
            print("Resetting environment...")
            reset_environment(args.stats_file, 'games_different.csv')
        
        # Run benchmark iteration
        should_continue = run_benchmark_iteration(
            model_configs, args.stats_file, device, fens,
            start_time, args.time_limit, args.use_fen_game, args.max_moves
        )
        
        if not should_continue:
            break
        
        # Check if we should continue based on time
        elapsed_time = time.time() - start_time
        if elapsed_time >= args.time_limit:
            print(f"Time limit reached after {iteration} iterations.")
            break

    # Final results
    if os.path.exists(args.stats_file):
        stats = load_stats(model_configs, args.stats_file)
        
        print("\nFinal Results:")
        print("=" * 80)
        sorted_configs = sorted(model_configs, key=lambda x: stats[x.name]['elo'], reverse=True)
        
        for config_obj in sorted_configs:
            s = stats[config_obj.name]
            mode_str = ""
            if config_obj.pure_network:
                mode_str = " (Pure Network)"
            
            print(f'{config_obj.name}{mode_str}: '
                  f'elo={s["elo"]:.1f} games={s["games"]} '
                  f'wins={s["wins"]} draws={s["draws"]} losses={s["losses"]}')
            
            if not config_obj.pure_network:
                print(f'  MCTS: sims={config_obj.num_mcts_sims}, cpuct={config_obj.cpuct}, '
                      f'batch={config_obj.batch_mcts_size}')
        
        total_time = time.time() - start_time
        print(f"\nTotal time: {total_time:.1f} seconds ({total_time / 3600:.1f} hours)")
        print(f"Iterations completed: {iteration}")

if __name__ == '__main__':
    main() 