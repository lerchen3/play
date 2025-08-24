#!/usr/bin/env python
import argparse
import itertools
import json
import os
import config
import random
import csv
import numpy as np
import torch

from model import LeelaChessZeroNet
from cpp_interface import ChessGame, MCTS, set_mcts_params
from move_encoding import INDEX_MOVE

DEFAULT_STATS_FILE = 'elo_stats.json'
INITIAL_ELO = 1200
K_FACTOR = 32


def load_net(path, device, in_channels, num_res_blocks, num_filters):
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


def play_game(net_w, net_b, device, fen=None, record_game=False, max_moves=200):
    game = ChessGame(fen=fen) if fen else ChessGame()
    mcts_w = MCTS(net_w, device=device)
    mcts_b = MCTS(net_b, device=device)
    
    pgn_moves = []
    move_count = 0

    while not game.is_game_over() and move_count < max_moves:
        if game.current_player() == 1:
            mcts = mcts_w
        else:
            mcts = mcts_b
        policy = mcts.search(game)
        move_idx = np.random.choice(len(policy), p=policy)
        uci = INDEX_MOVE[move_idx]
        
        if record_game:
            pgn_moves.append(game.san(uci))

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


def load_stats(paths, stats_file):
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = {}
    for p in paths:
        stats.setdefault(p, {
            'elo': INITIAL_ELO,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'games': 0
        })
    return stats


def save_stats(stats, stats_file):
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Benchmark ELO ratings')
    parser.add_argument('--num_games', type=int, default=2,
                        help='games per pair (played twice, colors swapped)')
    parser.add_argument('--stats_file', default=DEFAULT_STATS_FILE,
                        help='path to save updated elo statistics')
    parser.add_argument('--old_stats_pth', default=None,
                        help='optional existing stats file to load')
    parser.add_argument('--use_fen_game', action='store_true',
                        help='Use FEN games and save game data to csv.')
    parser.add_argument('--fen_file', type=str, default=None,
                        help='Path to a file with one FEN per line.')
    parser.add_argument('--max_moves', type=int, default=200,
                        help='Maximum plies per benchmark game')
    parser.add_argument('--in_channels', type=int, default=config.IN_CHANNELS,
                        help='Number of input feature channels')
    parser.add_argument('--num_res_blocks', type=int, default=config.NUM_RES_BLOCKS,
                        help='Number of residual blocks')
    parser.add_argument('--num_filters', type=int, default=config.NUM_FILTERS,
                        help='Number of convolutional filters')
    parser.add_argument('--num_mcts_sims', type=int, default=config.NUM_MCTS_SIMS)
    parser.add_argument('--cpuct', type=float, default=config.CPUCT)
    parser.add_argument('--batch_mcts_size', type=int, default=config.BATCH_MCTS_SIZE)
    parser.add_argument('--root_noise_eps', type=float, default=config.ROOT_NOISE_EPS)
    parser.add_argument('--dirichlet_alpha', type=float, default=config.DIRICHLET_ALPHA)
    parser.add_argument('--virtual_loss', type=float, default=config.VIRTUAL_LOSS)
    parser.add_argument('checkpoints', nargs='+', help='model checkpoint paths')
    args = parser.parse_args()

    # Keep config consistent with CLI overrides for correct state encoding
    config.IN_CHANNELS = args.in_channels
    config.NUM_RES_BLOCKS = args.num_res_blocks
    config.NUM_FILTERS = args.num_filters

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stats_path = args.old_stats_pth if args.old_stats_pth else args.stats_file
    stats = load_stats(args.checkpoints, stats_path)
    nets = {
        p: load_net(p, device, args.in_channels, args.num_res_blocks, args.num_filters)
        for p in args.checkpoints
    }
    set_mcts_params(args.num_mcts_sims, args.cpuct, args.batch_mcts_size,
                    args.root_noise_eps, args.dirichlet_alpha, args.virtual_loss)

    fens = []
    if args.use_fen_game:
        if not args.fen_file or not os.path.exists(args.fen_file):
            raise ValueError("FEN file must be provided and exist when using --use_fen_game")
        with open(args.fen_file, 'r') as f:
            fens = [line.strip() for line in f if line.strip()]

    games_to_log = []

    for a, b in itertools.combinations(args.checkpoints, 2):
        for i in range(args.num_games):
            net_w, net_b = (nets[a], nets[b]) if i % 2 == 0 else (nets[b], nets[a])
            model_w, model_b = (a, b) if i % 2 == 0 else (b, a)
            
            fen = random.choice(fens) if fens else None
            record_game = args.use_fen_game and i < 10
            
            score_w, score_b, pgn = play_game(net_w, net_b, device, fen=fen, record_game=record_game, max_moves=args.max_moves)
            
            if record_game:
                games_to_log.append({
                    'model_a': model_w,
                    'model_b': model_b,
                    'game_idx': i,
                    'start_fen': fen or "startpos",
                    'pgn': pgn
                })

            # Update stats based on white's score
            if score_w == 1.0:
                stats[model_w]['wins'] += 1
                stats[model_b]['losses'] += 1
            elif score_w == 0.0:
                stats[model_w]['losses'] += 1
                stats[model_b]['wins'] += 1
            else:
                stats[model_w]['draws'] += 1
                stats[model_b]['draws'] += 1

            stats[a]['games'] += 1
            stats[b]['games'] += 1
            
            elo_a, elo_b = (stats[a]['elo'], stats[b]['elo'])
            score_a = score_w if model_w == a else score_b

            new_a, new_b = update_elo(elo_a, elo_b, score_a, 1 - score_a)
            stats[a]['elo'] = new_a
            stats[b]['elo'] = new_b

            print(f'{os.path.basename(a)} vs {os.path.basename(b)} game {i+1}: '
                  f'score {score_w}-{score_b}')
        print(f'Current ELOs: {os.path.basename(a)}={stats[a]["elo"]:.1f}, {os.path.basename(b)}={stats[b]["elo"]:.1f}')

    if args.use_fen_game:
        with open('games.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model_a', 'model_b', 'game_idx', 'start_fen', 'pgn'])
            writer.writeheader()
            writer.writerows(games_to_log)
        print("Logged games to games.csv")

    save_stats(stats, args.stats_file)

    for p in args.checkpoints:
        s = stats[p]
        print(f'{p}: elo={s["elo"]:.1f} games={s["games"]} '
              f'wins={s["wins"]} draws={s["draws"]} losses={s["losses"]}')


if __name__ == '__main__':
    main()
