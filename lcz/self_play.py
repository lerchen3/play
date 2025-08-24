import numpy as np
import time
from cpp_interface import ChessGame, FenGame, MCTS
from move_encoding import MOVE_INDEX, INDEX_MOVE
from config import NUM_SELF_PLAY_GAMES

class SelfPlayWorker:
    def __init__(self, net, device='cpu', num_games=NUM_SELF_PLAY_GAMES, game_cls=FenGame, max_moves=200):
        self.net = net
        self.device = device
        self.num_games = num_games
        self.game_cls = game_cls
        self.max_moves = max_moves

    def self_play(self):
        data = []
        for game_idx in range(self.num_games):
            start = time.time()  # record start time
            print(f"SelfPlay: starting game {game_idx+1}/{self.num_games}")
            game = self.game_cls()
            states, pis, players = [], [], []
            mcts = MCTS(self.net, device=self.device)
            move_count = 0
            while not game.is_game_over() and move_count < self.max_moves:
                policy = mcts.search(game)
                states.append(game.get_state().numpy())
                pis.append(policy)
                players.append(game.current_player())
                # select move
                move_idx = np.random.choice(len(policy), p=policy)
                uci = INDEX_MOVE[move_idx]
                game.move(uci)
                move_count += 1
            if move_count >= self.max_moves and not game.is_game_over():
                result = 0
            else:
                result = game.result()
            for s, pi, player in zip(states, pis, players):
                z = result if player == 1 else -result
                data.append((s, pi, z))
            duration = time.time() - start
            print(f"SelfPlay: completed game {game_idx+1}/{self.num_games} in {duration:.2f}s")
        return data 
