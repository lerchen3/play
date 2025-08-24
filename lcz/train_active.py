#!/usr/bin/env python3
"""
Active Inference MCTS Training Script

This script runs MCTS with active inference where the network is trained 
during the search process itself. The policy is trained on visit counts
and the value is trained on max Q(s,a) across all actions from each state.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # Import the C++ extension and chess engine
    import mcts_cpp_active  # The compiled C++ module
    import chess
    import chess.engine
    import numpy as np
    from typing import List, Tuple, Optional
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure to build the C++ extension first with build_cpp_active.sh")
    sys.exit(1)


class ActiveInferenceTrainer:
    """
    Trainer that uses MCTS with active inference.
    The network is trained during MCTS search, not separately.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 num_games: int = 1000,
                 games_per_checkpoint: int = 100,
                 checkpoint_dir: str = "checkpoints_active"):
        """
        Initialize the active inference trainer.
        
        Args:
            model_path: Path to load initial model from (optional)
            num_games: Total number of self-play games to run
            games_per_checkpoint: How often to save checkpoints
            checkpoint_dir: Directory to save checkpoints
        """
        self.num_games = num_games
        self.games_per_checkpoint = games_per_checkpoint
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize the C++ MCTS with active inference
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}")
            self.mcts = mcts_cpp_active.MCTSActive(model_path)
        else:
            print("Initializing with random model")
            self.mcts = mcts_cpp_active.MCTSActive()
        
        # Statistics
        self.games_played = 0
        self.total_moves = 0
        self.start_time = time.time()
    
    def play_self_play_game(self) -> Tuple[str, int]:
        """
        Play a single self-play game using active inference MCTS.
        The network trains automatically during each MCTS search.
        
        Returns:
            Tuple of (final_fen, game_result)
        """
        board = chess.Board()
        move_count = 0
        
        print(f"\nGame {self.games_played + 1}/{self.num_games}")
        
        while not board.is_game_over() and move_count < 200:  # Limit game length
            # Convert chess.Board to the format expected by C++
            fen = board.fen()
            
            # Run MCTS search with active inference (training happens automatically)
            policy = self.mcts.search(fen)
            
            # Convert policy to move probabilities
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
                
            # Sample move based on policy
            move_probs = []
            for move in legal_moves:
                move_uci = move.uci()
                # Get probability from policy (this requires mapping UCI to policy indices)
                prob = self.get_move_probability(move_uci, policy)
                move_probs.append(prob)
            
            # Normalize probabilities
            total_prob = sum(move_probs)
            if total_prob > 0:
                move_probs = [p / total_prob for p in move_probs]
            else:
                # Fallback to uniform distribution
                move_probs = [1.0 / len(legal_moves)] * len(legal_moves)
            
            # Sample move (with some randomness for exploration)
            if move_count < 10:  # More random in opening
                temperature = 1.0
            else:
                temperature = 0.1
                
            # Apply temperature and sample
            if temperature > 0:
                scaled_probs = np.array(move_probs) ** (1.0 / temperature)
                scaled_probs /= scaled_probs.sum()
                chosen_idx = np.random.choice(len(legal_moves), p=scaled_probs)
            else:
                chosen_idx = np.argmax(move_probs)
            
            chosen_move = legal_moves[chosen_idx]
            board.push(chosen_move)
            move_count += 1
            
            if move_count % 20 == 0:
                print(f"  Move {move_count}, Training buffer size: {self.mcts.get_training_buffer_size()}")
        
        # Determine game result
        result = board.result()
        if result == "1-0":
            game_result = 1  # White wins
        elif result == "0-1":
            game_result = -1  # Black wins
        else:
            game_result = 0  # Draw
        
        self.total_moves += move_count
        print(f"  Game finished: {result} after {move_count} moves")
        print(f"  Final training buffer size: {self.mcts.get_training_buffer_size()}")
        
        return board.fen(), game_result
    
    def get_move_probability(self, move_uci: str, policy: List[float]) -> float:
        """
        Get the probability for a specific move from the policy vector.
        This is a simplified version - in practice you'd need the exact 
        move encoding used by the C++ code.
        """
        # This is a placeholder implementation
        # In practice, you'd need to match the move encoding in mcts_active_inference.cpp
        move_hash = hash(move_uci) % len(policy)
        return policy[move_hash]
    
    def save_checkpoint(self, game_num: int):
        """Save a model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"model_game_{game_num}.pt"
        try:
            self.mcts.save_model(str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
    
    def print_stats(self):
        """Print training statistics."""
        elapsed = time.time() - self.start_time
        games_per_hour = self.games_played / (elapsed / 3600) if elapsed > 0 else 0
        moves_per_game = self.total_moves / self.games_played if self.games_played > 0 else 0
        
        print(f"\n--- Training Statistics ---")
        print(f"Games played: {self.games_played}")
        print(f"Total moves: {self.total_moves}")
        print(f"Average moves per game: {moves_per_game:.1f}")
        print(f"Elapsed time: {elapsed:.1f}s")
        print(f"Games per hour: {games_per_hour:.1f}")
        print(f"Current training buffer size: {self.mcts.get_training_buffer_size()}")
    
    def train(self):
        """
        Main training loop.
        Plays self-play games where MCTS trains the network automatically.
        """
        print("Starting Active Inference MCTS Training")
        print(f"Target games: {self.num_games}")
        print(f"Checkpoint frequency: {self.games_per_checkpoint}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        
        try:
            for game_num in range(self.num_games):
                # Play one self-play game (training happens automatically during MCTS)
                final_fen, result = self.play_self_play_game()
                self.games_played += 1
                
                # Save checkpoint periodically
                if (game_num + 1) % self.games_per_checkpoint == 0:
                    self.save_checkpoint(game_num + 1)
                    self.print_stats()
                
                # Clear training buffer periodically to prevent memory issues
                if (game_num + 1) % (self.games_per_checkpoint // 2) == 0:
                    print("Clearing training buffer to free memory")
                    self.mcts.clear_training_buffer()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            raise
        finally:
            # Final checkpoint and stats
            if self.games_played > 0:
                self.save_checkpoint(self.games_played)
                self.print_stats()
            
            print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description="Train chess engine with Active Inference MCTS")
    parser.add_argument("--model", type=str, help="Path to initial model file")
    parser.add_argument("--games", type=int, default=1000, help="Number of self-play games")
    parser.add_argument("--checkpoint-freq", type=int, default=100, 
                       help="Save checkpoint every N games")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_active",
                       help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = ActiveInferenceTrainer(
        model_path=args.model,
        num_games=args.games,
        games_per_checkpoint=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir
    )
    
    trainer.train()


if __name__ == "__main__":
    main() 