import torch
import numpy as np
import argparse
from model import LeelaChessZeroNet
from cpp_interface import ChessGame
from move_encoding import INDEX_MOVE, MOVE_INDEX
from config import MODEL_DIR

# Interactive play script using only the network's policy head (no MCTS)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=f"{MODEL_DIR}/model_latest.pt")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    net = LeelaChessZeroNet().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    net.load_state_dict(state_dict)
    net.eval()

    # Initialize game state
    game = ChessGame()

    # Play loop
    while not game.is_game_over():
        print(game.board_ascii())
        print(f"Turn: {'White' if game.current_player() == 1 else 'Black'}")
        
        # Human move
        move_input = input("Your move (UCI): ").strip()
        if not game.move(move_input):
            print("Illegal move.")
            continue
        if game.is_game_over():
            break
            
        # Network move - use consistent representation
        print(game.board_ascii())
        print("Network is thinking...")
        
        # Use game.get_state() which calls fen_history_to_state consistently
        state_tensor = game.get_state().unsqueeze(0).to(device)
        
        with torch.no_grad():
            log_probs, value = net(state_tensor)
        probs = torch.exp(log_probs).cpu().numpy().flatten()
        
        # Get legal moves and filter probabilities
        legal_moves = game.legal_moves()
        
        if not legal_moves:
            print("No legal moves found!")
            break
        best_idx = np.argmax(probs)
        best_move = INDEX_MOVE[best_idx]
        
        print(f"Legal moves: {legal_moves}")
        # Find the probability assigned to the best_move (if it's legal)
        if best_move in legal_moves:
            move_prob = probs[best_idx]
            print(f"Network plays {best_move} (prob={move_prob:.3f}, value={value.item():.3f})")
        else:
            print(f"Network suggests illegal move {best_move} (value={value.item():.3f})")
        
        if not game.move(best_move):
            print(f"ERROR: Network move {best_move} was illegal. Network learned illegal moves!")
            break

    # Final status
    print("Game over:", game.result())

if __name__ == '__main__':
    main() 