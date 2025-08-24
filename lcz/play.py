import numpy as np
import torch
import config
from move_encoding import INDEX_MOVE
from model import LeelaChessZeroNet
from cpp_interface import ChessGame, MCTS, set_mcts_params

def play():
    import argparse
    parser = argparse.ArgumentParser(description='Play chess against the AI with configurable parameters')
    parser.add_argument('--model_path', type=str, default=f"{config.MODEL_DIR}/model_latest.pt",
                        help='Path to the model checkpoint')
    parser.add_argument('--max_moves', type=int, default=200, 
                        help='Maximum plies before declaring a draw')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: cuda, cpu, or auto')
    
    # Model architecture parameters
    parser.add_argument('--in_channels', type=int, default=config.IN_CHANNELS,
                        help='Number of input feature channels')
    parser.add_argument('--num_res_blocks', type=int, default=config.NUM_RES_BLOCKS,
                        help='Number of residual blocks')
    parser.add_argument('--num_filters', type=int, default=config.NUM_FILTERS,
                        help='Number of convolutional filters')
    
    # MCTS parameters
    parser.add_argument('--num_mcts_sims', type=int, default=config.NUM_MCTS_SIMS,
                        help='Number of MCTS simulations')
    parser.add_argument('--cpuct', type=float, default=config.CPUCT,
                        help='CPUCT exploration parameter')
    parser.add_argument('--batch_mcts_size', type=int, default=config.BATCH_MCTS_SIZE,
                        help='Batch size for MCTS')
    parser.add_argument('--root_noise_eps', type=float, default=config.ROOT_NOISE_EPS,
                        help='Root noise epsilon')
    parser.add_argument('--dirichlet_alpha', type=float, default=config.DIRICHLET_ALPHA,
                        help='Dirichlet noise alpha')
    parser.add_argument('--virtual_loss', type=float, default=config.VIRTUAL_LOSS,
                        help='Virtual loss parameter')
    
    # Display options
    parser.add_argument('--show_policy', action='store_true',
                        help='Show AI policy probabilities')
    parser.add_argument('--show_top_moves', type=int, default=5,
                        help='Number of top moves to display')
    parser.add_argument('--starting_fen', type=str, default=None,
                        help='Starting position in FEN notation')
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Loading model from {args.model_path}")
    print(f"Using device: {device}")
    print(f"MCTS simulations: {args.num_mcts_sims}")
    print(f"CPUCT: {args.cpuct}")
    
    # Load model with specified architecture
    net = LeelaChessZeroNet(
        in_channels=args.in_channels,
        num_res_blocks=args.num_res_blocks,
        num_filters=args.num_filters
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    net.load_state_dict(state_dict)
    net.eval()
    
    # Initialize game
    cg = ChessGame(fen=args.starting_fen) if args.starting_fen else ChessGame()
    mcts = MCTS(net, device=device)
    
    # Set MCTS parameters
    set_mcts_params(args.num_mcts_sims, args.cpuct, args.batch_mcts_size,
                    args.root_noise_eps, args.dirichlet_alpha, args.virtual_loss)

    move_count = 0
    print("\n" + "="*50)
    print("CHESS GAME STARTED")
    print("="*50)
    print("Commands: Enter UCI move (e.g. 'e2e4'), 'quit' to exit, 'help' for help")
    print(f"Using {args.num_mcts_sims} MCTS simulations per move")
    print("="*50 + "\n")
    
    while not cg.is_game_over() and move_count < args.max_moves:
        print(cg.board_ascii())
        print(f"\nMove {move_count // 2 + 1} ({'White' if cg.current_player() == 1 else 'Black'} to move)")
        
        # Human move
        move_input = input("Your move (UCI): ").strip().lower()
        
        if move_input == 'quit':
            print("Game ended by user.")
            break
        elif move_input == 'help':
            print("\nHelp:")
            print("- Enter moves in UCI format (e.g., 'e2e4', 'g1f3', 'e7e8q' for promotion)")
            print("- Type 'quit' to exit")
            print("- Type 'help' for this message")
            continue
        
        if not cg.move(move_input):
            print("Illegal move. Try again.")
            continue
        
        move_count += 1
        
        # Check game over after human move
        if cg.is_game_over():
            break
        
        # AI move
        print(cg.board_ascii())
        print(f"\nMove {move_count // 2 + 1} (AI thinking...)")
        
        policy = mcts.search(cg)
        
        # Find top moves
        if args.show_policy or args.show_top_moves > 0:
            move_probs = [(i, prob) for i, prob in enumerate(policy) if prob > 0]
            move_probs.sort(key=lambda x: x[1], reverse=True)
            
            if args.show_policy:
                print(f"\nFull policy (showing top {args.show_top_moves}):")
            else:
                print(f"\nTop {args.show_top_moves} moves considered:")
                
            for i, (move_idx, prob) in enumerate(move_probs[:args.show_top_moves]):
                move_uci = INDEX_MOVE[move_idx]
                print(f"  {i+1}. {move_uci}: {prob:.3f} ({prob*100:.1f}%)")
        
        best_idx = int(np.argmax(policy))
        best_uci = INDEX_MOVE[best_idx]
        
        print(f"\nAI plays: {best_uci} (confidence: {policy[best_idx]:.3f})")
        cg.move(best_uci)
        move_count += 1

    print("\n" + "="*50)
    if move_count >= args.max_moves and not cg.is_game_over():
        print("DRAW - Reached move limit")
    else:
        result = cg.result()
        if result == 1:
            print("WHITE WINS!")
        elif result == -1:
            print("BLACK WINS!")
        else:
            print("DRAW!")
    print("="*50)

if __name__ == '__main__':
    play() 