import torch
from tictactoe_interface import TicTacToeGame, MCTS
from tictactoe_net import TicTacToeNet

net = TicTacToeNet()
net.load_state_dict(torch.load('tictactoe_model.pt', map_location='cpu'))
net.eval()

mcts = MCTS(net)

game = TicTacToeGame()

symbols = {0: '.', 1: 'X', -1: 'O'}

while not game.is_game_over():
    board = game.state()[:9]
    print('Board:')
    for r in range(3):
        print(' '.join(board[r*3:(r+1)*3]))
    if game.turn() == 1:
        move = int(input('Enter move (0-8): '))
        if not game.move(move):
            print('Illegal move')
            continue
    else:
        policy = mcts.search(game)
        print(f'Policy: {policy}')
        move = int(policy.argmax())
        game.move(move)
        print(f'AI plays {move}')

res = game.result()
print('Result:', res)
