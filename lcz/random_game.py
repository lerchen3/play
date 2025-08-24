import random
from cpp_interface import ChessGame


MAX_RANDOM_MOVES = 2000


def random_game():
    g = ChessGame()
    move_count = 0
    while move_count < MAX_RANDOM_MOVES and not g.is_game_over():
        legal = g.legal_moves()
        if not legal:
            break
        m = random.choice(legal)
        g.move(m)
        move_count += 1
        print(g.board_ascii())
        print("\n\n")
    print("Result:", g.result(), "after", move_count, "moves")


if __name__ == "__main__":
    random_game()
