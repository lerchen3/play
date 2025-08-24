Building Self-Play Reinforcement Learning Fluency.

- AlphaZero‑style chess engine with C++ MCTS in `mcts.*` and high‑performance legal move generation in `position.*`.
- Batched policy/value inference via Python callback C‑API (`c_api.*` + `cpp_interface.py`) into a PyTorch ResNet in `model.py`
- Tree‑parallel MCTS: virtual loss, Dirichlet root noise, batched simulations, robust FEN handling; ctypes bridge in `cpp_interface.py`.
- Distributed self‑play + training (`train.py`): PyTorch DDP, replay buffer, entropy regularization, step LR schedule, resumable checkpoints.
- Toy env: minimal TicTacToe pipeline in `tictactoe_toy/` for fast iteration and sanity checks.
