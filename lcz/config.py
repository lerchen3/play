import os
from move_encoding import MOVE_INDEX

# Neural network parameters
IN_CHANNELS = 103         # 8*12 piece planes + 7 constant feature planes
NUM_RES_BLOCKS = 40       # number of residual blocks used in AlphaZero
NUM_FILTERS = 256         # number of convolutional filters per layer

# Action space
ACTION_SIZE = len(MOVE_INDEX)

# MCTS parameters
NUM_MCTS_SIMS = 1600       # number of MCTS simulations per move
CPUCT = 1.0               # PUCT exploration constant

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
# number of self-play games generated per invocation
NUM_SELF_PLAY_GAMES = 1

# Directories
MODEL_DIR = os.path.join(os.getcwd(), 'models')
# always create the models folder if missing, but don't error if it already exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Iterative self-play and training settings
NUM_ITERATIONS = 1           # number of outer self-play/train loops
GAMES_PER_ITER = NUM_SELF_PLAY_GAMES  # games generated per iteration
EPOCHS_PER_ITER = 2           # training epochs per iteration
REPLAY_BUFFER_SIZE = 10000    # max number of self-play samples to retain
BATCH_MCTS_SIZE = 32          # batch size for batched inference in MCTS

# Optimizer and training improvements
WEIGHT_DECAY = 1e-4           # L2 regularization on weights
LR_STEP_SIZE = 5              # step size (in epochs) for LR scheduler
LR_GAMMA = 0.9                # multiplicative factor of LR decay
ENTROPY_BONUS = 0.01          # coefficient for policy entropy regularization

# AlphaZero root noise for exploration
DIRICHLET_ALPHA = 0.03       # Dirichlet alpha for root noise
ROOT_NOISE_EPS = 0.25        # fraction of Dirichlet noise to mix at root 
VIRTUAL_LOSS = 0.1         # virtual loss penalty for tree-parallel MCTS 
