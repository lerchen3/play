#pragma once

struct MCTSParams {
    int NUM_MCTS_SIMS;       // number of MCTS simulations per move
    double CPUCT;            // PUCT exploration constant
    int BATCH_MCTS_SIZE;     // batch size for batched inference in MCTS
    double ROOT_NOISE_EPS;   // fraction of Dirichlet noise to mix at root
    double DIRICHLET_ALPHA;  // Dirichlet alpha for root noise
    double VIRTUAL_LOSS;     // virtual loss penalty for tree-parallel MCTS
};

extern MCTSParams g_mcts_params;
