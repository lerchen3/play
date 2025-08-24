#pragma once
#include <cstddef>
#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles
typedef struct GameHandle GameHandle;
typedef struct MCTSHandle MCTSHandle;

// Game functions
GameHandle* game_create();
void game_destroy(GameHandle* g);
void game_reset(GameHandle* g);
void game_set_fen(GameHandle* g, const char* fen);
const char* game_get_fen(GameHandle* g);
const char* game_get_past_fen(GameHandle* g, int idx);
int game_turn(GameHandle* g);
// Apply UCI move. Returns 1 on success, 0 if the move is illegal.
int game_move(GameHandle* g, const char* uci);
// Write space-separated legal moves to buf (size buf_size). Returns count.
int game_legal_moves(GameHandle* g, char* buf, size_t buf_size);
int game_is_game_over(GameHandle* g);
int game_result(GameHandle* g);

// FEN-only helpers
// Apply move to FEN string. Returns 1 if legal and writes new FEN to out_fen
int fen_move(const char* fen, const char* uci, char* out_fen, size_t buf_size);
// Write space-separated legal moves to buf from FEN. Returns count or -1 on error
int fen_legal_moves(const char* fen, char* buf, size_t buf_size);
int fen_is_game_over(const char* fen);
int fen_result(const char* fen);
int fen_turn(const char* fen);

// MCTS functions
// Batch predict callback: given newline-separated FENs, batch size, fill policy and value arrays
typedef void (*batch_predict_cb)(const char* fen_batch, int batch_size, float* policy_batch, float* value_batch);

MCTSHandle* mcts_create(batch_predict_cb cb);
void mcts_destroy(MCTSHandle* m);
void mcts_search(MCTSHandle* m, const char* fen, float* out_policy);
void mcts_set_params(int num_sims, double cpuct, int batch_size,
                     double root_noise_eps, double dirichlet_alpha,
                     double virtual_loss);

#ifdef __cplusplus
}
#endif
