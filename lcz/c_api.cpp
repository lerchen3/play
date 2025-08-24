#include "c_api.h"
#include "mcts.h"
#include "position.h"
#include <string>
#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <iostream>

struct GameHandle {
    Position pos;
};
struct MCTSHandle {
    std::unique_ptr<Net> net;
    std::unique_ptr<MCTS> mcts;
};

// Python callback based Net implementation
class CallbackNet : public Net {
public:
    batch_predict_cb cb;
    CallbackNet(batch_predict_cb c): cb(c) {}
    void predict(const std::vector<Position>& states,
                 std::vector<std::vector<double>>& logps,
                 std::vector<double>& vs) override {
        auto total_start = std::chrono::high_resolution_clock::now();

        auto fen_prep_start = std::chrono::high_resolution_clock::now();
        std::string fen_batch_str;
        for (const auto& state : states) {
            std::string group = state.fen();
            int cnt = 0;
            for(auto it = state.history_fens.rbegin(); it != state.history_fens.rend() && cnt < 7; ++it, ++cnt){
                group += "|" + *it;
            }
            fen_batch_str += group + "\n";
        }
        auto fen_prep_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> fen_prep_took = fen_prep_end - fen_prep_start;

        int batch_size = states.size();
        std::vector<float> policy_results(batch_size * INDEX_MOVE.size());
        std::vector<float> value_results(batch_size);

        auto cb_start = std::chrono::high_resolution_clock::now();
        cb(fen_batch_str.c_str(), batch_size, policy_results.data(), value_results.data());
        auto cb_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cb_took = cb_end - cb_start;

        auto demarshal_start = std::chrono::high_resolution_clock::now();
        logps.resize(batch_size);
        vs.resize(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            vs[i] = value_results[i];
            auto p_start = policy_results.begin() + i * INDEX_MOVE.size();
            auto p_end = p_start + INDEX_MOVE.size();
            logps[i].assign(p_start, p_end);
        }
        auto demarshal_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> demarshal_took = demarshal_end - demarshal_start;
        
        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_took = total_end - total_start;
    }
};

extern "C" {
GameHandle* game_create(){return new GameHandle();}
void game_destroy(GameHandle* g){delete g;}
void game_reset(GameHandle* g){
    // Reset the position to the standard starting board
    g->pos.clear();
    Position::set("startpos", g->pos);
}
void game_set_fen(GameHandle* g,const char* fen){Position::set(fen,g->pos);}
const char* game_get_fen(GameHandle* g) {
    // Use a fixed-size static buffer to return FEN, avoiding malloc/free
    static char fen_buf[256];
    std::string tmp = g->pos.fen();
    size_t len = tmp.size();
    if (len >= sizeof(fen_buf)) len = sizeof(fen_buf) - 1;
    std::memcpy(fen_buf, tmp.c_str(), len);
    fen_buf[len] = '\0';
    return fen_buf;
}

const char* game_get_past_fen(GameHandle* g, int idx) {
    static char fen_buf[256];
    std::string tmp;
    if(idx == 0) {
        tmp = g->pos.fen();
    } else if(idx <= (int)g->pos.history_fens.size()) {
        tmp = g->pos.history_fens[g->pos.history_fens.size() - idx];
    } else {
        tmp = "";
    }
    size_t len = tmp.size();
    if (len >= sizeof(fen_buf)) len = sizeof(fen_buf) - 1;
    std::memcpy(fen_buf, tmp.c_str(), len);
    fen_buf[len] = '\0';
    return fen_buf;
}
int game_turn(GameHandle* g){return g->pos.turn()==WHITE?1:-1;}
int game_move(GameHandle* g,const char* uci){
    std::string s(uci);
    bool legal = false;
    if(g->pos.turn()==WHITE){
        MoveList<WHITE> ml(g->pos);
        for(auto m: ml){
            if(m.uci() == s){
                legal = true;
                break;
            }
        }
    } else {
        MoveList<BLACK> ml(g->pos);
        for(auto m: ml){
            if(m.uci() == s){
                legal = true;
                break;
            }
        }
    }
    if(!legal) return 0;
    Move m = uciToMove(s, g->pos);
    g->pos.play(m);
    return 1;
}
int game_legal_moves(GameHandle* g,char* buf,size_t buf_size){
    std::vector<std::string> moves;
    if(g->pos.turn()==WHITE){
        MoveList<WHITE> ml(g->pos);
        for(auto m: ml) moves.push_back(m.uci());
    } else {
        MoveList<BLACK> ml(g->pos);
        for(auto m: ml) moves.push_back(m.uci());
    }
    std::string s;
    for(size_t i=0;i<moves.size();++i){
        s += moves[i];
        if(i+1<moves.size()) s.push_back(' ');
    }
    if(s.size()+1 > buf_size) return -1;
    std::memcpy(buf, s.c_str(), s.size()+1);
    return (int)moves.size();
}
int game_is_game_over(GameHandle* g){return g->pos.isGameOver();}
int game_result(GameHandle* g){return g->pos.result();}

// FEN-only helper implementations
int fen_turn(const char* fen){
    Position p; Position::set(fen, p);
    return p.turn()==WHITE?1:-1;
}

int fen_legal_moves(const char* fen, char* buf, size_t buf_size){
    Position p; Position::set(fen, p);
    std::vector<std::string> moves;
    if(p.turn()==WHITE){
        MoveList<WHITE> ml(p);
        for(auto m: ml) moves.push_back(m.uci());
    }else{
        MoveList<BLACK> ml(p);
        for(auto m: ml) moves.push_back(m.uci());
    }
    std::string s;
    for(size_t i=0;i<moves.size();++i){
        s += moves[i];
        if(i+1<moves.size()) s.push_back(' ');
    }
    if(s.size()+1 > buf_size) return -1;
    std::memcpy(buf, s.c_str(), s.size()+1);
    return (int)moves.size();
}

int fen_is_game_over(const char* fen){
    Position p; Position::set(fen, p);
    return p.isGameOver();
}

int fen_result(const char* fen){
    Position p; Position::set(fen, p);
    return p.result();
}

int fen_move(const char* fen, const char* uci, char* out_fen, size_t buf_size){
    Position p; Position::set(fen, p);
    std::string s(uci);
    bool legal = false;
    if(p.turn()==WHITE){
        MoveList<WHITE> ml(p);
        for(auto m: ml){
            if(m.uci()==s){ legal=true; break; }
        }
    }else{
        MoveList<BLACK> ml(p);
        for(auto m: ml){
            if(m.uci()==s){ legal=true; break; }
        }
    }
    if(!legal) return 0;
    Move m = uciToMove(s, p);
    p.play(m);
    std::string out = p.fen();
    if(out.size()+1 > buf_size) return -1;
    std::memcpy(out_fen, out.c_str(), out.size()+1);
    return 1;
}

MCTSHandle* mcts_create(batch_predict_cb cb){
    auto h = new MCTSHandle();
    h->net = std::unique_ptr<Net>(new CallbackNet(cb));
    h->mcts = std::unique_ptr<MCTS>(new MCTS(h->net.get()));
    return h;
}
void mcts_destroy(MCTSHandle* h){delete h;}
void mcts_search(MCTSHandle* h,const char* fen,float* out_policy){
    Position root; Position::set(fen, root);
    auto pol = h->mcts->search(root);
    for(size_t i=0;i<pol.size();++i) out_policy[i] = (float)pol[i];
}

void mcts_set_params(int num_sims, double cpuct, int batch_size,
                     double root_noise_eps, double dirichlet_alpha,
                     double virtual_loss) {
    g_mcts_params.NUM_MCTS_SIMS = num_sims;
    g_mcts_params.CPUCT = cpuct;
    g_mcts_params.BATCH_MCTS_SIZE = batch_size;
    g_mcts_params.ROOT_NOISE_EPS = root_noise_eps;
    g_mcts_params.DIRICHLET_ALPHA = dirichlet_alpha;
    g_mcts_params.VIRTUAL_LOSS = virtual_loss;
}
}
