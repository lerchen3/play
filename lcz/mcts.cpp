#include "mcts.h"
#include <cmath>
#include <thread>
#include <future>
#include <chrono>
#include <iostream>
#include <random>
#include <algorithm>
#include <set>
#include <numeric>
#include <cstdlib>


// Global move encoding tables
// Mapping between UCI move strings and continuous indices
std::map<std::string, int> MOVE_INDEX;
std::vector<std::string> INDEX_MOVE;

// Initialize move encoding mapping (64x64 moves + 4 promotions each)
static std::once_flag init_flag;
void initMoveEncoding() {
    std::call_once(init_flag, [](){
        std::vector<std::string> square_names;
        square_names.reserve(64);
        for (char file = 'a'; file <= 'h'; ++file) {
            for (char rank = '1'; rank <= '8'; ++rank) {
                square_names.push_back(std::string() + file + rank);
            }
        }
        int index = 0;
        for (const auto& from : square_names) {
            for (const auto& to : square_names) {
                std::string uci = from + to;
                MOVE_INDEX[uci] = index;
                INDEX_MOVE.push_back(uci);
                ++index;
                
                // Add promotions for ALL moves (even impossible ones) for consistency with existing models
                for (char promo : {'q', 'r', 'b', 'n'}) {
                    std::string uci_p = uci + promo;
                    MOVE_INDEX[uci_p] = index;
                    INDEX_MOVE.push_back(uci_p);
                    ++index;
                }
            }
        }
    });
}

// Convert an internal Move to standard UCI notation
std::string moveToUCI(const Move& m) {
    return m.uci();
}

Move uciToMove(const std::string& uci, const Position& pos) {
    if (uci.size() < 4) return Move();

    // Generate all legal moves from the position and find a match
    if (pos.turn() == WHITE) {
        MoveList<WHITE> ml(const_cast<Position&>(pos));
        for (auto m : ml) if (moveToUCI(m) == uci) return m;
    } else {
        MoveList<BLACK> ml(const_cast<Position&>(pos));
        for (auto m : ml) if (moveToUCI(m) == uci) return m;
    }

    // Fallback to a quiet move construction if not found
    int from = (uci[0] - 'a') + (uci[1] - '1') * 8;
    int to   = (uci[2] - 'a') + (uci[3] - '1') * 8;
    return Move(from, to, QUIET);
}

// Result of a single MCTS simulation
struct SimulationResult {
    bool terminal;
    MCTSNode* leaf;
    std::vector<MCTSNode*> path;
};



// MCTSNode implementations
MCTSNode::MCTSNode(const Position& gameState, MCTSNode* parent_, double prior_)
    : game(gameState), parent(parent_), prior(prior_), visit_count(0), value_sum(0.0) {}

// Destructor to recursively clean up child nodes
MCTSNode::~MCTSNode() {
    for (auto &kv : children) delete kv.second;
}

bool MCTSNode::isExpanded() const {
    return !children.empty();
}

double MCTSNode::value() const {
    return visit_count > 0 ? (value_sum / visit_count) : 0.0;
}

// MCTS implementations
MCTS::MCTS(Net* net_) : net(net_) {}

std::vector<double> MCTS::search(const Position& root_game) {
    initMoveEncoding();
    const char* rank_env = std::getenv("RANK");
    int rank = rank_env ? std::atoi(rank_env) : 0;
    const char* ws_env = std::getenv("WORLD_SIZE");
    int world_size = ws_env ? std::atoi(ws_env) : 1;

    // 1) Initial expansion at root
    MCTSNode* root = new MCTSNode(root_game);
    // Initial inference
    std::vector<Position> root_states{root_game};
    std::vector<std::vector<double>> logp_b;
    std::vector<double> v_b;
    net->predict(root_states, logp_b, v_b);
    const auto& log_p = logp_b[0];
    double v = v_b[0];
    root->visit_count = 1;
    root->value_sum = v;

    // Compute full priors
    std::vector<double> priors_full(INDEX_MOVE.size());
    for (size_t i = 0; i < INDEX_MOVE.size(); ++i) {
        priors_full[i] = std::exp(log_p[i]);
    }
    // Legal moves and their indices
    std::vector<Move> legal_moves;
    std::vector<int> legal_indices;
    if (root_game.turn() == WHITE) {
        MoveList<WHITE> ml(root->game);
        for (auto m : ml) {
            std::string uci = moveToUCI(m);
            if (m.from() < 0 || m.from() >= 64 || m.to() < 0 || m.to() >= 64) {
                std::cerr << "WARNING: Filtering invalid move from=" << m.from() << " to=" << m.to() << " UCI=" << uci << std::endl;
                continue;
            }
            
            legal_moves.push_back(m);
            if (MOVE_INDEX.find(uci) != MOVE_INDEX.end()) {
                legal_indices.push_back(MOVE_INDEX[uci]);
            } else {
                std::cerr << "WARNING: Root legal move " << uci << " not found in MOVE_INDEX!" << std::endl;
            }
        }
    } else {
        MoveList<BLACK> ml(root->game);
        for (auto m : ml) {
            std::string uci = moveToUCI(m);
            if (m.from() < 0 || m.from() >= 64 || m.to() < 0 || m.to() >= 64) {
                std::cerr << "WARNING: Filtering invalid move from=" << m.from() << " to=" << m.to() << " UCI=" << uci << std::endl;
                continue;
            }
            
            legal_moves.push_back(m);
            if (MOVE_INDEX.find(uci) != MOVE_INDEX.end()) {
                legal_indices.push_back(MOVE_INDEX[uci]);
            } else {
                std::cerr << "WARNING: Root legal move " << uci << " not found in MOVE_INDEX!" << std::endl;
            }
        }
    }
    // Extract and normalize priors for legal moves
    std::vector<double> priors;
    priors.reserve(legal_indices.size());
    for (int idx : legal_indices) priors.push_back(priors_full[idx]);
    double total0 = std::accumulate(priors.begin(), priors.end(), 0.0);
    if (total0 > 0) for (auto& p : priors) p /= total0;
    
    if (legal_moves.size() != priors.size()) {
        std::cerr << "ERROR: Mismatch between legal_moves (" << legal_moves.size() 
                  << ") and priors (" << priors.size() << ")" << std::endl;
        legal_moves.resize(priors.size());
    }

    // Add Dirichlet noise
    if (!priors.empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::gamma_distribution<> gamma(g_mcts_params.DIRICHLET_ALPHA, 1.0);
        std::vector<double> noise(priors.size());
        double noise_sum = 0;
        for (double& n : noise) { n = gamma(gen); noise_sum += n; }
        for (double& n : noise) n /= noise_sum;
        for (size_t i = 0; i < priors.size(); ++i) {
            priors[i] = (1 - g_mcts_params.ROOT_NOISE_EPS) * priors[i] + g_mcts_params.ROOT_NOISE_EPS * noise[i];
        }
        double total1 = std::accumulate(priors.begin(), priors.end(), 0.0);
        if (total1 > 0) for (auto& p : priors) p /= total1;
    }

    // Create root children
    for (size_t i = 0; i < legal_moves.size(); ++i) {
        Position ng = root_game;
        ng.play(legal_moves[i]);
        root->children[moveToUCI(legal_moves[i])] = new MCTSNode(ng, root, priors[i]);
    }

    // 2) Batched MCTS simulations
    auto simulateOnce = [&](int) {
        MCTSNode* node = root;
        std::vector<MCTSNode*> path{root};
        while (true) {
            std::unique_lock<std::mutex> nl(node->lock);
            if (node->children.empty()) {
                nl.unlock();
                break;
            }
            double sqrt_visits = std::sqrt(node->visit_count);
            MCTSNode* best_child = nullptr;
            double best_ucb = -std::numeric_limits<double>::infinity();
            for (auto& kv : node->children) {
                MCTSNode* child = kv.second;
                double child_value;
                int child_visits;
                {
                    std::lock_guard<std::mutex> cl(child->lock);
                    child_visits = child->visit_count;
                    child_value = child_visits > 0 ? (child->value_sum / child_visits) : 0.0;
                }
                double exploit = -child_value;
                double explore = g_mcts_params.CPUCT * child->prior * sqrt_visits / (1 + child_visits);
                double ucb = exploit + explore;
                if (ucb > best_ucb) {
                    best_ucb = ucb;
                    best_child = child;
                }
            }
            nl.unlock();
            {
                std::lock_guard<std::mutex> lg(best_child->lock);
                best_child->value_sum -= g_mcts_params.VIRTUAL_LOSS;
                best_child->visit_count += 1;
            }
            node = best_child;
            path.push_back(node);
        }
        bool term = node->game.isGameOver();
        return SimulationResult{term, node, path};
    };

    int num_done = 0;
    int batch_idx = 1;
    while (num_done < g_mcts_params.NUM_MCTS_SIMS) {
        auto t_start = std::chrono::high_resolution_clock::now();
        int batch_size = g_mcts_params.BATCH_MCTS_SIZE;
        std::vector<std::future<SimulationResult>> futures;
        for (int i = 0; i < batch_size; ++i) futures.push_back(
            std::async(std::launch::async, simulateOnce, i));
        std::vector<MCTSNode*> pending_nodes;
        std::vector<std::vector<MCTSNode*>> pending_paths;
        for (auto& fut : futures) {
            SimulationResult res = fut.get();
            if (res.terminal) {
                int res_val = res.leaf->game.result();
                double leaf_value;
                if (res_val == 0) {
                    leaf_value = 0.0;
                } else {
                    Color leaf_turn = res.leaf->game.turn();
                    if ((res_val == 1 && leaf_turn == BLACK) || (res_val == -1 && leaf_turn == WHITE)) {
                        leaf_value = -1.0;
                    } else {
                        throw std::runtime_error("Unexpected winner's perspective in terminal node: this shouldn't happen. (the person who lost should be the one who has no possible next moves)");
                    }
                }
                size_t path_len = res.path.size();
                for (size_t d = 0; d < path_len; ++d) {
                    MCTSNode* n = res.path[path_len - 1 - d];
                    std::lock_guard<std::mutex> lg(n->lock);
                    if (d < path_len - 1) {
                        n->value_sum += g_mcts_params.VIRTUAL_LOSS;
                        n->visit_count -= 1;
                    }
                    double node_value = (d % 2 == 0) ? leaf_value : -leaf_value;
                    n->value_sum += node_value;
                    n->visit_count++;
                }
            } else {
                pending_nodes.push_back(res.leaf);
                pending_paths.push_back(res.path);
            }
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        double sel_time = std::chrono::duration<double>(t_end - t_start).count();
        if (pending_nodes.empty()) {
            break;
        }
        std::vector<std::string> local_fens;
        for (auto* nd : pending_nodes) local_fens.push_back(nd->game.fen());
        std::set<std::string> seen;
        std::vector<std::string> unique_fens;
        for (auto& fen : local_fens) if (seen.insert(fen).second) unique_fens.push_back(fen);
        std::vector<std::string> shard(
            unique_fens.begin(),
            unique_fens.begin() + std::min<int>(unique_fens.size(), g_mcts_params.BATCH_MCTS_SIZE)
        );
        num_done += futures.size();
        std::vector<Position> states;
        for (auto& fen : shard) {
            Position g;
            Position::set(fen, g);
            states.push_back(g);
        }
        auto t_fp_start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<double>> logp_b2;
        std::vector<double> v_b2;
        net->predict(states, logp_b2, v_b2);
        auto t_fp_end = std::chrono::high_resolution_clock::now();
        double fp_time = std::chrono::duration<double>(t_fp_end - t_fp_start).count();
        std::map<std::string, std::pair<std::vector<double>, double>> global_res;
        for (size_t i = 0; i < shard.size(); ++i) {
            global_res[shard[i]] = {logp_b2[i], v_b2[i]};
        }
        auto t_ex_start = std::chrono::high_resolution_clock::now();
        struct NodeGroup {
            std::vector<MCTSNode*> nodes;
            std::vector<std::vector<MCTSNode*>> paths;
        };
        std::map<std::string, NodeGroup> fen_map;
        for (size_t i = 0; i < pending_nodes.size(); ++i) {
            auto* nd = pending_nodes[i];
            auto fen = nd->game.fen();
            fen_map[fen].nodes.push_back(nd);
            fen_map[fen].paths.push_back(pending_paths[i]);
        }
        std::vector<std::thread> threads;
        for (auto kv : fen_map) {
            threads.emplace_back([kv, &global_res]() {
                auto nodes = kv.second.nodes;
                auto& paths = kv.second.paths;
                auto it = global_res.find(kv.first);
                if (it == global_res.end()) {
                    for (auto& path : paths) {
                        size_t len = path.size();
                        for (size_t d = 0; d < len; ++d) {
                            MCTSNode* n = path[len - 1 - d];
                            std::lock_guard<std::mutex> lg(n->lock);
                            if (d < len - 1) {
                                n->value_sum += g_mcts_params.VIRTUAL_LOSS;
                                n->visit_count -= 1;
                            }
                        }
                    }
                    return;
                }
                const auto& pr = it->second;
                const auto& logp = pr.first;
                double vv = pr.second;
                Color leaf_turn = nodes.front()->game.turn();
                std::vector<Move> legals;
                std::vector<int> legal_inds;
                if (leaf_turn == WHITE) {
                    MoveList<WHITE> ml(nodes.front()->game);
                    for (auto m : ml) {
                        legals.push_back(m);
                        std::string uci = moveToUCI(m);
                        if (MOVE_INDEX.find(uci) != MOVE_INDEX.end()) {
                            legal_inds.push_back(MOVE_INDEX[uci]);
                        } else {
                            std::cerr << "WARNING: Legal move " << uci << " not found in MOVE_INDEX!" << std::endl;
                        }
                    }
                } else {
                    MoveList<BLACK> ml(nodes.front()->game);
                    for (auto m : ml) {
                        legals.push_back(m);
                        std::string uci = moveToUCI(m);
                        if (MOVE_INDEX.find(uci) != MOVE_INDEX.end()) {
                            legal_inds.push_back(MOVE_INDEX[uci]);
                        } else {
                            std::cerr << "WARNING: Legal move " << uci << " not found in MOVE_INDEX!" << std::endl;
                        }
                    }
                }
                std::vector<double> probs;
                probs.reserve(legal_inds.size());
                for (int idx : legal_inds) probs.push_back(std::exp(logp[idx]));
                double sum_p = std::accumulate(probs.begin(), probs.end(), 0.0);
                if (sum_p > 0) for (auto& p : probs) p /= sum_p;
                
                if (legals.size() != probs.size()) {
                    std::cerr << "ERROR: Mismatch between legals (" << legals.size() 
                              << ") and probs (" << probs.size() << ")" << std::endl;
                    legals.resize(probs.size());
                }
                for (auto* nd : nodes) {
                    std::lock_guard<std::mutex> lg(nd->lock);
                    for (size_t j = 0; j < legals.size(); ++j) {
                        Position ng = nd->game;
                        ng.play(legals[j]);
                        nd->children[moveToUCI(legals[j])] = new MCTSNode(ng, nd, probs[j]);
                    }
                }
                for (auto& path : paths) {
                    size_t len = path.size();
                    for (size_t d = 0; d < len; ++d) {
                        MCTSNode* n = path[len - 1 - d];
                        std::lock_guard<std::mutex> lg(n->lock);
                        if (d < len - 1) {
                            n->value_sum += g_mcts_params.VIRTUAL_LOSS;
                            n->visit_count -= 1;
                        }
                        double node_value = (d % 2 == 0) ? vv : -vv;
                        n->value_sum += node_value;
                        n->visit_count++;
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
        auto t_ex_end = std::chrono::high_resolution_clock::now();
        double ex_time = std::chrono::duration<double>(t_ex_end - t_ex_start).count();
        batch_idx++;
    }
    std::vector<double> policy(INDEX_MOVE.size(), 0.0);
    for (auto& kv : root->children) {
        if (MOVE_INDEX.find(kv.first) != MOVE_INDEX.end()) {
            policy[MOVE_INDEX[kv.first]] = kv.second->visit_count;
        } else {
            std::cerr << "WARNING: Child move " << kv.first << " not found in MOVE_INDEX!" << std::endl;
        }
    }
    double tot = std::accumulate(policy.begin(), policy.end(), 0.0);
    if (tot > 0) {
        for (auto& p : policy) p /= tot;
    } else {
        std::cerr << "ERROR: Total visit count is 0!" << std::endl;
    }
    delete root;
    return policy;
}
