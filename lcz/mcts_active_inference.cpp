#include "mcts_active_inference.h"
#include "position.h"
#include "config.h"
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
#include <limits>

// Forward declarations from mcts.cpp
extern std::string moveToUCI(const Move& m);

// Global move encoding tables for active inference
std::map<std::string, int> MOVE_INDEX_ACTIVE;
std::vector<std::string> INDEX_MOVE_ACTIVE;

// Initialize move encoding mapping
static std::once_flag init_flag_active;
void initMoveEncodingActive() {
    std::call_once(init_flag_active, [](){
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
                MOVE_INDEX_ACTIVE[uci] = index;
                INDEX_MOVE_ACTIVE.push_back(uci);
                ++index;
                
                // Add promotions for ALL moves for consistency
                for (char promo : {'q', 'r', 'b', 'n'}) {
                    std::string uci_p = uci + promo;
                    MOVE_INDEX_ACTIVE[uci_p] = index;
                    INDEX_MOVE_ACTIVE.push_back(uci_p);
                    ++index;
                }
            }
        }
    });
}

std::string moveToUCIActive(const Move& m) {
    return moveToUCI(m);
}

Move uciToMoveActive(const std::string& uci, const Position& pos) {
    if (uci.size() < 4) return Move();

    if (pos.turn() == WHITE) {
        MoveList<WHITE> ml(const_cast<Position&>(pos));
        for (auto m : ml) if (moveToUCIActive(m) == uci) return m;
    } else {
        MoveList<BLACK> ml(const_cast<Position&>(pos));
        for (auto m : ml) if (moveToUCIActive(m) == uci) return m;
    }

    int from = (uci[0] - 'a') + (uci[1] - '1') * 8;
    int to   = (uci[2] - 'a') + (uci[3] - '1') * 8;
    return Move(from, to, QUIET);
}

// MCTSNodeActive implementations
MCTSNodeActive::MCTSNodeActive(const Position& gameState, MCTSNodeActive* parent_, double prior_)
    : MCTSNode(gameState, static_cast<MCTSNode*>(parent_), prior_) {}

MCTSNodeActive::~MCTSNodeActive() {
    // Destructor is handled by base class
}

double MCTSNodeActive::getQValue() const {
    std::lock_guard<std::mutex> lg(const_cast<MCTSNodeActive*>(this)->lock);
    return visit_count > 0 ? (value_sum / visit_count) : 0.0;
}

TrainingData MCTSNodeActive::collectTrainingData() const {
    TrainingData data;
    data.fen = game.fen();
    
    // Initialize policy target with zeros
    data.policy_target.resize(INDEX_MOVE_ACTIVE.size(), 0.0);
    
    // Fill in visit counts for legal moves
    double total_visits = 0.0;
    double max_q_value = -std::numeric_limits<double>::infinity();
    
    std::lock_guard<std::mutex> lg(const_cast<MCTSNodeActive*>(this)->lock);
    
    for (const auto& kv : children) {
        const std::string& move_uci = kv.first;
        MCTSNode* child = kv.second;
        
        // Get visit count for policy target
        int visits = 0;
        double q_value = 0.0;
        {
            std::lock_guard<std::mutex> cl(child->lock);
            visits = child->visit_count;
            q_value = visits > 0 ? (child->value_sum / visits) : 0.0;
        }
        
        // Update policy target
        auto it = MOVE_INDEX_ACTIVE.find(move_uci);
        if (it != MOVE_INDEX_ACTIVE.end()) {
            data.policy_target[it->second] = static_cast<double>(visits);
            total_visits += visits;
        }
        
        // Track max Q-value for value target
        max_q_value = std::max(max_q_value, q_value);
    }
    
    // Normalize policy target
    if (total_visits > 0) {
        for (double& p : data.policy_target) {
            p /= total_visits;
        }
    }
    
    // Set value target to max Q-value (from current player's perspective)
    data.value_target = max_q_value != -std::numeric_limits<double>::infinity() ? max_q_value : 0.0;
    
    return data;
}

// MCTSActive implementations
MCTSActive::MCTSActive(Net* net_) : net(dynamic_cast<NetActive*>(net_)) {
    if (!net) {
        throw std::invalid_argument("Net must be an instance of NetActive for active inference");
    }
}

void MCTSActive::trainNetwork() {
    std::lock_guard<std::mutex> lg(training_mutex);
    
    if (training_buffer.size() < TRAINING_BATCH_SIZE) {
        return; // Not enough data to train
    }
    
    // Prepare training batch
    std::vector<Position> positions;
    std::vector<std::vector<double>> policy_targets;
    std::vector<double> value_targets;
    
    // Take a batch from the buffer
    size_t batch_size = std::min(static_cast<size_t>(TRAINING_BATCH_SIZE), training_buffer.size());
    
    for (size_t i = 0; i < batch_size; ++i) {
        const auto& data = training_buffer[i];
        
        Position pos;
        Position::set(data.fen, pos);
        positions.push_back(pos);
        policy_targets.push_back(data.policy_target);
        value_targets.push_back(data.value_target);
    }
    
    // Remove trained data from buffer
    training_buffer.erase(training_buffer.begin(), training_buffer.begin() + batch_size);
    
    // Train the network
    try {
        net->train(positions, policy_targets, value_targets);
        std::cout << "Trained network on " << batch_size << " samples. Buffer size: " 
                  << training_buffer.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Training failed: " << e.what() << std::endl;
    }
}

void MCTSActive::clearTrainingBuffer() {
    std::lock_guard<std::mutex> lg(training_mutex);
    training_buffer.clear();
}

size_t MCTSActive::getTrainingBufferSize() const {
    std::lock_guard<std::mutex> lg(training_mutex);
    return training_buffer.size();
}

std::vector<double> MCTSActive::search(const Position& root_game) {
    initMoveEncodingActive();
    
    // 1) Initial expansion at root
    MCTSNodeActive* root = new MCTSNodeActive(root_game);
    
    // Initial inference
    std::vector<Position> root_states{root_game};
    std::vector<std::vector<double>> logp_b;
    std::vector<double> v_b;
    net->predict(root_states, logp_b, v_b);
    const auto& log_p = logp_b[0];
    double v = v_b[0];
    
    {
        std::lock_guard<std::mutex> lg(root->lock);
        root->visit_count = 1;
        root->value_sum = v;
    }

    // Compute full priors
    std::vector<double> priors_full(INDEX_MOVE_ACTIVE.size());
    for (size_t i = 0; i < INDEX_MOVE_ACTIVE.size(); ++i) {
        priors_full[i] = std::exp(log_p[i]);
    }
    
    // Legal moves and their indices
    std::vector<Move> legal_moves;
    std::vector<int> legal_indices;
    if (root_game.turn() == WHITE) {
        MoveList<WHITE> ml(root->game);
        for (auto m : ml) {
            legal_moves.push_back(m);
            std::string uci = moveToUCIActive(m);
            if (MOVE_INDEX_ACTIVE.find(uci) != MOVE_INDEX_ACTIVE.end()) {
                legal_indices.push_back(MOVE_INDEX_ACTIVE[uci]);
            }
        }
    } else {
        MoveList<BLACK> ml(root->game);
        for (auto m : ml) {
            legal_moves.push_back(m);
            std::string uci = moveToUCIActive(m);
            if (MOVE_INDEX_ACTIVE.find(uci) != MOVE_INDEX_ACTIVE.end()) {
                legal_indices.push_back(MOVE_INDEX_ACTIVE[uci]);
            }
        }
    }
    
    // Extract and normalize priors for legal moves
    std::vector<double> priors;
    priors.reserve(legal_indices.size());
    for (int idx : legal_indices) priors.push_back(priors_full[idx]);
    double total0 = std::accumulate(priors.begin(), priors.end(), 0.0);
    if (total0 > 0) for (auto& p : priors) p /= total0;
    
    legal_moves.resize(priors.size());

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
    {
        std::lock_guard<std::mutex> lg(root->lock);
        for (size_t i = 0; i < legal_moves.size(); ++i) {
            Position ng = root_game;
            ng.play(legal_moves[i]);
            root->children[moveToUCIActive(legal_moves[i])] = new MCTSNodeActive(ng, root, priors[i]);
        }
    }

    // 2) Batched MCTS simulations with training
    auto simulateOnce = [&](int) {
        MCTSNodeActive* node = root;
        std::vector<MCTSNodeActive*> path{root};
        
        while (true) {
            std::unique_lock<std::mutex> nl(node->lock);
            if (node->children.empty()) {
                nl.unlock();
                break;
            }
            
            double sqrt_visits = std::sqrt(node->visit_count);
            MCTSNodeActive* best_child = nullptr;
            double best_ucb = -std::numeric_limits<double>::infinity();
            
            for (auto& kv : node->children) {
                MCTSNodeActive* child = static_cast<MCTSNodeActive*>(kv.second);
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
        return std::make_pair(term, std::make_pair(node, path));
    };

    int num_done = 0;
    int batch_idx = 1;
    
    while (num_done < g_mcts_params.NUM_MCTS_SIMS) {
        // Selection
        int batch_size = g_mcts_params.BATCH_MCTS_SIZE;
        std::vector<std::future<std::pair<bool, std::pair<MCTSNodeActive*, std::vector<MCTSNodeActive*>>>>> futures;
        for (int i = 0; i < batch_size; ++i) {
            futures.push_back(std::async(std::launch::async, simulateOnce, i));
        }
        
        std::vector<MCTSNodeActive*> pending_nodes;
        std::vector<std::vector<MCTSNodeActive*>> pending_paths;
        
        for (auto& fut : futures) {
            auto result = fut.get();
            bool terminal = result.first;
            MCTSNodeActive* leaf = result.second.first;
            auto& path = result.second.second;
            
            if (terminal) {
                int res_val = leaf->game.result();
                double leaf_value = (res_val == 0) ? 0.0 : -1.0;
                size_t path_len = path.size();
                for (size_t d = 0; d < path_len; ++d) {
                    MCTSNodeActive* n = path[path_len - 1 - d];
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
                pending_nodes.push_back(leaf);
                pending_paths.push_back(path);
            }
        }
        
        if (pending_nodes.empty()) {
            break;
        }
        
        // Deduplicate and prepare for inference
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
        
        // Inference
        std::vector<Position> states;
        for (auto& fen : shard) {
            Position g;
            Position::set(fen, g);
            states.push_back(g);
        }
        
        std::vector<std::vector<double>> logp_b2;
        std::vector<double> v_b2;
        net->predict(states, logp_b2, v_b2);
        
        // Gather results and expand nodes
        std::map<std::string, std::pair<std::vector<double>, double>> global_res;
        for (size_t i = 0; i < shard.size(); ++i) {
            global_res[shard[i]] = {logp_b2[i], v_b2[i]};
        }
        
        // Expand & backprop with training data collection
        std::map<std::string, std::vector<std::pair<MCTSNodeActive*, std::vector<MCTSNodeActive*>>>> fen_map;
        for (size_t i = 0; i < pending_nodes.size(); ++i) {
            auto* nd = pending_nodes[i];
            auto fen = nd->game.fen();
            fen_map[fen].push_back({nd, pending_paths[i]});
        }
        
        std::vector<std::thread> threads;
        for (auto& kv : fen_map) {
            threads.emplace_back([&kv, &global_res, this]() {
                auto& node_path_pairs = kv.second;
                auto it = global_res.find(kv.first);
                if (it == global_res.end()) {
                    // No inference result; restore virtual loss
                    for (auto& [node, path] : node_path_pairs) {
                        size_t len = path.size();
                        for (size_t d = 0; d < len; ++d) {
                            MCTSNodeActive* n = path[len - 1 - d];
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
                
                MCTSNodeActive* first_node = node_path_pairs[0].first;
                Color leaf_turn = first_node->game.turn();
                
                // Get legal moves
                std::vector<Move> legals;
                std::vector<int> legal_inds;
                if (leaf_turn == WHITE) {
                    MoveList<WHITE> ml(first_node->game);
                    for (auto m : ml) {
                        legals.push_back(m);
                        std::string uci = moveToUCIActive(m);
                        if (MOVE_INDEX_ACTIVE.find(uci) != MOVE_INDEX_ACTIVE.end()) {
                            legal_inds.push_back(MOVE_INDEX_ACTIVE[uci]);
                        }
                    }
                } else {
                    MoveList<BLACK> ml(first_node->game);
                    for (auto m : ml) {
                        legals.push_back(m);
                        std::string uci = moveToUCIActive(m);
                        if (MOVE_INDEX_ACTIVE.find(uci) != MOVE_INDEX_ACTIVE.end()) {
                            legal_inds.push_back(MOVE_INDEX_ACTIVE[uci]);
                        }
                    }
                }
                
                // Compute and normalize priors
                std::vector<double> probs;
                probs.reserve(legal_inds.size());
                for (int idx : legal_inds) probs.push_back(std::exp(logp[idx]));
                double sum_p = std::accumulate(probs.begin(), probs.end(), 0.0);
                if (sum_p > 0) for (auto& p : probs) p /= sum_p;
                
                legals.resize(probs.size());
                
                // Expand all nodes with this position
                for (auto& [nd, path] : node_path_pairs) {
                    {
                        std::lock_guard<std::mutex> lg(nd->lock);
                        for (size_t j = 0; j < legals.size(); ++j) {
                            Position ng = nd->game;
                            ng.play(legals[j]);
                            nd->children[moveToUCIActive(legals[j])] = new MCTSNodeActive(ng, nd, probs[j]);
                        }
                    }
                    
                    // Backpropagate
                    size_t len = path.size();
                    for (size_t d = 0; d < len; ++d) {
                        MCTSNodeActive* n = path[len - 1 - d];
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
        
        // Collect training data from expanded nodes and train periodically
        if (batch_idx % TRAINING_FREQUENCY == 0) {
            {
                std::lock_guard<std::mutex> lg(training_mutex);
                for (auto* node : pending_nodes) {
                    if (!node->children.empty()) {
                        training_buffer.push_back(node->collectTrainingData());
                    }
                }
            }
            trainNetwork();
        }
        
        batch_idx++;
    }
    
    // Final training data collection from root
    {
        std::lock_guard<std::mutex> lg(training_mutex);
        if (!root->children.empty()) {
            training_buffer.push_back(root->collectTrainingData());
        }
    }
    trainNetwork(); // Final training
    
    // Generate final policy
    std::vector<double> policy(INDEX_MOVE_ACTIVE.size(), 0.0);
    {
        std::lock_guard<std::mutex> lg(root->lock);
        for (auto& kv : root->children) {
            if (MOVE_INDEX_ACTIVE.find(kv.first) != MOVE_INDEX_ACTIVE.end()) {
                policy[MOVE_INDEX_ACTIVE[kv.first]] = kv.second->visit_count;
            }
        }
    }
    
    double tot = std::accumulate(policy.begin(), policy.end(), 0.0);
    if (tot > 0) {
        for (auto& p : policy) p /= tot;
    }
    
    delete root;
    return policy;
}
