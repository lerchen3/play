#pragma once

#include <map>
#include <vector>
#include <string>
#include <mutex>
#include "config.h"
#include "position.h"

// Expose move encoding tables
extern std::map<std::string, int> MOVE_INDEX;
extern std::vector<std::string> INDEX_MOVE;

// Convert UCI string to a Move using board context
Move uciToMove(const std::string& uci, const Position& pos);

// Node for Monte Carlo Tree Search
struct MCTSNode {
    Position game;              // game state at this node
    MCTSNode* parent;           // parent node in the tree
    double prior;               // prior probability from policy network
    int visit_count;            // number of visits
    double value_sum;           // cumulative value sum
    std::map<std::string, MCTSNode*> children; // child nodes keyed by UCI move string
    mutable std::mutex lock;    // mutex for thread-safe updates

    MCTSNode(const Position& gameState, MCTSNode* parent = nullptr, double prior = 0.0);
    ~MCTSNode();
    bool isExpanded() const;
    double value() const;
};

// Abstract neural network interface for batch prediction
class Net {
public:
    // For each state in 'states', produce log probabilities for all moves and a value estimate
    virtual void predict(const std::vector<Position>& states,
                         std::vector<std::vector<double>>& logps,
                         std::vector<double>& vs) = 0;
    virtual ~Net() = default;
};

// Monte Carlo Tree Search class
class MCTS {
public:
    explicit MCTS(Net* net);
    // Runs MCTS starting from root_game and returns a policy vector over all moves
    std::vector<double> search(const Position& root_game);

private:
    Net* net;  // neural network for policy/value predictions
}; 