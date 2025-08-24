#pragma once

#include "mcts.h"
#include "position.h"
#include "config.h"
#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include <limits>

// Training data structure for active inference
struct TrainingData {
    std::string fen;
    std::vector<double> policy_target;  // Visit counts normalized
    double value_target;                // Max Q(s,a) across all actions
};

// Enhanced Net interface with training capability
class NetActive : public Net {
public:
    // Training method for active inference
    virtual void train(const std::vector<Position>& positions,
                      const std::vector<std::vector<double>>& policy_targets,
                      const std::vector<double>& value_targets) = 0;
    virtual ~NetActive() = default;
};

// Active inference MCTS node - same as MCTSNode but with training data collection
class MCTSNodeActive : public MCTSNode {
public:
    MCTSNodeActive(const Position& gameState, MCTSNodeActive* parent_ = nullptr, double prior_ = 0.0);
    virtual ~MCTSNodeActive();
    
    // Collect training data from this node
    TrainingData collectTrainingData() const;
    
    // Get Q-value for this node (average value)
    double getQValue() const;
};

// Active inference MCTS implementation
class MCTSActive {
private:
    NetActive* net;
    std::vector<TrainingData> training_buffer;
    mutable std::mutex training_mutex;
    
    // Training parameters
    static constexpr int TRAINING_BATCH_SIZE = 32;
    static constexpr int TRAINING_FREQUENCY = 10; // Train every N MCTS batches
    
public:
    explicit MCTSActive(Net* net_);
    
    // Main search function - returns policy and trains network
    std::vector<double> search(const Position& root_game);
    
    // Train the network on collected data
    void trainNetwork();
    
    // Clear training buffer
    void clearTrainingBuffer();
    
    // Get current training buffer size
    size_t getTrainingBufferSize() const;
};

// Helper functions
void initMoveEncodingActive();
std::string moveToUCIActive(const Move& m);
Move uciToMoveActive(const std::string& uci, const Position& pos); 