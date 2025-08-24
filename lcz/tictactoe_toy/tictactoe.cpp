#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <mutex>
#include <memory>
#include <cstring>
#include <iostream>

extern "C" {
struct TTTGameHandle;
struct TTTMCTSHandle;

// Game API
TTTGameHandle* ttt_game_create();
void ttt_game_destroy(TTTGameHandle* g);
void ttt_game_reset(TTTGameHandle* g);
const char* ttt_game_get_state(TTTGameHandle* g);
int ttt_game_turn(TTTGameHandle* g); // 1=X, -1=O
int ttt_game_move(TTTGameHandle* g, int idx); // 0-8
int ttt_game_is_game_over(TTTGameHandle* g);
int ttt_game_result(TTTGameHandle* g); // 1=X win, -1=O win, 0=draw/ongoing

// MCTS API
typedef void (*ttt_predict_cb)(const char* state, float* policy, float* value);
TTTMCTSHandle* ttt_mcts_create(ttt_predict_cb cb);
void ttt_mcts_destroy(TTTMCTSHandle* m);
void ttt_mcts_search(TTTMCTSHandle* m, const char* state, float* out_policy);
}

// ----------------------------------
struct TTTPosition {
    char board[9]; // '.', 'X', 'O'
    char turn; // 'X' or 'O'
    TTTPosition() { reset(); }
    void reset() {
        for(int i=0;i<9;++i) board[i]='.'; turn='X';
    }
    std::string str() const {
        std::string s(board, board+9);
        s.push_back(turn);
        return s;
    }
    static TTTPosition from_str(const std::string& s){
        TTTPosition p;
        for(int i=0;i<9;++i) p.board[i]=s[i];
        p.turn=s[9];
        return p;
    }
};

static int check_winner(const TTTPosition& p){
    const int wins[8][3]={{0,1,2},{3,4,5},{6,7,8},{0,3,6},{1,4,7},{2,5,8},{0,4,8},{2,4,6}};
    for(auto& w:wins){
        char a=p.board[w[0]],b=p.board[w[1]],c=p.board[w[2]];
        if(a!='.' && a==b && b==c) return a=='X'?1:-1;
    }
    for(int i=0;i<9;++i) if(p.board[i]=='.') return 2; // ongoing
    return 0; // draw
}

static std::vector<int> legal_moves(const TTTPosition& p){
    std::vector<int> m; m.reserve(9);
    for(int i=0;i<9;++i) if(p.board[i]=='.') m.push_back(i);
    return m;
}

static void play_move(TTTPosition& p, int idx) {
    // Cache legal moves once
    auto moves = legal_moves(p);
    if (std::find(moves.begin(), moves.end(), idx) == moves.end()) {
        std::cerr << "Error: Invalid move index: " << idx << std::endl;
        return;
    }
    p.board[idx]=p.turn;
    p.turn=(p.turn=='X')?'O':'X';
}

// ----------------------------------
struct TTTNode {
    TTTPosition pos;
    TTTNode* parent;
    double prior;
    int visit_count;
    double value_sum;
    std::map<int,TTTNode*> children;
    std::mutex mtx;
    TTTNode(const TTTPosition& p, TTTNode* par=nullptr,double pr=0.0)
        :pos(p),parent(par),prior(pr),visit_count(0),value_sum(0.0){}
    bool expanded() const { return !children.empty(); }
    double value() const { return visit_count? value_sum/visit_count : 0.0; }
};

class TTTNet {
public:
    virtual void predict(const std::vector<TTTPosition>& states,
                         std::vector<std::vector<double>>& logps,
                         std::vector<double>& vs)=0;
    virtual ~TTTNet()=default;
};

class TTTMCTS {
public:
    explicit TTTMCTS(TTTNet* n):net(n){}
    // Perform MCTS search from the given root position using the neural network for policy and value predictions
    std::vector<double> search(const TTTPosition& root_pos){
        // Constants controlling number of simulations and exploration behavior
        const int NUM_SIMS    = 200;      // how many MCTS simulations to run
        const double CPUCT    = 1.0;      // exploration constant in UCB formula
        const double NOISE_EPS= 0.10;     // mixing fraction for Dirichlet noise
        const double DIR_ALPHA= 0.10;      // alpha parameter for Dirichlet distribution

        // Create root node for MCTS tree with initial position
        TTTNode* root = new TTTNode(root_pos);
        
        // Remember root player for consistent value calculation
        int root_player = (root_pos.turn == 'X') ? 1 : -1;

        // --- Initial Expansion of Root Node ---
        // Prepare a batch of one state for neural network prediction
        std::vector<TTTPosition> state_batch{root_pos};
        std::vector<std::vector<double>> logps; // will hold log-probabilities from network
        std::vector<double> values;             // will hold value estimates from network
        net->predict(state_batch, logps, values); // fill logps[0] and values[0]

        // Convert network log-probabilities to raw priors via exponentiation
        auto root_logp = logps[0];             // log-probabilities for root position
        std::vector<double> priors(9);         // priors for all 9 possible moves
        for(int i = 0; i < 9; ++i){
            priors[i] = std::exp(root_logp[i]); // compute raw prior for move i
        }

        // Filter priors to only legal moves at root
        auto legal = legal_moves(root_pos);    // indices of legal moves at root
        std::vector<double> move_priors;       // priors corresponding to legal moves
        move_priors.reserve(legal.size());
        for(int m : legal){
            move_priors.push_back(priors[m]);  // pick prior for each legal move
        }

        // Normalize move_priors so they sum to 1 (if non-zero)
        double sum_pr = std::accumulate(move_priors.begin(), move_priors.end(), 0.0);
        if(sum_pr > 0){
            for(double &p : move_priors){
                p /= sum_pr;                   // normalization step
            }
        }

        // Add Dirichlet noise to move priors to encourage exploration
        if(!move_priors.empty()){
            std::gamma_distribution<double> gamma(DIR_ALPHA, 1.0); // noise generator
            std::mt19937 rng(std::random_device{}());             // random engine
            std::vector<double> noise(move_priors.size());
            double noise_sum = 0.0;
            // Sample noise values and accumulate
            for(double &n : noise){
                n = gamma(rng);            // sample from gamma(DIR_ALPHA,1)
                noise_sum += n;
            }
            // Normalize noise to sum to 1
            for(double &n : noise){
                n /= noise_sum;
            }
            // Mix noise into move_priors according to NOISE_EPS
            for(size_t i = 0; i < move_priors.size(); ++i){
                move_priors[i] = (1 - NOISE_EPS) * move_priors[i]
                                 + NOISE_EPS * noise[i];
            }
            // Re-normalize mixed priors
            double mixed_sum = std::accumulate(move_priors.begin(), move_priors.end(), 0.0);
            if(mixed_sum > 0){
                for(double &p : move_priors){
                    p /= mixed_sum;
                }
            }
        }

        // Create initial children of root for each legal move, setting prior probabilities
        for(size_t i = 0; i < legal.size(); ++i){
            int move = legal[i];               // move index
            TTTPosition next_pos = root_pos;   // copy root position
            play_move(next_pos, move);         // apply move to get new position
            double prior = move_priors[i];     // prior for this child
            root->children[move] = new TTTNode(next_pos, root, prior);
        }

        // --- Run MCTS Simulations ---
        for(int sim = 0; sim < NUM_SIMS; ++sim){
            TTTNode* node = root;              // start at root
            std::vector<TTTNode*> path{root};   // record path for backpropagation

            // --- Selection Phase ---
            // Traverse down the tree using UCB until a leaf or terminal node is reached
            while(node->expanded() && check_winner(node->pos) == 2){
                double best_ucb = -1e9;         // best UCB score seen so far
                TTTNode* best_child = nullptr;
                for(const auto &kv : node->children){
                    TTTNode* child = kv.second;
                    // exploitation term = average value of child FROM PARENT'S PERSPECTIVE
                    // child->value() is from child's perspective, so negate it
                    double exploit = -child->value();
                    // exploration term = CPUCT * prior * sqrt(N_parent) / (1 + N_child)
                    double explore = CPUCT * child->prior
                                     * std::sqrt(node->visit_count + 1e-8)
                                     / (1 + child->visit_count);
                    double ucb = exploit + explore;
                    if(ucb > best_ucb){
                        best_ucb = ucb;
                        best_child = child;
                    }
                }
                node = best_child;               // descend to best child
                path.push_back(node);            // add node to path
            }

            // --- Expansion & Evaluation Phase ---
            int result = check_winner(node->pos); // check if node is terminal
            double leaf_value = 0.0;              // value from LEAF player's perspective
            if(result == 2){
                // Non-terminal leaf: expand with network evaluation
                std::vector<TTTPosition> leaf_batch{node->pos};
                std::vector<std::vector<double>> leaf_logps;
                std::vector<double> leaf_vs;
                net->predict(leaf_batch, leaf_logps, leaf_vs); // network eval
                leaf_value = leaf_vs[0];         // value from leaf player's perspective (current player)

                // Convert log-probabilities to priors
                std::vector<double> leaf_priors(9);
                for(int i = 0; i < 9; ++i){
                    leaf_priors[i] = std::exp(leaf_logps[0][i]);
                }
                // Collect legal moves from leaf
                auto leaf_moves = legal_moves(node->pos);
                std::vector<double> child_priors;
                child_priors.reserve(leaf_moves.size());
                for(int m : leaf_moves){
                    child_priors.push_back(leaf_priors[m]);
                }
                // Normalize child priors
                double lp_sum = std::accumulate(child_priors.begin(), child_priors.end(), 0.0);
                if(lp_sum > 0){
                    for(double &p : child_priors){
                        p /= lp_sum;
                    }
                }
                // Expand leaf: create child nodes
                for(size_t i = 0; i < leaf_moves.size(); ++i){
                    int mv = leaf_moves[i];
                    TTTPosition child_pos = node->pos;
                    play_move(child_pos, mv);
                    node->children[mv] = new TTTNode(child_pos, node, child_priors[i]);
                }
            }
            else if(result == 0){
                // Terminal draw: neutral outcome for leaf player
                leaf_value = 0.0;
            }
            else {
                // Terminal win/loss: someone just won by making the previous move
                // node->pos.turn is the player who WOULD move next (the loser)
                // So from the leaf player's perspective (the loser), this is bad
                leaf_value = -1.0;
            }

            // --- Backpropagation Phase ---
            // Each node stores value from the perspective of the player to move at that node
            // leaf_value is from leaf player's perspective, alternate signs going up the tree
            for(auto it = path.rbegin(); it != path.rend(); ++it){
                TTTNode* n = *it;
                // Determine how many levels up from leaf we are
                int levels_from_leaf = std::distance(path.rbegin(), it);
                // Leaf (levels=0): use leaf_value as-is
                // Parent (levels=1): flip sign since it's the opponent's perspective
                // Grandparent (levels=2): flip back, etc.
                double node_value = (levels_from_leaf % 2 == 0) ? leaf_value : -leaf_value;
                n->value_sum += node_value;
                n->visit_count += 1;
            }
        }

        // --- Compute Final Policy from Visit Counts ---
        std::vector<double> policy(9, 0.0);
        for(const auto &kv : root->children){
            int mv = kv.first;
            policy[mv] = static_cast<double>(kv.second->visit_count); // visits = score
        }
        // Normalize policy vector to sum to 1
        double total = std::accumulate(policy.begin(), policy.end(), 0.0);
        if(total > 0){
            for(double &p : policy){
                p /= total;
            }
        }

        // --- Cleanup: free allocated nodes ---
        for(const auto &kv : root->children){
            delete kv.second;
        }
        delete root;

        return policy; // return normalized policy over all moves
    }
private:
    TTTNet* net; // pointer to the neural network used for policy/value inference
};

// CallbackNet for C API
struct CallbackNet : public TTTNet {
    ttt_predict_cb cb;
    explicit CallbackNet(ttt_predict_cb c):cb(c){}
    void predict(const std::vector<TTTPosition>& states,
                 std::vector<std::vector<double>>& logps,
                 std::vector<double>& vs) override{
        logps.resize(states.size()); vs.resize(states.size());
        for(size_t i=0;i<states.size();++i){
            auto s=states[i].str();
            std::vector<float> buf(9); float val;
            cb(s.c_str(), buf.data(), &val);
            vs[i]=val; logps[i].assign(buf.begin(), buf.end());
        }
    }
};

// Handles
struct TTTGameHandle{TTTPosition pos; std::string cached;};
struct TTTMCTSHandle{std::unique_ptr<TTTNet> net; std::unique_ptr<TTTMCTS> mcts;};

extern "C" {
TTTGameHandle* ttt_game_create(){return new TTTGameHandle();}
void ttt_game_destroy(TTTGameHandle* g){delete g;}
void ttt_game_reset(TTTGameHandle* g){g->pos.reset();}
const char* ttt_game_get_state(TTTGameHandle* g){g->cached=g->pos.str(); return g->cached.c_str();}
int ttt_game_turn(TTTGameHandle* g){return g->pos.turn=='X'?1:-1;}
int ttt_game_move(TTTGameHandle* g, int idx){
    auto moves=legal_moves(g->pos);
    if(std::find(moves.begin(),moves.end(),idx)==moves.end()) return 0;
    play_move(g->pos,idx); return 1;
}
int ttt_game_is_game_over(TTTGameHandle* g){return check_winner(g->pos)!=2;}
int ttt_game_result(TTTGameHandle* g){int r=check_winner(g->pos); return r==2?0:r;}

TTTMCTSHandle* ttt_mcts_create(ttt_predict_cb cb){
    auto h=new TTTMCTSHandle();
    h->net=std::unique_ptr<TTTNet>(new CallbackNet(cb));
    h->mcts=std::unique_ptr<TTTMCTS>(new TTTMCTS(h->net.get()));
    return h;
}
void ttt_mcts_destroy(TTTMCTSHandle* h){delete h;}
void ttt_mcts_search(TTTMCTSHandle* h,const char* state,float* out_policy){
    TTTPosition p=TTTPosition::from_str(state);
    auto pol=h->mcts->search(p);
    for(int i=0;i<9;++i) out_policy[i]=(float)pol[i];
}
}
