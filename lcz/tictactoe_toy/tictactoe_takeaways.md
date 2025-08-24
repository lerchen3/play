# TicTacToe MCTS Implementation: Complete Technical Analysis

## Overview

This document provides a comprehensive analysis of our Monte Carlo Tree Search (MCTS) implementation for TicTacToe, including all the critical insights discovered during debugging, the complete algorithm pseudocode, and corresponding C++ code snippets.

## Key Insights Discovered

### 1. **Value Perspective Consistency**
The most critical bug was inconsistent value perspectives between network evaluation and terminal evaluation. The fix required ensuring ALL values entering backpropagation are from the **leaf player's perspective**.

### 2. **Terminal vs Non-Terminal Value Logic**
- **Terminal nodes**: If someone won, the leaf represents the *loser's* turn, so `leaf_value = -1.0`
- **Non-terminal nodes**: Network returns value from current player's perspective directly

### 3. **UCB Exploitation Term**
Child values are stored from the child's perspective, but UCB needs parent's perspective, requiring negation: `exploit = -child->value()`

## Complete MCTS Pseudocode with C++ Mappings

### **Phase 1: Initialization**

```pseudocode
1.  CREATE root_node from input position
2.  RECORD root_player = (root_pos.turn == 'X') ? 1 : -1
3.  INITIALIZE constants: NUM_SIMS=200, CPUCT=1.0, NOISE_EPS=0.10, DIR_ALPHA=0.10
```

**C++ Implementation:**
```cpp
// Lines 108-113
TTTNode* root = new TTTNode(root_pos);
int root_player = (root_pos.turn == 'X') ? 1 : -1;
const int NUM_SIMS = 200;
const double CPUCT = 1.0;
const double NOISE_EPS = 0.10;
const double DIR_ALPHA = 0.10;
```

### **Phase 2: Root Expansion**

```pseudocode
4.  CALL neural_network.predict(root_position) → (log_probs, value)
5.  CONVERT log_probs to raw_priors via exp()
6.  FILTER priors to only legal moves
7.  NORMALIZE filtered priors to sum to 1
```

**C++ Implementation:**
```cpp
// Lines 118-134
std::vector<TTTPosition> state_batch{root_pos};
net->predict(state_batch, logps, values);
auto root_logp = logps[0];
for(int i = 0; i < 9; ++i){
    priors[i] = std::exp(root_logp[i]);
}
auto legal = legal_moves(root_pos);
for(int m : legal){
    move_priors.push_back(priors[m]);
}
```

```pseudocode
8.  GENERATE Dirichlet noise with alpha=DIR_ALPHA
9.  NORMALIZE noise to sum to 1
10. MIX priors = (1-NOISE_EPS)*priors + NOISE_EPS*noise
11. RE-NORMALIZE mixed priors
```

**C++ Implementation:**
```cpp
// Lines 145-170
std::gamma_distribution<double> gamma(DIR_ALPHA, 1.0);
for(double &n : noise){
    n = gamma(rng);
    noise_sum += n;
}
for(double &n : noise){
    n /= noise_sum;
}
for(size_t i = 0; i < move_priors.size(); ++i){
    move_priors[i] = (1 - NOISE_EPS) * move_priors[i] + NOISE_EPS * noise[i];
}
```

```pseudocode
12. FOR each legal_move:
13.     CREATE child_position by applying move to root_position
14.     CREATE child_node with child_position and corresponding prior
15.     ADD child_node to root.children[move]
```

**C++ Implementation:**
```cpp
// Lines 178-184
for(size_t i = 0; i < legal.size(); ++i){
    int move = legal[i];
    TTTPosition next_pos = root_pos;
    play_move(next_pos, move);
    double prior = move_priors[i];
    root->children[move] = new TTTNode(next_pos, root, prior);
}
```

### **Phase 3: MCTS Simulation Loop**

```pseudocode
16. FOR simulation = 1 to NUM_SIMS:
17.     INITIALIZE current_node = root
18.     INITIALIZE path = [root]
```

**C++ Implementation:**
```cpp
// Lines 187-189
for(int sim = 0; sim < NUM_SIMS; ++sim){
    TTTNode* node = root;
    std::vector<TTTNode*> path{root};
```

### **Phase 3a: Selection Phase**

```pseudocode
19.     WHILE current_node.expanded() AND game_not_terminal(current_node):
20.         INITIALIZE best_ucb = -infinity
21.         INITIALIZE best_child = null
22.         FOR each child in current_node.children:
23.             CALCULATE exploit = -child.value()  // Negate for parent perspective
24.             CALCULATE explore = CPUCT * child.prior * sqrt(parent.visits) / (1 + child.visits)
25.             CALCULATE ucb = exploit + explore
26.             IF ucb > best_ucb:
27.                 SET best_ucb = ucb
28.                 SET best_child = child
29.         SET current_node = best_child
30.         ADD current_node to path
```

**C++ Implementation:**
```cpp
// Lines 192-212
while(node->expanded() && check_winner(node->pos) == 2){
    double best_ucb = -1e9;
    TTTNode* best_child = nullptr;
    for(const auto &kv : node->children){
        TTTNode* child = kv.second;
        double exploit = -child->value();  // CRITICAL: Negate for perspective
        double explore = CPUCT * child->prior 
                        * std::sqrt(node->visit_count + 1e-8) 
                        / (1 + child->visit_count);
        double ucb = exploit + explore;
        if(ucb > best_ucb){
            best_ucb = ucb;
            best_child = child;
        }
    }
    node = best_child;
    path.push_back(node);
}
```

### **Phase 3b: Expansion & Evaluation**

```pseudocode
31.     CHECK game_result = check_winner(current_node.position)
32.     IF game_result == ONGOING:
33.         CALL neural_network.predict(current_node.position) → (leaf_logps, leaf_value)
34.         SET leaf_value = leaf_vs[0]  // From leaf player's perspective
35.         CONVERT leaf_logps to leaf_priors via exp()
36.         FILTER leaf_priors to legal moves only
37.         NORMALIZE filtered leaf_priors
38.         FOR each legal_move from leaf:
39.             CREATE grandchild_position by applying move
40.             CREATE grandchild_node with corresponding prior
41.             ADD grandchild_node to current_node.children[move]
```

**C++ Implementation:**
```cpp
// Lines 216-254
int result = check_winner(node->pos);
double leaf_value = 0.0;
if(result == 2){
    std::vector<TTTPosition> leaf_batch{node->pos};
    std::vector<std::vector<double>> leaf_logps;
    std::vector<double> leaf_vs;
    net->predict(leaf_batch, leaf_logps, leaf_vs);
    leaf_value = leaf_vs[0];  // From leaf player's perspective
    
    std::vector<double> leaf_priors(9);
    for(int i = 0; i < 9; ++i){
        leaf_priors[i] = std::exp(leaf_logps[0][i]);
    }
    auto leaf_moves = legal_moves(node->pos);
    // ... normalize and create children
}
```

```pseudocode
42.     ELIF game_result == DRAW:
43.         SET leaf_value = 0.0
44.     ELSE:  // Terminal win/loss
45.         SET leaf_value = -1.0  // Bad for leaf player (who is the loser)
```

**C++ Implementation:**
```cpp
// Lines 255-262
else if(result == 0){
    leaf_value = 0.0;  // Draw
}
else {
    // Terminal win/loss: leaf player is the loser
    leaf_value = -1.0;
}
```

### **Phase 3c: Backpropagation**

```pseudocode
46.     FOR each node in reverse(path):  // From leaf to root
47.         CALCULATE levels_from_leaf = distance_from_leaf
48.         IF levels_from_leaf is EVEN:
49.             SET node_value = leaf_value    // Same perspective as leaf
50.         ELSE:
51.             SET node_value = -leaf_value   // Opposite perspective (alternating)
52.         ADD node_value to node.value_sum
53.         INCREMENT node.visit_count
```

**C++ Implementation:**
```cpp
// Lines 265-277
for(auto it = path.rbegin(); it != path.rend(); ++it){
    TTTNode* n = *it;
    int levels_from_leaf = std::distance(path.rbegin(), it);
    // Leaf (levels=0): use leaf_value as-is
    // Parent (levels=1): flip sign since it's opponent's perspective  
    // Grandparent (levels=2): flip back, etc.
    double node_value = (levels_from_leaf % 2 == 0) ? leaf_value : -leaf_value;
    n->value_sum += node_value;
    n->visit_count += 1;
}
```

### **Phase 4: Final Policy Computation**

```pseudocode
54. INITIALIZE policy[9] = zeros
55. FOR each child in root.children:
56.     SET policy[child.move] = child.visit_count
57. NORMALIZE policy to sum to 1
58. CLEANUP all allocated nodes
59. RETURN policy
```

**C++ Implementation:**
```cpp
// Lines 281-299
std::vector<double> policy(9, 0.0);
for(const auto &kv : root->children){
    int mv = kv.first;
    policy[mv] = static_cast<double>(kv.second->visit_count);
}
double total = std::accumulate(policy.begin(), policy.end(), 0.0);
if(total > 0){
    for(double &p : policy){
        p /= total;
    }
}
```

## Critical Value Perspective Logic

### **The Core Insight: Consistent Leaf Perspective**

All values entering backpropagation must be from the **leaf player's perspective**:

1. **Non-terminal leaves**: Network directly returns value from current player's perspective
2. **Terminal leaves**: Convert game result to leaf player's perspective
3. **Backpropagation**: Alternate signs going up the tree

### **Terminal Value Logic Explained**

```cpp
// If X wins (result = +1) and current leaf turn is O:
// O is the loser, so from O's perspective this is bad = -1.0
// If X wins (result = +1) and current leaf turn is X:  
// This case never happens - X just won, so turn already switched to O
leaf_value = -1.0;  // Always -1.0 because leaf represents loser's turn
```

### **Backpropagation Alternating Logic**

```
Example: Root(X) → Child(O) → leaf_value = -1.0 (bad for O)

levels_from_leaf = 0 (Child): (-1.0) → Child stores -1.0 (bad for O) ✓
levels_from_leaf = 1 (Root): -(-1.0) = +1.0 → Root stores +1.0 (good for X) ✓
```

### **UCB Selection Logic**

```cpp
// At Root (X's turn), evaluating Child (O's move):
// child.value() = -1.0 (bad for O)
// exploit = -child.value() = -(-1.0) = +1.0 (excellent for X!)
// This ensures X strongly prefers moves that are bad for O
double exploit = -child->value();
```

## Network Interface Details

### **Input Representation**
```cpp
// 3×3×3 tensor: [X-plane, O-plane, turn-plane]
x = [1 if cell=='X' else 0 for cell in board]
o = [1 if cell=='O' else 0 for cell in board]  
turn_plane = [1.0 if current_player=='X' else 0.0] * 9
```

### **Output Processing**
```cpp
// Network returns (logits, value)
// Convert to log-probabilities for policy
policy_logprobs = log_softmax(logits)
// Value is from current player's perspective
value = tanh(value_head_output)  // ∈ [-1, +1]
```

## Why This Implementation Works

1. **Consistent Perspective**: All values are from leaf perspective, then consistently alternated
2. **Proper UCB**: Parent negates child values to get correct exploitation term
3. **Terminal Handling**: Terminal nodes correctly represent loser's perspective (-1.0)
4. **Visit-Count Policy**: Final policy based on visit counts reflects true search preference

## Common Pitfalls Avoided

1. **❌ Mixing Perspectives**: Don't mix root-perspective and leaf-perspective values
2. **❌ Wrong Terminal Values**: Don't use `value = result` directly
3. **❌ Missing UCB Negation**: Must negate child values in UCB formula
4. **❌ Inconsistent Backprop**: Must alternate signs consistently every level, starting from the leaf and going back up.

This implementation now correctly identifies winning moves, blocks opponent wins, and produces strong tactical play in TicTacToe positions.