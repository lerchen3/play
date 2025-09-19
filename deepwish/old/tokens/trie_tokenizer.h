#pragma once

#include <vector>
#include <unordered_map>

// Trie node structure for tokenization
struct TrieNode {
    std::unordered_map<int, TrieNode*> children;
    int token_id = -1;
};

// C API for trie tokenizer
extern "C" {
    // Initialize a trie with vocabulary
    // vocab is an array of (sequence, sequence_length, token_id) tuples
    TrieNode* create_trie(const int* sequences, const int* sequence_lengths, const int* token_ids, int vocab_size);
    
    // Encode tokens using the trie
    // Returns encoded tokens and sets output_length
    int* encode_with_trie(TrieNode* root, const int* input_tokens, int input_length, int* output_length);
    
    // Clean up trie memory
    void destroy_trie(TrieNode* root);
    
    // Free encoded result memory
    void free_encoded_tokens(int* tokens);
} 