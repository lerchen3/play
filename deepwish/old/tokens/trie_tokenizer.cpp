#include "trie_tokenizer.h"
#include <cstdlib>
#include <cstring>
#include <vector>

// Helper function to recursively clear trie nodes
static void clear_trie_recursive(TrieNode* node) {
    if (!node) return;
    for (auto& pair : node->children) {
        clear_trie_recursive(pair.second);
    }
    delete node;
}

extern "C" {

TrieNode* create_trie(const int* sequences, const int* sequence_lengths, const int* token_ids, int vocab_size) {
    TrieNode* root = new TrieNode();
    
    int seq_offset = 0;
    for (int i = 0; i < vocab_size; ++i) {
        int seq_len = sequence_lengths[i];
        int token_id = token_ids[i];
        
        TrieNode* node = root;
        for (int j = 0; j < seq_len; ++j) {
            int token = sequences[seq_offset + j];
            if (node->children.find(token) == node->children.end()) {
                node->children[token] = new TrieNode();
            }
            node = node->children[token];
        }
        node->token_id = token_id;
        
        seq_offset += seq_len;
    }
    
    return root;
}

int* encode_with_trie(TrieNode* root, const int* input_tokens, int input_length, int* output_length) {
    std::vector<int> output;
    
    int i = 0;
    while (i < input_length) {
        TrieNode* node = root;
        int j = i;
        int best_token_id = -1;
        int best_length = 0;
        
        // Traverse the trie to find the longest match
        while (j < input_length && node->children.find(input_tokens[j]) != node->children.end()) {
            node = node->children[input_tokens[j]];
            j++;
            if (node->token_id != -1) {
                best_token_id = node->token_id;
                best_length = j - i;
            }
        }
        
        if (best_token_id != -1) {
            output.push_back(best_token_id);
            i += best_length;
        } else {
            output.push_back(input_tokens[i]);
            i++;
        }
    }
    
    *output_length = output.size();
    int* result = (int*)malloc(output.size() * sizeof(int));
    for (size_t k = 0; k < output.size(); ++k) {
        result[k] = output[k];
    }
    
    return result;
}

void destroy_trie(TrieNode* root) {
    clear_trie_recursive(root);
}

void free_encoded_tokens(int* tokens) {
    free(tokens);
}

}
