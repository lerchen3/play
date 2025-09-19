import ctypes
import os
import sys
from typing import List, Tuple

# Global variable to hold the loaded library
_lib = None

def _load_library():
    """Load the trie tokenizer shared library."""
    global _lib
    if _lib is not None:
        return _lib
    
    # Try to find the library in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, "libtrie_tokenizer.so")
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Trie tokenizer library not found at {lib_path}. Please run build_trie_tokenizer.sh first.")
    
    _lib = ctypes.CDLL(lib_path)
    
    # Define function signatures
    _lib.create_trie.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), 
                                ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    _lib.create_trie.restype = ctypes.c_void_p
    
    _lib.encode_with_trie.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), 
                                     ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    _lib.encode_with_trie.restype = ctypes.POINTER(ctypes.c_int)
    
    _lib.destroy_trie.argtypes = [ctypes.c_void_p]
    _lib.destroy_trie.restype = None
    
    _lib.free_encoded_tokens.argtypes = [ctypes.POINTER(ctypes.c_int)]
    _lib.free_encoded_tokens.restype = None
    
    return _lib

class TrieTokenizer:
    """Python wrapper for the C++ trie tokenizer."""
    
    def __init__(self, vocab: List[Tuple[List[int], int]]):
        """Initialize the trie with vocabulary.
        
        Args:
            vocab: List of (sequence, token_id) tuples
        """
        self.lib = _load_library()
        self._trie_ptr = None
        self._build_trie(vocab)
    
    def _build_trie(self, vocab: List[Tuple[List[int], int]]):
        """Build the trie from vocabulary."""
        if not vocab:
            raise ValueError("Vocabulary cannot be empty")
        
        # Flatten sequences and prepare arrays
        all_sequences = []
        sequence_lengths = []
        token_ids = []
        
        for seq, token_id in vocab:
            all_sequences.extend(seq)
            sequence_lengths.append(len(seq))
            token_ids.append(token_id)
        
        # Convert to ctypes arrays
        sequences_array = (ctypes.c_int * len(all_sequences))(*all_sequences)
        lengths_array = (ctypes.c_int * len(sequence_lengths))(*sequence_lengths)
        ids_array = (ctypes.c_int * len(token_ids))(*token_ids)
        
        # Create the trie
        self._trie_ptr = self.lib.create_trie(sequences_array, lengths_array, ids_array, len(vocab))
        if not self._trie_ptr:
            raise RuntimeError("Failed to create trie")
    
    def encode(self, tokens: List[int]) -> List[int]:
        """Encode tokens using the trie.
        
        Args:
            tokens: Input tokens to encode
            
        Returns:
            Encoded tokens
        """
        if not tokens:
            return []
        
        if self._trie_ptr is None:
            raise RuntimeError("Trie not initialized")
        
        # Convert input to ctypes array
        input_array = (ctypes.c_int * len(tokens))(*tokens)
        output_length = ctypes.c_int()
        
        # Call the C function
        result_ptr = self.lib.encode_with_trie(self._trie_ptr, input_array, len(tokens), ctypes.byref(output_length))
        
        if not result_ptr:
            raise RuntimeError("Encoding failed")
        
        # Extract results
        output = []
        for i in range(output_length.value):
            output.append(result_ptr[i])
        
        # Free the result memory
        self.lib.free_encoded_tokens(result_ptr)
        
        return output
    
    def __del__(self):
        """Clean up the trie when the object is destroyed."""
        if hasattr(self, '_trie_ptr') and self._trie_ptr is not None:
            try:
                self.lib.destroy_trie(self._trie_ptr)
            except:
                pass  # Ignore errors during cleanup

# Global trie tokenizer instance
_global_tokenizer = None

def encode_trie(tokens: List[int], vocab: List[Tuple[List[int], int]]) -> List[int]:
    """Encode tokens using trie tokenization.
    
    This function maintains API compatibility with the original Python implementation.
    
    Args:
        tokens: Input tokens to encode
        vocab: Vocabulary as list of (sequence, token_id) tuples
        
    Returns:
        Encoded tokens
    """
    global _global_tokenizer
    
    # For efficiency, we could cache the tokenizer if the vocab doesn't change
    # For now, we create a new one each time to maintain compatibility
    tokenizer = TrieTokenizer(vocab)
    return tokenizer.encode(tokens) 