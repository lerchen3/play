#include "tables.h"
#include <random>

namespace zobrist {
    uint64_t zobrist_table[NPIECES][NSQUARES];
    void initialise_zobrist_keys() {
        std::mt19937_64 rng(0x12345678);
        for(int p=0; p<NPIECES; ++p) {
            for(int s=0; s<NSQUARES; ++s) {
                zobrist_table[p][s] = rng();
            }
        }
    }
}
