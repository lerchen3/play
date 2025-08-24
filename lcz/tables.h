#pragma once

#include "types.h"
#include <cstdlib>
#include <cmath>

// Zobrist keys
namespace zobrist {
    extern uint64_t zobrist_table[NPIECES][NSQUARES];
    void initialise_zobrist_keys();
}

inline int rank_of(Square s);
inline int file_of(Square s);

// Bitboard tables
inline Bitboard SQUARE_BB[NSQUARES] = {
    1ULL<<0,  1ULL<<1,  1ULL<<2,  1ULL<<3,  1ULL<<4,  1ULL<<5,  1ULL<<6,  1ULL<<7,
    1ULL<<8,  1ULL<<9,  1ULL<<10, 1ULL<<11, 1ULL<<12, 1ULL<<13, 1ULL<<14, 1ULL<<15,
    1ULL<<16, 1ULL<<17, 1ULL<<18, 1ULL<<19, 1ULL<<20, 1ULL<<21, 1ULL<<22, 1ULL<<23,
    1ULL<<24, 1ULL<<25, 1ULL<<26, 1ULL<<27, 1ULL<<28, 1ULL<<29, 1ULL<<30, 1ULL<<31,
    1ULL<<32, 1ULL<<33, 1ULL<<34, 1ULL<<35, 1ULL<<36, 1ULL<<37, 1ULL<<38, 1ULL<<39,
    1ULL<<40, 1ULL<<41, 1ULL<<42, 1ULL<<43, 1ULL<<44, 1ULL<<45, 1ULL<<46, 1ULL<<47,
    1ULL<<48, 1ULL<<49, 1ULL<<50, 1ULL<<51, 1ULL<<52, 1ULL<<53, 1ULL<<54, 1ULL<<55,
    1ULL<<56, 1ULL<<57, 1ULL<<58, 1ULL<<59, 1ULL<<60, 1ULL<<61, 1ULL<<62, 1ULL<<63
};

inline Bitboard MASK_RANK[8] = {
    0xFFULL << 0, 0xFFULL << 8, 0xFFULL << 16, 0xFFULL << 24,
    0xFFULL << 32, 0xFFULL << 40, 0xFFULL << 48, 0xFFULL << 56
};

inline Bitboard SQUARES_BETWEEN_BB[NSQUARES][NSQUARES];
inline Bitboard LINE[NSQUARES][NSQUARES];

struct TablesInit {
    TablesInit() {
        for (int i = 0; i < NSQUARES; ++i) {
            SQUARE_BB[i] = 1ULL << i;
        }
        for (int r = 0; r < 8; ++r)
            MASK_RANK[r] = 0xFFULL << (8 * r);

        for (int a = 0; a < NSQUARES; ++a) {
            for (int b = 0; b < NSQUARES; ++b) {
                Bitboard bb = 0;
                if (a != b) {
                    int ra = a / 8, fa = a % 8;
                    int rb = b / 8, fb = b % 8;
                    int dr = (rb > ra) ? 1 : (rb < ra ? -1 : 0);
                    int df = (fb > fa) ? 1 : (fb < fa ? -1 : 0);
                    if (dr == 0 || df == 0 || std::abs(rb - ra) == std::abs(fb - fa)) {
                        int step = dr * 8 + df;
                        if (step != 0) {
                            int sq = a + step;
                            while (sq != b) {
                                bb |= 1ULL << sq;
                                sq += step;
                            }
                        }
                    }
                }
                SQUARES_BETWEEN_BB[a][b] = bb;
                LINE[a][b] = bb ? (bb | SQUARE_BB[a] | SQUARE_BB[b]) : (a==b ? SQUARE_BB[a] : 0);
            }
        }
    }
};
inline TablesInit TABLES_INIT;

// Attack functions stubs
template<Color C>
inline Bitboard pawn_attacks(Bitboard b) {
    if constexpr (C == WHITE)
        return ((b & ~0x0101010101010101ULL) << 7) |
               ((b & ~0x8080808080808080ULL) << 9);
    else
        return ((b & ~0x8080808080808080ULL) >> 7) |
               ((b & ~0x0101010101010101ULL) >> 9);
}

template<Color C>
inline Bitboard pawn_attacks(Square s) { return pawn_attacks<C>(SQUARE_BB[s]); }

template<int PT>
inline Bitboard attacks(Square s, Bitboard occ);

template<>
inline Bitboard attacks<KNIGHT>(Square s, Bitboard) {
    int r = rank_of(s), f = file_of(s);
    Bitboard bb = 0;
    const int dr[8] = {2, 1,-1,-2,-2,-1,1,2};
    const int df[8] = {1, 2, 2, 1,-1,-2,-2,-1};
    for (int i=0;i<8;i++) {
        int rr = r + dr[i];
        int ff = f + df[i];
        if (rr>=0 && rr<8 && ff>=0 && ff<8)
            bb |= 1ULL << (rr*8+ff);
    }
    return bb;
}

template<>
inline Bitboard attacks<KING>(Square s, Bitboard) {
    int r = rank_of(s), f = file_of(s);
    Bitboard bb = 0;
    for(int dr=-1; dr<=1; ++dr) for(int df=-1; df<=1; ++df) {
        if(dr==0 && df==0) continue;
        int rr=r+dr, ff=f+df;
        if(rr>=0&&rr<8&&ff>=0&&ff<8) bb|=1ULL<<(rr*8+ff);
    }
    return bb;
}

template<>
inline Bitboard attacks<BISHOP>(Square s, Bitboard occ) {
    Bitboard bb = 0;
    int r = rank_of(s), f = file_of(s);
    for(int dr=1, df=1; r+dr<8 && f+df<8; ++dr, ++df){ int sq=(r+dr)*8+f+df; bb|=1ULL<<sq; if(occ & (1ULL<<sq)) break; }
    for(int dr=1, df=-1; r+dr<8 && f+df>=0; ++dr, --df){ int sq=(r+dr)*8+f+df; bb|=1ULL<<sq; if(occ & (1ULL<<sq)) break; }
    for(int dr=-1, df=1; r+dr>=0 && f+df<8; --dr, ++df){ int sq=(r+dr)*8+f+df; bb|=1ULL<<sq; if(occ & (1ULL<<sq)) break; }
    for(int dr=-1, df=-1; r+dr>=0 && f+df>=0; --dr, --df){ int sq=(r+dr)*8+f+df; bb|=1ULL<<sq; if(occ & (1ULL<<sq)) break; }
    return bb;
}

template<>
inline Bitboard attacks<ROOK>(Square s, Bitboard occ) {
    Bitboard bb = 0;
    int r = rank_of(s), f = file_of(s);
    for(int r1=r+1;r1<8;++r1){ int sq=r1*8+f; bb|=1ULL<<sq; if(occ & (1ULL<<sq)) break; }
    for(int r1=r-1;r1>=0;--r1){ int sq=r1*8+f; bb|=1ULL<<sq; if(occ & (1ULL<<sq)) break; }
    for(int f1=f+1;f1<8;++f1){ int sq=r*8+f1; bb|=1ULL<<sq; if(occ & (1ULL<<sq)) break; }
    for(int f1=f-1;f1>=0;--f1){ int sq=r*8+f1; bb|=1ULL<<sq; if(occ & (1ULL<<sq)) break; }
    return bb;
}

template<>
inline Bitboard attacks<QUEEN>(Square s, Bitboard occ) {
    return attacks<BISHOP>(s, occ) | attacks<ROOK>(s, occ);
}

inline Bitboard attacks(PieceType pt, Square s, Bitboard occ) {
    switch(pt) {
        case PAWN:   return 0; // pawns handled separately
        case KNIGHT: return attacks<KNIGHT>(s, occ);
        case BISHOP: return attacks<BISHOP>(s, occ);
        case ROOK:   return attacks<ROOK>(s, occ);
        case QUEEN:  return attacks<QUEEN>(s, occ);
        case KING:   return attacks<KING>(s, occ);
    }
    return 0;
}

inline Bitboard sliding_attacks(Square s, Bitboard occ, Bitboard mask) {
    Bitboard bb = 0;
    Bitboard moves = attacks<BISHOP>(s, occ) | attacks<ROOK>(s, occ);
    bb = moves & mask;
    return bb;
}

inline Bitboard get_xray_rook_attacks(Square s, Bitboard occ, Bitboard us) {
    Bitboard attacks1 = attacks<ROOK>(s, occ);
    Bitboard blockers = attacks1 & us;
    occ ^= blockers;
    Bitboard attacks2 = attacks<ROOK>(s, occ);
    return attacks2 & ~attacks1;
}

inline Bitboard get_xray_bishop_attacks(Square s, Bitboard occ, Bitboard us) {
    Bitboard attacks1 = attacks<BISHOP>(s, occ);
    Bitboard blockers = attacks1 & us;
    occ ^= blockers;
    Bitboard attacks2 = attacks<BISHOP>(s, occ);
    return attacks2 & ~attacks1;
}

// Castling masks stubs
template<Color C> inline Bitboard oo_mask() {
    if constexpr (C == WHITE)
        return SQUARE_BB[e1] | SQUARE_BB[h1];
    else
        return SQUARE_BB[e8] | SQUARE_BB[h8];
}

template<Color C> inline Bitboard ooo_mask() {
    if constexpr (C == WHITE)
        return SQUARE_BB[e1] | SQUARE_BB[a1];
    else
        return SQUARE_BB[e8] | SQUARE_BB[a8];
}

template<Color C> inline Bitboard oo_blockers_mask() {
    if constexpr (C == WHITE)
        return SQUARE_BB[f1] | SQUARE_BB[g1];
    else
        return SQUARE_BB[f8] | SQUARE_BB[g8];
}

template<Color C> inline Bitboard ooo_blockers_mask() {
    if constexpr (C == WHITE)
        return SQUARE_BB[b1] | SQUARE_BB[c1] | SQUARE_BB[d1];
    else
        return SQUARE_BB[b8] | SQUARE_BB[c8] | SQUARE_BB[d8];
}

// Starting rook squares for castling moves
template<Color C> inline Square rook_oo_sq() {
    if constexpr (C == WHITE) return h1; else return h8;
}

template<Color C> inline Square rook_ooo_sq() {
    if constexpr (C == WHITE) return a1; else return a8;
}

template<Color C> inline Bitboard ignore_ooo_danger() {
    if constexpr (C == WHITE)
        return SQUARE_BB[b1];
    else
        return SQUARE_BB[b8];
}

// Direction enumeration
enum Direction {
    NORTH = 8,
    SOUTH = -8,
    NORTH_WEST = 7,
    NORTH_EAST = 9,
    NORTH_NORTH = 16,
    SOUTH_WEST = -9,
    SOUTH_EAST = -7,
    SOUTH_SOUTH = -16
};

// Relative direction and rank stubs
template<Color C>
constexpr Direction relative_dir(Direction d) {
    if constexpr (C == WHITE) return d;
    switch (d) {
        case NORTH: return SOUTH;
        case SOUTH: return NORTH;
        case NORTH_WEST: return SOUTH_EAST;
        case NORTH_EAST: return SOUTH_WEST;
        case NORTH_NORTH: return SOUTH_SOUTH;
        case SOUTH_WEST: return NORTH_EAST;
        case SOUTH_EAST: return NORTH_WEST;
        case SOUTH_SOUTH: return NORTH_NORTH;
    }
    return d;
}

template<Color C>
inline int relative_rank(int r) {
    return C == WHITE ? r : 7 - r;
}

// Utility stubs
inline int rank_of(Square s) { return s / 8; }
inline int file_of(Square s) { return s % 8; }
// Counts bits set
inline int sparse_pop_count(Bitboard bb) {
    return __builtin_popcountll(bb);
}

template<Direction D>
inline Bitboard shift(Bitboard bb) {
    // Input bitboard is already valid (64-bit), no masking needed
    
    if constexpr (D == NORTH) return (bb & 0x00FFFFFFFFFFFFFFULL) << 8;  // Mask out rank 8 to prevent overflow
    else if constexpr (D == SOUTH) return bb >> 8;
    else if constexpr (D == NORTH_EAST) return (bb & ~0x8080808080808080ULL) << 9;
    else if constexpr (D == NORTH_WEST) return (bb & ~0x0101010101010101ULL) << 7;
    else if constexpr (D == SOUTH_EAST) return (bb & ~0x8080808080808080ULL) >> 7;
    else if constexpr (D == SOUTH_WEST) return (bb & ~0x0101010101010101ULL) >> 9;
    else if constexpr (D == NORTH_NORTH) return (bb & 0x0000FFFFFFFFFFFFULL) << 16;  // Mask out ranks 7-8
    else /* SOUTH_SOUTH */ return bb >> 16;
}

inline int bsf(Bitboard bb) {
    return __builtin_ctzll(bb);
}

// Disabling duplicate pop_lsb definition
#if 0
inline int pop_lsb(Bitboard* bb) {
    int idx = __builtin_ctzll(*bb);
    *bb &= *bb - 1;
    // ... debug code disabled ...
    return idx;
}
#endif
