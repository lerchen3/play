#pragma once

#include <cstdint>
#include <string>
#include <iostream>

// Number of squares and piece types
constexpr int NSQUARES = 64;
constexpr int NPIECES = 12;
// Maximum number of legal moves in any position
constexpr int MAX_MOVES = 218;

// Color enumeration
enum Color { WHITE = 0, BLACK = 1 };
constexpr Color operator~(Color c) { return c == WHITE ? BLACK : WHITE; }

// Bitboard and square types
using Bitboard = uint64_t;
using Square = int;

// Board square constants a1-h8 encoded as 0-63
enum : Square {
    a1, b1, c1, d1, e1, f1, g1, h1,
    a2, b2, c2, d2, e2, f2, g2, h2,
    a3, b3, c3, d3, e3, f3, g3, h3,
    a4, b4, c4, d4, e4, f4, g4, h4,
    a5, b5, c5, d5, e5, f5, g5, h5,
    a6, b6, c6, d6, e6, f6, g6, h6,
    a7, b7, c7, d7, e7, f7, g7, h7,
    a8, b8, c8, d8, e8, f8, g8, h8
};

// Rank constants for convenience
enum Rank { RANK1, RANK2, RANK3, RANK4, RANK5, RANK6, RANK7, RANK8 };

// Add Piece enumeration, NO_SQUARE sentinel, and make_piece helper
enum PieceType { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };

enum Piece {
    NO_PIECE = -1,
    WHITE_PAWN = 0, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
    BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING
};
constexpr Piece make_piece(Color c, PieceType pt) {
    return static_cast<Piece>(int(pt) + (c == WHITE ? 0 : 6));
}
constexpr Square NO_SQUARE = -1;

// Move flags
enum MoveFlags { QUIET, DOUBLE_PUSH, OO, OOO, PR_KNIGHT, PR_BISHOP, PR_ROOK, PR_QUEEN, PC_KNIGHT, PC_BISHOP, PC_ROOK, PC_QUEEN, PROMOTION_CAPTURES, CAPTURE };

// Move struct with UCI conversion
struct Move {
    Square from_sq, to_sq;
    MoveFlags flag;
    Move() : from_sq(0), to_sq(0), flag(QUIET) {}
    Move(Square f, Square t, MoveFlags fl) : from_sq(f), to_sq(t), flag(fl) {}
    Square from() const { return from_sq; }
    Square to() const { return to_sq; }
    MoveFlags flags() const { return flag; }
    std::string uci() const {
        Square from = from_sq;
        Square to = to_sq;
        
        // CRITICAL FIX: Validate squares before UCI conversion
        if (from < 0 || from >= 64 || to < 0 || to >= 64) {
            return "@@@@";  // Invalid UCI string that won't be in MOVE_INDEX
        }

        // Castling moves store the rook square in 'to_sq'.
        // Convert to king destination for UCI notation.
        if (flag == OO) {
            to = (from_sq == 4 ? 6  /* e1 -> g1 */
                               : 62 /* e8 -> g8 */);
        } else if (flag == OOO) {
            to = (from_sq == 4 ? 2  /* e1 -> c1 */
                               : 58 /* e8 -> c8 */);
        }
        
        // Double-check after castling conversion
        if (from < 0 || from >= 64 || to < 0 || to >= 64) {
            return "@@@@";  // Invalid UCI string
        }

        char f = 'a' + (from % 8);
        char r = '1' + (from / 8);
        char t = 'a' + (to % 8);
        char s = '1' + (to / 8);
        std::string s_uci;
        s_uci += f; s_uci += r; s_uci += t; s_uci += s;
        switch (flag) {
            case PR_KNIGHT: case PC_KNIGHT: s_uci += 'n'; break;
            case PR_BISHOP: case PC_BISHOP: s_uci += 'b'; break;
            case PR_ROOK:   case PC_ROOK:   s_uci += 'r'; break;
            case PR_QUEEN:  case PC_QUEEN:  s_uci += 'q'; break;
            default: break;
        }
        return s_uci;
    }
};

// Helper to get piece type from a piece enum
constexpr PieceType type_of(Piece p) { return PieceType(p % 6); }

// Generic move constructor used by move generation
template<MoveFlags Flag>
inline Move* make(Square from, Bitboard to_bb, Move* list) {
    while (to_bb) {
        Square to = __builtin_ctzll(to_bb);
        to_bb &= to_bb - 1;
        
        // CRITICAL FIX: Validate squares before creating moves
        if (from < 0 || from >= 64 || to < 0 || to >= 64) {
            // Silently skip invalid moves to avoid spam
            continue;  // Skip this move, continue with next bit
        }
        
        // CRITICAL DEBUG: Alert when we find the a9 moves!
        if (from == 64 || to == 64) {
            std::cerr << "*** FOUND A9 MOVE SOURCE! ***" << std::endl;
            std::cerr << "from=" << from << " to=" << to << std::endl;
            std::cerr << "UCI would be: " << char('a' + (from % 8)) << (1 + from / 8) 
                      << char('a' + (to % 8)) << (1 + to / 8) << std::endl;
            std::cerr << "to_bb before pop: " << std::hex << (to_bb | (1ULL << to)) << std::dec << std::endl;
            std::cerr << "Flag: " << Flag << std::endl;
            continue;  // Skip creating this invalid move
        }
        
        // ADDITIONAL SAFETY: Double-check square validity before Move constructor
        if (from < 0 || from >= 64 || to < 0 || to >= 64) {
            std::cerr << "*** DOUBLE-CHECK FAILED: Still invalid! ***" << std::endl;
            continue;
        }
        
        if constexpr (Flag == PROMOTION_CAPTURES) {
            // Only create promotion moves if squares are valid
            if (from >= 0 && from < 64 && to >= 0 && to < 64) {
                *list++ = Move(from, to, PC_KNIGHT);
                *list++ = Move(from, to, PC_BISHOP);
                *list++ = Move(from, to, PC_ROOK);
                *list++ = Move(from, to, PC_QUEEN);
            }
        } else {
            // Only create move if squares are valid
            if (from >= 0 && from < 64 && to >= 0 && to < 64) {
                *list++ = Move(from, to, Flag);
            }
        }
    }
    return list;
}

inline Square pop_lsb(Bitboard* b) {
    if (*b == 0) {
        return NO_SQUARE;  // Empty bitboard
    }
    
    Square s = __builtin_ctzll(*b);
    *b &= *b - 1;
    
    // CRITICAL FIX: Validate square is in valid range
    if (s < 0 || s >= 64) {
        return NO_SQUARE;  // Invalid square
    }
    
    return s;
}
