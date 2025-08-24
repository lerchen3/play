#include "position.h"
#include <sstream>
#include <vector>
#include <cctype>
#include <cassert>

void Position::clear() {
    for (int i = 0; i < NPIECES; ++i) piece_bb[i] = 0;
    for (int i = 0; i < NSQUARES; ++i) board[i] = NO_PIECE;
    side_to_play = WHITE;
    game_ply = 0;
    halfmove_clock = 0;
    hash = 0;
    pinned = 0;
    checkers = 0;
    for (int i = 0; i < NSQUARES; ++i) pin_line[i] = 0;
    ep_square = NO_SQUARE;
    state = PositionState();
    history_fens.clear();
}

// Moves a piece quietly (no capture)
void Position::move_piece_quiet(Square from, Square to) {
    Piece pc = board[from];
    assert(pc != NO_PIECE);
    remove_piece(from);
    put_piece(pc, to);
}

// Moves a piece, possibly capturing
void Position::move_piece(Square from, Square to) {
    if (board[to] != NO_PIECE)
        remove_piece(to);          // remove captured piece from board and bitboards
    move_piece_quiet(from, to);
}

// Sets position from FEN
void Position::set(const std::string& fen, Position& p) {
    std::string fen_str = fen;
    if (fen == "startpos")
        fen_str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    // Clear the position
    p.clear();

    std::stringstream ss(fen_str);
    std::string boardFEN, turnStr, castlingStr, epStr, hmStr, fmStr;
    ss >> boardFEN >> turnStr >> castlingStr >> epStr >> hmStr >> fmStr;

    // Parse board layout
    std::vector<std::string> ranks;
    std::stringstream boards(boardFEN);
    std::string rank;
    while (std::getline(boards, rank, '/'))
        ranks.push_back(rank);
    for (int ri = 0; ri < 8; ++ri) {
        int file = 0;
        const std::string& rstr = ranks[ri];
        for (char c : rstr) {
            if (std::isdigit(c)) {
                file += (c - '0');
            } else {
                // Direct mapping: FEN rank order top-to-bottom maps to row index 7->0
                Square sq = (7 - ri) * 8 + file;
                Piece pc = NO_PIECE;
                switch (c) {
                case 'P': pc = WHITE_PAWN; break;
                case 'N': pc = WHITE_KNIGHT; break;
                case 'B': pc = WHITE_BISHOP; break;
                case 'R': pc = WHITE_ROOK; break;
                case 'Q': pc = WHITE_QUEEN; break;
                case 'K': pc = WHITE_KING; break;
                case 'p': pc = BLACK_PAWN; break;
                case 'n': pc = BLACK_KNIGHT; break;
                case 'b': pc = BLACK_BISHOP; break;
                case 'r': pc = BLACK_ROOK; break;
                case 'q': pc = BLACK_QUEEN; break;
                case 'k': pc = BLACK_KING; break;
                default: pc = NO_PIECE; break;
                }
                p.put_piece(pc, sq);
                ++file;
            }
        }
    }

    // Side to move
    p.side_to_play = (turnStr == "w") ? WHITE : BLACK;

    // Castling rights -> entry bitboard
    p.state.entry = 0;
    // Mark king start square only if both rights are absent (king moved)
    if (castlingStr.find('K') == std::string::npos && castlingStr.find('Q') == std::string::npos)
        p.state.entry |= (1ULL << 4);                    // e1
    if (castlingStr.find('K') == std::string::npos)
        p.state.entry |= (1ULL << 7);                    // h1
    if (castlingStr.find('Q') == std::string::npos)
        p.state.entry |= (1ULL << 0);                    // a1
    if (castlingStr.find('k') == std::string::npos && castlingStr.find('q') == std::string::npos)
        p.state.entry |= (1ULL << 60);                   // e8
    if (castlingStr.find('k') == std::string::npos)
        p.state.entry |= (1ULL << 63);                   // h8
    if (castlingStr.find('q') == std::string::npos)
        p.state.entry |= (1ULL << 56);                   // a8

    // En passant target square
    if (epStr != "-") {
        int file = epStr[0] - 'a';
        int rank = epStr[1] - '1';
        p.ep_square = rank * 8 + file;
    } else {
        p.ep_square = NO_SQUARE;
    }


    // Halfmove clock and move number -> game ply
    p.halfmove_clock = hmStr.empty() ? 0 : std::stoi(hmStr);
    int fullmove = fmStr.empty() ? 1 : std::stoi(fmStr);
    p.game_ply = (fullmove - 1) * 2;
    if (p.side_to_play == BLACK)
        ++p.game_ply;
}

// Returns FEN string of the current position
std::string Position::fen() const {
    std::string out;

    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            Square sq = rank * 8 + file;
            Piece pc = board[sq];
            if (pc == NO_PIECE) {
                ++empty;
                continue;
            }
            if (empty) { out += std::to_string(empty); empty = 0; }
            char c = '.';
            switch (pc) {
            case WHITE_PAWN:   c = 'P'; break;
            case WHITE_KNIGHT: c = 'N'; break;
            case WHITE_BISHOP: c = 'B'; break;
            case WHITE_ROOK:   c = 'R'; break;
            case WHITE_QUEEN:  c = 'Q'; break;
            case WHITE_KING:   c = 'K'; break;
            case BLACK_PAWN:   c = 'p'; break;
            case BLACK_KNIGHT: c = 'n'; break;
            case BLACK_BISHOP: c = 'b'; break;
            case BLACK_ROOK:   c = 'r'; break;
            case BLACK_QUEEN:  c = 'q'; break;
            case BLACK_KING:   c = 'k'; break;
            default:           c = '.'; break;
            }
            out += c;
        }
        if (empty) out += std::to_string(empty);
        if (rank > 0) out += '/';
    }

    out += ' ';
    out += (side_to_play == WHITE ? 'w' : 'b');
    out += ' ';

    std::string castling;
    if ((state.entry & ((1ULL << 4) | (1ULL << 7))) == 0) castling += 'K';
    if ((state.entry & ((1ULL << 4) | (1ULL << 0))) == 0) castling += 'Q';
    if ((state.entry & ((1ULL << 60) | (1ULL << 63))) == 0) castling += 'k';
    if ((state.entry & ((1ULL << 60) | (1ULL << 56))) == 0) castling += 'q';
    if (castling.empty()) castling = "-";
    out += castling;

    out += ' ';
    if (ep_square == NO_SQUARE) {
        out += '-';
    } else {
        char f = 'a' + file_of(ep_square);
        char r = '1' + rank_of(ep_square);
        out += f;
        out += r;
    }

    out += ' ';
    out += std::to_string(halfmove_clock);
    out += ' ';
    int fullmove = game_ply / 2 + 1;
    out += std::to_string(fullmove);
    return out;
}

// Outputs an ASCII representation of the board
std::ostream& operator<<(std::ostream& os, const Position& p) {
    for (int rank = 0; rank < 8; ++rank) {  // Fixed: now goes 0->7 instead of 7->0
        for (int file = 0; file < 8; ++file) {
            Square sq = rank * 8 + file;
            Piece pc = p.board[sq];
            char c = '.';
            switch (pc) {
                case WHITE_PAWN:   c = 'P'; break;
                case WHITE_KNIGHT: c = 'N'; break;
                case WHITE_BISHOP: c = 'B'; break;
                case WHITE_ROOK:   c = 'R'; break;
                case WHITE_QUEEN:  c = 'Q'; break;
                case WHITE_KING:   c = 'K'; break;
                case BLACK_PAWN:   c = 'p'; break;
                case BLACK_KNIGHT: c = 'n'; break;
                case BLACK_BISHOP: c = 'b'; break;
                case BLACK_ROOK:   c = 'r'; break;
                case BLACK_QUEEN:  c = 'q'; break;
                case BLACK_KING:   c = 'k'; break;
                default:           c = '.'; break;
            }
            os << c;
        }
        os << "\n";
    }
    return os;
}
