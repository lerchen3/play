#pragma once

#include "types.h"
#include <ostream>
#include <string>
#include "tables.h"
#include <utility>
#include <deque>

//A pseudorandom number generator
//Source: Stockfish
class PRNG {
	uint64_t s;

	uint64_t rand64() {
		s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
		return s * 2685821657736338717LL;
	}

public:
	PRNG(uint64_t seed) : s(seed) {}

        //Generate pseudorandom number
	template<typename T> T rand() { return T(rand64()); }

        //Generate pseudorandom number with only a few set bits
	template<typename T> 
	T sparse_rand() {
		return T(rand64() & rand64() & rand64());
	}
};


namespace zobrist {
	extern uint64_t zobrist_table[NPIECES][NSQUARES];
	extern void initialise_zobrist_keys();
}

//Stores position information which cannot be recovered on undo-ing a move
struct PositionState {
        //The bitboard of squares on which pieces have either moved from, or have been moved to. Used for castling
        //legality checks
        Bitboard entry;

        constexpr PositionState() : entry(0) {}
};

class Position {
private:
	//A bitboard of the locations of each piece
	Bitboard piece_bb[NPIECES];
	
	//A mailbox representation of the board. Stores the piece occupying each square on the board
	Piece board[NSQUARES];
	
	//The side whose turn it is to play next
	Color side_to_play;
	
	//The current game ply (depth), incremented after each move 
        int game_ply;
        int halfmove_clock;
	
	//The zobrist hash of the position, which can be incrementally updated and rolled back after each
	//make/unmake
	uint64_t hash;
public:
        //Additional state required for castling
        PositionState state;
	
        //The bitboard of enemy pieces that are currently attacking the king, updated whenever generate_moves()
        //is called
        mutable Bitboard checkers;

        //The bitboard of pieces that are currently pinned to the king by enemy sliders, updated whenever
        //generate_moves() is called
        mutable Bitboard pinned;

        // For each square, if pinned, the line from our king through the pinner
        // used to restrict legal moves of that piece
        mutable Bitboard pin_line[NSQUARES];

        // En passant target square or NO_SQUARE if none
        Square ep_square;

        std::deque<std::string> history_fens;
	
	
        Position() : piece_bb{ 0 }, board{}, side_to_play(WHITE), game_ply(0), halfmove_clock(0),
                hash(0), state(), checkers(0), pinned(0), ep_square(NO_SQUARE), history_fens() {

                //Sets all squares on the board as empty
                for (int i = 0; i < 64; i++) {
                        board[i] = NO_PIECE;
                        pin_line[i] = 0;
                }
        }

        // Reset the position to the default starting state
        void clear();
	
	//Places a piece on a particular square and updates the hash. Placing a piece on a square that is 
	//already occupied is an error
	inline void put_piece(Piece pc, Square s) {
		board[s] = pc;
		piece_bb[pc] |= SQUARE_BB[s];
		hash ^= zobrist::zobrist_table[pc][s];
	}

	//Removes a piece from a particular square and updates the hash. 
	inline void remove_piece(Square s) {
		hash ^= zobrist::zobrist_table[board[s]][s];
		piece_bb[board[s]] &= ~SQUARE_BB[s];
		board[s] = NO_PIECE;
	}

	void move_piece(Square from, Square to);
	void move_piece_quiet(Square from, Square to);


	friend std::ostream& operator<<(std::ostream& os, const Position& p);
	static void set(const std::string& fen, Position& p);
	std::string fen() const;

	Position& operator=(const Position&) = delete;
	inline bool operator==(const Position& other) const { return hash == other.hash; }

	inline Bitboard bitboard_of(Piece pc) const { return piece_bb[pc]; }
	inline Bitboard bitboard_of(Color c, PieceType pt) const { return piece_bb[make_piece(c, pt)]; }
	inline Piece at(Square sq) const { return board[sq]; }
	inline Color turn() const { return side_to_play; }
	inline int ply() const { return game_ply; }
	inline uint64_t get_hash() const { return hash; }

	template<Color C> inline Bitboard diagonal_sliders() const;
	template<Color C> inline Bitboard orthogonal_sliders() const;
	template<Color C> inline Bitboard all_pieces() const;
	template<Color C> inline Bitboard attackers_from(Square s, Bitboard occ) const;

	template<Color C> inline bool in_check() const {
		return attackers_from<~C>(bsf(bitboard_of(C, KING)), all_pieces<WHITE>() | all_pieces<BLACK>());
	}

        template<Color C> void play(Move m);

	// Non-templated wrappers for convenience
        void play(Move m) {
                if (side_to_play == WHITE) play<WHITE>(m);
                else                         play<BLACK>(m);
        }

        template<Color Us>
        Move *generate_legals(Move* list) const;

	// Game termination: legal moves and result
	inline bool hasLegalMoves() const {
                Move list[MAX_MOVES];
		if (turn() == WHITE) {
			return (generate_legals<WHITE>(list) - list) > 0;
		} else {
			return (generate_legals<BLACK>(list) - list) > 0;
		}
	}

	inline bool isGameOver() const {
		return !hasLegalMoves();
	}

        inline int result() const {
                if (hasLegalMoves()) return 0;
		bool check = (turn() == WHITE) ? in_check<WHITE>() : in_check<BLACK>();
		if (check) {
			return (turn() == WHITE) ? -1 : 1;
		} else {
			return 0;
		}
	}
};

//Returns the bitboard of all bishops and queens of a given color
template<Color C> 
inline Bitboard Position::diagonal_sliders() const {
	return C == WHITE ? piece_bb[WHITE_BISHOP] | piece_bb[WHITE_QUEEN] :
		piece_bb[BLACK_BISHOP] | piece_bb[BLACK_QUEEN];
}

//Returns the bitboard of all rooks and queens of a given color
template<Color C> 
inline Bitboard Position::orthogonal_sliders() const {
	return C == WHITE ? piece_bb[WHITE_ROOK] | piece_bb[WHITE_QUEEN] :
		piece_bb[BLACK_ROOK] | piece_bb[BLACK_QUEEN];
}

//Returns a bitboard containing all the pieces of a given color
template<Color C> 
inline Bitboard Position::all_pieces() const {
	return C == WHITE ? piece_bb[WHITE_PAWN] | piece_bb[WHITE_KNIGHT] | piece_bb[WHITE_BISHOP] |
		piece_bb[WHITE_ROOK] | piece_bb[WHITE_QUEEN] | piece_bb[WHITE_KING] :

		piece_bb[BLACK_PAWN] | piece_bb[BLACK_KNIGHT] | piece_bb[BLACK_BISHOP] |
		piece_bb[BLACK_ROOK] | piece_bb[BLACK_QUEEN] | piece_bb[BLACK_KING];
}

//Returns a bitboard containing all pieces of a given color attacking a particluar square
template<Color C> 
inline Bitboard Position::attackers_from(Square s, Bitboard occ) const {
	return C == WHITE ? (pawn_attacks<BLACK>(s) & piece_bb[WHITE_PAWN]) |
		(attacks<KNIGHT>(s, occ) & piece_bb[WHITE_KNIGHT]) |
		(attacks<BISHOP>(s, occ) & (piece_bb[WHITE_BISHOP] | piece_bb[WHITE_QUEEN])) |
		(attacks<ROOK>(s, occ) & (piece_bb[WHITE_ROOK] | piece_bb[WHITE_QUEEN])) :

		(pawn_attacks<WHITE>(s) & piece_bb[BLACK_PAWN]) |
		(attacks<KNIGHT>(s, occ) & piece_bb[BLACK_KNIGHT]) |
		(attacks<BISHOP>(s, occ) & (piece_bb[BLACK_BISHOP] | piece_bb[BLACK_QUEEN])) |
		(attacks<ROOK>(s, occ) & (piece_bb[BLACK_ROOK] | piece_bb[BLACK_QUEEN]));
}


/*template<Color C>
Bitboard Position::pinned(Square s, Bitboard us, Bitboard occ) const {
	Bitboard pinned = 0;

	Bitboard pinners = get_xray_rook_attacks(s, occ, us) & orthogonal_sliders<~C>();
	while (pinners) pinned |= SQUARES_BETWEEN_BB[s][pop_lsb(&pinners)] & us;

	pinners = get_xray_bishop_attacks(s, occ, us) & diagonal_sliders<~C>();
	while (pinners) pinned |= SQUARES_BETWEEN_BB[s][pop_lsb(&pinners)] & us;

	return pinned;
}

template<Color C>
Bitboard Position::blockers_to(Square s, Bitboard occ) const {
	Bitboard blockers = 0;
	Bitboard candidates = get_rook_attacks(s, occ) & occ;
	Bitboard attackers = get_rook_attacks(s, occ ^ candidates) & orthogonal_sliders<~C>();

	candidates = get_bishop_attacks(s, occ) & occ;
	attackers |= get_bishop_attacks(s, occ ^ candidates) & diagonal_sliders<~C>();

	while (attackers) blockers |= SQUARES_BETWEEN_BB[s][pop_lsb(&attackers)];
	return blockers;
}*/

//Plays a move in the position
template<Color C>
void Position::play(const Move m) {
        if(history_fens.size() >= 7) history_fens.pop_front();
        history_fens.push_back(fen());
        side_to_play = ~side_to_play;
        ++game_ply;

        MoveFlags type = m.flags();
        Piece moving_piece = board[m.from()];
        bool capture_move = board[m.to()] != NO_PIECE || type == CAPTURE || type == PC_KNIGHT || type == PC_BISHOP || type == PC_ROOK || type == PC_QUEEN;
        bool pawn_move = type_of(moving_piece) == PAWN;
        state.entry |= SQUARE_BB[m.to()] | SQUARE_BB[m.from()];
        Square old_ep = ep_square;
        ep_square = NO_SQUARE;

	switch (type) {
	case QUIET:
		//The to square is guaranteed to be empty here
		move_piece_quiet(m.from(), m.to());
		break;
        case DOUBLE_PUSH:
                //The to square is guaranteed to be empty here
                move_piece_quiet(m.from(), m.to());
                ep_square = m.from() + relative_dir<C>(NORTH);
                break;
	case OO:
		if (C == WHITE) {
			move_piece_quiet(e1, g1);
			move_piece_quiet(h1, f1);
		} else {
			move_piece_quiet(e8, g8);
			move_piece_quiet(h8, f8);
		}			
		break;
	case OOO:
		if (C == WHITE) {
			move_piece_quiet(e1, c1); 
			move_piece_quiet(a1, d1);
		} else {
			move_piece_quiet(e8, c8);
			move_piece_quiet(a8, d8);
		}
		break;
	case PR_KNIGHT:
		remove_piece(m.from());
		put_piece(make_piece(C, KNIGHT), m.to());
		break;
	case PR_BISHOP:
		remove_piece(m.from());
		put_piece(make_piece(C, BISHOP), m.to());
		break;
	case PR_ROOK:
		remove_piece(m.from());
		put_piece(make_piece(C, ROOK), m.to());
		break;
	case PR_QUEEN:
		remove_piece(m.from());
		put_piece(make_piece(C, QUEEN), m.to());
		break;
        case PC_KNIGHT:
                remove_piece(m.from());
                remove_piece(m.to());
                put_piece(make_piece(C, KNIGHT), m.to());
                break;
        case PC_BISHOP:
                remove_piece(m.from());
                remove_piece(m.to());
                put_piece(make_piece(C, BISHOP), m.to());
                break;
        case PC_ROOK:
                remove_piece(m.from());
                remove_piece(m.to());
                put_piece(make_piece(C, ROOK), m.to());
                break;
        case PC_QUEEN:
                remove_piece(m.from());
                remove_piece(m.to());
                put_piece(make_piece(C, QUEEN), m.to());
                break;
        case PROMOTION_CAPTURES:
                // Should never be generated directly; handled via PC_* cases
                break;
        case CAPTURE:
                if (type_of(board[m.from()]) == PAWN && board[m.to()] == NO_PIECE && m.to() == old_ep) {
                        // en passant capture
                        Square captured_pawn = m.to() - relative_dir<C>(NORTH);
                        if (captured_pawn >= 0 && captured_pawn < 64) {
                                remove_piece(captured_pawn);
                        }
                        move_piece_quiet(m.from(), m.to());
                } else {
                        move_piece(m.from(), m.to());
                }
                break;
        }

        if (pawn_move || capture_move)
                halfmove_clock = 0;
        else
                ++halfmove_clock;
}

//Generates all legal moves in a position for the given side. Advances the move pointer and returns it.
template<Color Us>
Move* Position::generate_legals(Move* list) const {
        constexpr Color Them = ~Us;

        auto add_move = [&list](Square from, Square to, MoveFlags flag) {
                // CRITICAL FIX: Validate squares before creating moves
                if (from < 0 || from >= 64 || to < 0 || to >= 64) {
                        return;  // Don't create invalid moves
                }
                
                // ADDITIONAL CHECK: Specifically catch the A9 pattern at the lambda level
                if (from == 64 || to == 64) {
                        std::cerr << "*** BLOCKED A9 MOVE IN ADD_MOVE LAMBDA! ***" << std::endl;
                        std::cerr << "from=" << from << " to=" << to << " flag=" << flag << std::endl;
                        return;  // Don't create A9 moves
                }
                
                *list++ = Move(from, to, flag);
        };

        auto add_promotion_moves = [&](Square from, Square to, bool is_capture) {
                // CRITICAL FIX: Explicit validation for promotion moves
                if (from < 0 || from >= 64 || to < 0 || to >= 64) {
                        std::cerr << "*** INVALID PROMOTION MOVE SQUARES! ***" << std::endl;
                        std::cerr << "from=" << from << " to=" << to << " is_capture=" << is_capture << std::endl;
                        return;  // Don't create invalid promotion moves
                }
                
                // ADDITIONAL CHECK: Specifically catch the A9 pattern in promotions
                if (from == 64 || to == 64) {
                        std::cerr << "*** BLOCKED A9 PROMOTION MOVE! ***" << std::endl;
                        std::cerr << "from=" << from << " to=" << to << " is_capture=" << is_capture << std::endl;
                        return;  // Don't create A9 promotion moves
                }
                
                if (is_capture) {
                        *list++ = Move(from, to, PC_KNIGHT);
                        *list++ = Move(from, to, PC_BISHOP);
                        *list++ = Move(from, to, PC_ROOK);
                        *list++ = Move(from, to, PC_QUEEN);
                } else {
                        *list++ = Move(from, to, PR_KNIGHT);
                        *list++ = Move(from, to, PR_BISHOP);
                        *list++ = Move(from, to, PR_ROOK);
                        *list++ = Move(from, to, PR_QUEEN);
                }
        };

        for (int i = 0; i < NSQUARES; ++i) pin_line[i] = 0;

	const Bitboard us_bb = all_pieces<Us>();
	const Bitboard them_bb = all_pieces<Them>();
	const Bitboard all = us_bb | them_bb;

	const Square our_king = bsf(bitboard_of(Us, KING));
	const Square their_king = bsf(bitboard_of(Them, KING));

	const Bitboard our_diag_sliders = diagonal_sliders<Us>();
	const Bitboard their_diag_sliders = diagonal_sliders<Them>();
	const Bitboard our_orth_sliders = orthogonal_sliders<Us>();
	const Bitboard their_orth_sliders = orthogonal_sliders<Them>();

	//General purpose bitboards for attacks, masks, etc.
	Bitboard b1, b2, b3;
	
	//Squares that our king cannot move to
	Bitboard danger = 0;

	//For each enemy piece, add all of its attacks to the danger bitboard
	danger |= pawn_attacks<Them>(bitboard_of(Them, PAWN)) | attacks<KING>(their_king, all);
	
	b1 = bitboard_of(Them, KNIGHT); 
	while (b1) danger |= attacks<KNIGHT>(pop_lsb(&b1), all);
	
	b1 = their_diag_sliders;
	//all ^ SQUARE_BB[our_king] is written to prevent the king from moving to squares which are 'x-rayed'
	//by enemy bishops and queens
	while (b1) danger |= attacks<BISHOP>(pop_lsb(&b1), all ^ SQUARE_BB[our_king]);
	
	b1 = their_orth_sliders;
	//all ^ SQUARE_BB[our_king] is written to prevent the king from moving to squares which are 'x-rayed'
	//by enemy rooks and queens
	while (b1) danger |= attacks<ROOK>(pop_lsb(&b1), all ^ SQUARE_BB[our_king]);

	//The king can move to all of its surrounding squares, except ones that are attacked, and
	//ones that have our own pieces on them
	b1 = attacks<KING>(our_king, all) & ~(us_bb | danger);
	
	// A general purpose square for king moves
	Square s;
	
	// King quiet moves
	Bitboard king_quiet = b1 & ~them_bb;
	while (king_quiet) {
		s = pop_lsb(&king_quiet);
		if (s != NO_SQUARE) add_move(our_king, s, QUIET);
	}
	// King captures
	Bitboard king_captures = b1 & them_bb;
	while (king_captures) {
		s = pop_lsb(&king_captures);
		if (s != NO_SQUARE) add_move(our_king, s, CAPTURE);
	}

	//The capture mask filters destination squares to those that contain an enemy piece that is checking the 
	//king and must be captured
	Bitboard capture_mask;
	
	//The quiet mask filter destination squares to those where pieces must be moved to block an incoming attack 
	//to the king
	Bitboard quiet_mask;

	//Checkers of each piece type are identified by:
	//1. Projecting attacks FROM the king square
	//2. Intersecting this bitboard with the enemy bitboard of that piece type
        checkers = (attacks<KNIGHT>(our_king, all) & bitboard_of(Them, KNIGHT))
                | (pawn_attacks<Us>(our_king) & bitboard_of(Them, PAWN));
	
	//Here, we identify slider checkers and pinners simultaneously, and candidates for such pinners 
	//and checkers are represented by the bitboard <candidates>
        Bitboard candidates = (attacks<ROOK>(our_king, them_bb) & their_orth_sliders)
                | (attacks<BISHOP>(our_king, them_bb) & their_diag_sliders);

        pinned = 0;
        while (candidates) {
                s = pop_lsb(&candidates);
                b1 = SQUARES_BETWEEN_BB[our_king][s] & us_bb;

                //Do the squares in between the enemy slider and our king contain any of our pieces?
                //If not, add the slider to the checker bitboard
                if (b1 == 0) checkers ^= SQUARE_BB[s];
                //If there is only one of our pieces between them, record pin line
                else if ((b1 & (b1 - 1)) == 0) {
                        pinned |= b1; // not strictly necessary since each piece can only be pinned by at most one thing (u could do ^) but whatever
                        pin_line[bsf(b1)] = LINE[our_king][s];
                }
        }

	//This makes it easier to mask pieces
	const Bitboard not_pinned = ~pinned;

	switch (sparse_pop_count(checkers)) {
	case 2:
		//If there is a double check, the only legal moves are king moves out of check
		return list;
	case 1: {
		//It's a single check!
		
		Square checker_square = bsf(checkers);

		switch (board[checker_square]) {
                case make_piece(Them, PAWN):
                case make_piece(Them, KNIGHT):
                        // If the checker is either a pawn or a knight, the only legal moves
                        // are to capture the checker. Only non-pinned pieces can capture it.
						// a pawn on the 7th rank can CAPTURE WITH PROMOTION A KNIGHT THAT IS CHECKING THE KING
                        b1 = attackers_from<Us>(checker_square, all) & not_pinned;
                        while (b1) {
                                Square from = pop_lsb(&b1);
                                if (board[from] == make_piece(Us, PAWN) &&
                                    rank_of(from) == relative_rank<Us>(RANK7)) {
                                        add_promotion_moves(from, checker_square, true);
                                } else {
                                        add_move(from, checker_square, CAPTURE);
                                }
                        }

                        // Special case: en passant capture of a pawn giving check
                        if (board[checker_square] == make_piece(Them, PAWN) &&
                            ep_square != NO_SQUARE &&
                            checker_square == ep_square - relative_dir<Us>(NORTH)) {
                                Square from_left = ep_square - relative_dir<Us>(NORTH_WEST);
                                Square from_right = ep_square - relative_dir<Us>(NORTH_EAST);
                                
                                if ((bitboard_of(Us, PAWN) & SQUARE_BB[from_left])) {
                                        if ((not_pinned & SQUARE_BB[from_left]) || (pin_line[from_left] & SQUARE_BB[ep_square]))
                                                add_move(from_left, ep_square, CAPTURE);
                                }
                                if ((bitboard_of(Us, PAWN) & SQUARE_BB[from_right])) {
                                        if ((not_pinned & SQUARE_BB[from_right]) || (pin_line[from_right] & SQUARE_BB[ep_square]))
                                                add_move(from_right, ep_square, CAPTURE);
                                }
                        }

                        return list;
		default:
			//We must capture the checking piece
			capture_mask = checkers;
			
			//...or we can block it since it is guaranteed to be a slider
			quiet_mask = SQUARES_BETWEEN_BB[our_king][checker_square];
			break;
		}

		break;
	}

	default:
		//We can capture any enemy piece
		capture_mask = them_bb;
		
		//...and we can play a quiet move to any square which is not occupied
		quiet_mask = ~all;


		//Only add castling if:
		//1. The king and the rook have both not moved
		//2. No piece is attacking between the the rook and the king
		//3. The king is not in check
                if (!((state.entry & oo_mask<Us>()) | ((all | danger) & oo_blockers_mask<Us>()))) {
                        if (board[rook_oo_sq<Us>()] == make_piece(Us, ROOK))
                                add_move(Us == WHITE ? e1 : e8, Us == WHITE ? h1 : h8, OO);
                }
                if (!((state.entry & ooo_mask<Us>()) |
                        ((all | (danger & ~ignore_ooo_danger<Us>())) & ooo_blockers_mask<Us>()))) {
                        if (board[rook_ooo_sq<Us>()] == make_piece(Us, ROOK))
                                add_move(Us == WHITE ? e1 : e8, Us == WHITE ? c1 : c8, OOO);
                }

		//For each pinned rook, bishop or queen...
                b1 = ~(not_pinned | bitboard_of(Us, KNIGHT));
                while (b1) {
                        s = pop_lsb(&b1);

                        //...only include attacks that are aligned with our king, since pinned pieces
                        //are constrained to move in this direction only
                        b2 = attacks(type_of(board[s]), s, all) & pin_line[s];
                        // Pinned piece quiet moves
                        Bitboard pinned_quiet = b2 & quiet_mask;
                        while (pinned_quiet) {
                                Square to = pop_lsb(&pinned_quiet);
                                if (to != NO_SQUARE) add_move(s, to, QUIET);
                        }
                        // Pinned piece captures
                        Bitboard pinned_captures = b2 & capture_mask;
                        while (pinned_captures) {
                                Square to = pop_lsb(&pinned_captures);
                                if (to != NO_SQUARE) add_move(s, to, CAPTURE);
                        }
                }

		//For each pinned pawn...
		b1 = ~not_pinned & bitboard_of(Us, PAWN);
		while (b1) {
			s = pop_lsb(&b1);

			if (rank_of(s) == relative_rank<Us>(RANK7)) {
				//Quiet promotions are impossible since the square in front of the pawn will
				//either be occupied by the king or the pinner, or doing so would leave our king
				//in check
                                b2 = pawn_attacks<Us>(s) & capture_mask & pin_line[s];
				// Pinned pawn promotion captures
				while (b2) {
					Square to = pop_lsb(&b2);
					if (to != NO_SQUARE) add_promotion_moves(s, to, true);
				}
			}
			else {
                                b2 = pawn_attacks<Us>(s) & them_bb & pin_line[s];
				// Pinned pawn captures
				while (b2) {
					Square to = pop_lsb(&b2);
					if (to != NO_SQUARE) add_move(s, to, CAPTURE);
				}
				
				//Single pawn pushes
                                b2 = shift<relative_dir<Us>(NORTH)>(SQUARE_BB[s]) & ~all & pin_line[s];
				//Double pawn pushes (only pawns on rank 3/6 are eligible)
                                b3 = shift<relative_dir<Us>(NORTH)>(b2 &
                                        MASK_RANK[relative_rank<Us>(RANK3)]) & ~all & pin_line[s];
				// Pinned pawn single pushes
				while (b2) {
					Square to = pop_lsb(&b2);
					if (to != NO_SQUARE) add_move(s, to, QUIET);
				}
				// Pinned pawn double pushes
				while (b3) {
					Square to = pop_lsb(&b3);
					if (to != NO_SQUARE) add_move(s, to, DOUBLE_PUSH);
				}
			}
		}
		
		//Pinned knights cannot move anywhere, so we're done with pinned pieces!

		break;
	}

	//Non-pinned knight moves
	b1 = bitboard_of(Us, KNIGHT) & not_pinned;
	while (b1) {
		s = pop_lsb(&b1);
		if (s == NO_SQUARE) continue;
		b2 = attacks<KNIGHT>(s, all);
		// Knight quiet moves
		Bitboard knight_quiet = b2 & quiet_mask;
		while (knight_quiet) {
			Square to = pop_lsb(&knight_quiet);
			if (to != NO_SQUARE) add_move(s, to, QUIET);
		}
		// Knight captures
		Bitboard knight_captures = b2 & capture_mask;
		while (knight_captures) {
			Square to = pop_lsb(&knight_captures);
			if (to != NO_SQUARE) add_move(s, to, CAPTURE);
		}
	}

	//Non-pinned bishops and queens
	b1 = our_diag_sliders & not_pinned;
	while (b1) {
		s = pop_lsb(&b1);
		if (s == NO_SQUARE) continue;
		b2 = attacks<BISHOP>(s, all);
		// Bishop/Queen quiet moves
		Bitboard diag_quiet = b2 & quiet_mask;
		while (diag_quiet) {
			Square to = pop_lsb(&diag_quiet);
			if (to != NO_SQUARE) add_move(s, to, QUIET);
		}
		// Bishop/Queen captures
		Bitboard diag_captures = b2 & capture_mask;
		while (diag_captures) {
			Square to = pop_lsb(&diag_captures);
			if (to != NO_SQUARE) add_move(s, to, CAPTURE);
		}
	}

	//Non-pinned rooks and queens
	b1 = our_orth_sliders & not_pinned;
	while (b1) {
		s = pop_lsb(&b1);
		if (s == NO_SQUARE) continue;
		b2 = attacks<ROOK>(s, all);
		// Rook/Queen quiet moves
		Bitboard orth_quiet = b2 & quiet_mask;
		while (orth_quiet) {
			Square to = pop_lsb(&orth_quiet);
			if (to != NO_SQUARE) add_move(s, to, QUIET);
		}
		// Rook/Queen captures
		Bitboard orth_captures = b2 & capture_mask;
		while (orth_captures) {
			Square to = pop_lsb(&orth_captures);
			if (to != NO_SQUARE) add_move(s, to, CAPTURE);
		}
	}

	//b1 contains non-pinned pawns which are not on the last rank
	b1 = bitboard_of(Us, PAWN) & not_pinned & ~MASK_RANK[relative_rank<Us>(RANK7)];
	
	//Single pawn pushes
	b2 = shift<relative_dir<Us>(NORTH)>(b1) & ~all;
	
	//Double pawn pushes (only pawns on rank 3/6 are eligible)
	// The previous version attempted to mask with LINE[our_king][s] but
	// `s` is not defined in this context, leading to undefined behaviour
	// and an empty set of double pawn push moves.  Simply generate the
	// candidate squares without the erroneous mask.
	b3 = shift<relative_dir<Us>(NORTH)>(b2 & MASK_RANK[relative_rank<Us>(RANK3)]) & ~all;
	
	//We & this with the quiet mask only later, as a non-check-blocking single push does NOT mean that the 
	//corresponding double push is not blocking check either.
	b2 &= quiet_mask;

	while (b2) {
		s = pop_lsb(&b2);
		Square from = s - relative_dir<Us>(NORTH);
		
		// CRITICAL FIX: Validate calculated source square
		if (from < 0 || from >= 64 || s < 0 || s >= 64) {
			std::cerr << "*** INVALID PAWN PUSH SOURCE! ***" << std::endl;
			std::cerr << "calculated from=" << from << " to=" << s << std::endl;
			std::cerr << "Direction offset: " << relative_dir<Us>(NORTH) << std::endl;
			continue;  // Skip this invalid move
		}
		
		add_move(from, s, QUIET);
	}

	while (b3) {
		s = pop_lsb(&b3);
		Square from = s - relative_dir<Us>(NORTH_NORTH);
		
		// CRITICAL FIX: Validate calculated source square for double pushes
		if (from < 0 || from >= 64 || s < 0 || s >= 64) {
			std::cerr << "*** INVALID PAWN DOUBLE PUSH SOURCE! ***" << std::endl;
			std::cerr << "calculated from=" << from << " to=" << s << std::endl;
			std::cerr << "Direction offset: " << relative_dir<Us>(NORTH_NORTH) << std::endl;
			continue;  // Skip this invalid move
		}
		
		add_move(from, s, DOUBLE_PUSH);
	}

	//Pawn captures
	b2 = shift<relative_dir<Us>(NORTH_WEST)>(b1) & capture_mask;
	b3 = shift<relative_dir<Us>(NORTH_EAST)>(b1) & capture_mask;

	while (b2) {
		s = pop_lsb(&b2);
		Square from = s - relative_dir<Us>(NORTH_WEST);
		
		// CRITICAL FIX: Validate calculated source square for captures
		if (from < 0 || from >= 64 || s < 0 || s >= 64) {
			std::cerr << "*** INVALID PAWN CAPTURE WEST SOURCE! ***" << std::endl;
			std::cerr << "calculated from=" << from << " to=" << s << std::endl;
			std::cerr << "Direction offset: " << relative_dir<Us>(NORTH_WEST) << std::endl;
			continue;  // Skip this invalid move
		}
		
		add_move(from, s, CAPTURE);
	}

        while (b3) {
                s = pop_lsb(&b3);
                Square from = s - relative_dir<Us>(NORTH_EAST);
                
                // CRITICAL FIX: Validate calculated source square for captures
                if (from < 0 || from >= 64 || s < 0 || s >= 64) {
                        std::cerr << "*** INVALID PAWN CAPTURE EAST SOURCE! ***" << std::endl;
                        std::cerr << "calculated from=" << from << " to=" << s << std::endl;
                        std::cerr << "Direction offset: " << relative_dir<Us>(NORTH_EAST) << std::endl;
                        continue;  // Skip this invalid move
                }
                
                add_move(from, s, CAPTURE);
        }

        // En passant captures
        if (ep_square != NO_SQUARE) {
                Square cap_sq = ep_square - relative_dir<Us>(NORTH);
                if (board[cap_sq] == make_piece(Them, PAWN)) {
                        Square from_left = ep_square - relative_dir<Us>(NORTH_WEST);
                        Square from_right = ep_square - relative_dir<Us>(NORTH_EAST);
                        
                        if ((bitboard_of(Us, PAWN) & SQUARE_BB[from_left])) {
                                if ((not_pinned & SQUARE_BB[from_left]) || (pin_line[from_left] & SQUARE_BB[ep_square]))
                                        add_move(from_left, ep_square, CAPTURE);
                        }
                        if ((bitboard_of(Us, PAWN) & SQUARE_BB[from_right])) {
                                if ((not_pinned & SQUARE_BB[from_right]) || (pin_line[from_right] & SQUARE_BB[ep_square]))
                                        add_move(from_right, ep_square, CAPTURE);
                        }
                }
        }

	//b1 now contains non-pinned pawns which ARE on the last rank (about to promote)
	b1 = bitboard_of(Us, PAWN) & not_pinned & MASK_RANK[relative_rank<Us>(RANK7)];
	if (b1) {
		//Quiet promotions
		b2 = shift<relative_dir<Us>(NORTH)>(b1) & quiet_mask;
			while (b2) {
		s = pop_lsb(&b2);
		
		// CRITICAL FIX: Skip if pop_lsb returned invalid square
		if (s == NO_SQUARE || s < 0 || s >= 64) {
			continue;  // Skip invalid square
		}
		
		Square from = s - relative_dir<Us>(NORTH);
		
		// CRITICAL FIX: Validate calculated source square for promotions
		if (from < 0 || from >= 64) {
			continue;  // Skip this invalid move calculation
		}
		
		add_promotion_moves(from, s, false);
	}

		//Promotion captures
		b2 = shift<relative_dir<Us>(NORTH_WEST)>(b1) & capture_mask;
		b3 = shift<relative_dir<Us>(NORTH_EAST)>(b1) & capture_mask;

		while (b2) {
			s = pop_lsb(&b2);
			
			// CRITICAL FIX: Skip if pop_lsb returned invalid square
			if (s == NO_SQUARE || s < 0 || s >= 64) {
				continue;  // Skip invalid square
			}
			
			Square from = s - relative_dir<Us>(NORTH_WEST);
			
			// CRITICAL FIX: Validate calculated source square
			if (from < 0 || from >= 64) {
				continue;  // Skip this invalid move calculation
			}
			
			add_promotion_moves(from, s, true);
		}

		while (b3) {
			s = pop_lsb(&b3);
			
			// CRITICAL FIX: Skip if pop_lsb returned invalid square
			if (s == NO_SQUARE || s < 0 || s >= 64) {
				continue;  // Skip invalid square
			}
			
			Square from = s - relative_dir<Us>(NORTH_EAST);
			
			// CRITICAL FIX: Validate calculated source square
			if (from < 0 || from >= 64) {
				continue;  // Skip this invalid move calculation
			}
			
			add_promotion_moves(from, s, true);
		}
	}

	return list;
}

//A convenience class for interfacing with legal moves, rather than using the low-level
//generate_legals() function directly. It can be iterated over.
template<Color Us>
class MoveList {
public:
	explicit MoveList(Position& p) : last(p.generate_legals<Us>(list)) {}

	const Move* begin() const { return list; }
	const Move* end() const { return last; }
	size_t size() const { return last - list; }
private:
        Move list[MAX_MOVES];
        Move *last;
};

// Disabling debug-only generate_moves function
#if 0
template<MoveFlags Flags, PieceType Piece>
inline Move* generate_moves(Position& pos, Move* list) {
    constexpr Color Us = Flags & WHITE_TURN ? WHITE : BLACK;
    constexpr Color Them = ~Us;
    constexpr bool Checks = Flags & CHECKS_ONLY;

    const Square our_king = pos.king_square(Us);
    const Bitboard our_pieces = pos.pieces(Us);
    const Bitboard their_pieces = pos.pieces(Them);
    const Bitboard occupied = our_pieces | their_pieces;
    
    Bitboard targets;
    
    if constexpr (Piece == PAWN) {
        constexpr Direction Up = relative_dir<Us>(NORTH);
        constexpr Direction Down = -Up;
        
        const Bitboard pawns = pos.pieces(Us, PAWN);
        const Bitboard empty = ~occupied;
        
        // Single pawn pushes
        Bitboard b1 = shift<Up>(pawns) & empty;
        std::cerr << "DEBUG: Pawn single pushes bitboard: " << std::hex << b1 << std::dec << std::endl;
        
        // Double pawn pushes  
        Bitboard b2 = shift<Up>(b1) & empty & relative_rank_bb<Us>(RANK_4);
        std::cerr << "DEBUG: Pawn double pushes bitboard: " << std::hex << b2 << std::dec << std::endl;
        
        if constexpr (!Checks) targets = their_pieces;
        else targets = pos.checkers() ? their_pieces & pos.checkers() : their_pieces;
        
        // Generate single pushes
        while (b1) {
            Square s = pop_lsb(&b1);
            Square from = s - Up;
            std::cerr << "DEBUG: Single push - from=" << from << " to=" << s << std::endl;
            if (from >= 64 || s >= 64) {
                std::cerr << "ERROR: Invalid single push move detected! from=" << from << " to=" << s << std::endl;
            }
            if (relative_rank<Us>(s) == RANK_8)
                list = make<PROMOTION>(from, square_bb(s), list);
            else
                list = make<QUIET>(from, square_bb(s), list);
        }
        
        // Generate double pushes
        while (b2) {
            Square s = pop_lsb(&b2);
            Square from = s - Up - Up;
            std::cerr << "DEBUG: Double push - from=" << from << " to=" << s << std::endl;
            if (from >= 64 || s >= 64) {
                std::cerr << "ERROR: Invalid double push move detected! from=" << from << " to=" << s << std::endl;
            }
            list = make<QUIET>(from, square_bb(s), list);
        }
        
        // Captures
        Bitboard b3 = shift<Up + WEST>(pawns) & targets;
        Bitboard b4 = shift<Up + EAST>(pawns) & targets;
        
        std::cerr << "DEBUG: Pawn west captures bitboard: " << std::hex << b3 << std::dec << std::endl;
        std::cerr << "DEBUG: Pawn east captures bitboard: " << std::hex << b4 << std::dec << std::endl;
        
        while (b3) {
            Square s = pop_lsb(&b3);
            Square from = s - Up - WEST;
            std::cerr << "DEBUG: West capture - from=" << from << " to=" << s << std::endl;
            if (from >= 64 || s >= 64) {
                std::cerr << "ERROR: Invalid west capture move detected! from=" << from << " to=" << s << std::endl;
            }
            if (relative_rank<Us>(s) == RANK_8)
                list = make<PROMOTION | CAPTURE>(from, square_bb(s), list);
            else
                list = make<CAPTURE>(from, square_bb(s), list);
        }
        
        while (b4) {
            Square s = pop_lsb(&b4);
            Square from = s - Up - EAST;
            std::cerr << "DEBUG: East capture - from=" << from << " to=" << s << std::endl;
            if (from >= 64 || s >= 64) {
                std::cerr << "ERROR: Invalid east capture move detected! from=" << from << " to=" << s << std::endl;
            }
            if (relative_rank<Us>(s) == RANK_8)
                list = make<PROMOTION | CAPTURE>(from, square_bb(s), list);
            else
                list = make<CAPTURE>(from, square_bb(s), list);
        }
        
        // En passant
        if (pos.en_passant() != NO_SQUARE) {
            Square ep_square = pos.en_passant();
            std::cerr << "DEBUG: En passant square: " << ep_square << std::endl;
            
            Bitboard attackers = pawns & pawn_attacks<Them>(ep_square);
            while (attackers) {
                Square from = pop_lsb(&attackers);
                std::cerr << "DEBUG: En passant - from=" << from << " to=" << ep_square << std::endl;
                if (from >= 64 || ep_square >= 64) {
                    std::cerr << "ERROR: Invalid en passant move detected! from=" << from << " to=" << ep_square << std::endl;
                }
                list = make<EN_PASSANT>(from, square_bb(ep_square), list);
            }
        }
    }
    
    else if constexpr (Piece == KNIGHT) {
        Bitboard knights = pos.pieces(Us, KNIGHT);
        std::cerr << "DEBUG: Knights bitboard: " << std::hex << knights << std::dec << std::endl;
        
        while (knights) {
            Square from = pop_lsb(&knights);
            std::cerr << "DEBUG: Knight from square: " << from << std::endl;
            if (from >= 64) {
                std::cerr << "ERROR: Invalid knight source square: " << from << std::endl;
                continue;
            }
            
            Bitboard attacks = knight_attacks(from);
            std::cerr << "DEBUG: Knight attacks from " << from << ": " << std::hex << attacks << std::dec << std::endl;
            
            if constexpr (!Checks) attacks &= ~our_pieces;
            else attacks &= pos.checkers() ? pos.checkers() : ~our_pieces;
            
            while (attacks) {
                Square to = pop_lsb(&attacks);
                std::cerr << "DEBUG: Knight move - from=" << from << " to=" << to << std::endl;
                if (to >= 64) {
                    std::cerr << "ERROR: Invalid knight destination square: " << to << std::endl;
                    continue;
                }
                
                if (pos.piece_on(to) != NO_PIECE)
                    list = make<CAPTURE>(from, square_bb(to), list);
                else
                    list = make<QUIET>(from, square_bb(to), list);
            }
        }
    }
    
    else if constexpr (Piece == BISHOP || Piece == ROOK || Piece == QUEEN) {
        Bitboard pieces = pos.pieces(Us, Piece);
        std::cerr << "DEBUG: " << (Piece == BISHOP ? "Bishop" : Piece == ROOK ? "Rook" : "Queen") 
                  << " pieces bitboard: " << std::hex << pieces << std::dec << std::endl;
        
        while (pieces) {
            Square from = pop_lsb(&pieces);
            std::cerr << "DEBUG: Sliding piece from square: " << from << std::endl;
            if (from >= 64) {
                std::cerr << "ERROR: Invalid sliding piece source square: " << from << std::endl;
                continue;
            }
            
            Bitboard attacks;
            if constexpr (Piece == BISHOP) attacks = bishop_attacks(from, occupied);
            else if constexpr (Piece == ROOK) attacks = rook_attacks(from, occupied);
            else attacks = queen_attacks(from, occupied);
            
            std::cerr << "DEBUG: Sliding piece attacks from " << from << ": " << std::hex << attacks << std::dec << std::endl;
            
            if constexpr (!Checks) attacks &= ~our_pieces;
            else attacks &= pos.checkers() ? pos.checkers() : ~our_pieces;
            
            while (attacks) {
                Square to = pop_lsb(&attacks);
                std::cerr << "DEBUG: Sliding piece move - from=" << from << " to=" << to << std::endl;
                if (to >= 64) {
                    std::cerr << "ERROR: Invalid sliding piece destination square: " << to << std::endl;
                    continue;
                }
                
                if (pos.piece_on(to) != NO_PIECE)
                    list = make<CAPTURE>(from, square_bb(to), list);
                else
                    list = make<QUIET>(from, square_bb(to), list);
            }
        }
    }
    
    else if constexpr (Piece == KING) {
        Square from = pos.king_square(Us);
        std::cerr << "DEBUG: King square: " << from << std::endl;
        if (from >= 64) {
            std::cerr << "ERROR: Invalid king square: " << from << std::endl;
            return list;
        }
        
        Bitboard attacks = king_attacks(from);
        std::cerr << "DEBUG: King attacks from " << from << ": " << std::hex << attacks << std::dec << std::endl;
        
        if constexpr (!Checks) attacks &= ~our_pieces;
        else attacks &= pos.checkers() ? pos.checkers() : ~our_pieces;
        
        while (attacks) {
            Square to = pop_lsb(&attacks);
            std::cerr << "DEBUG: King move - from=" << from << " to=" << to << std::endl;
            if (to >= 64) {
                std::cerr << "ERROR: Invalid king destination square: " << to << std::endl;
                continue;
            }
            
            if (pos.piece_on(to) != NO_PIECE)
                list = make<CAPTURE>(from, square_bb(to), list);
            else
                list = make<QUIET>(from, square_bb(to), list);
        }
    }
    
    return list;
}
#endif
