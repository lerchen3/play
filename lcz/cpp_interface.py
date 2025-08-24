import os
import ctypes
import numpy as np
import torch
import subprocess
from move_encoding import MOVE_INDEX, INDEX_MOVE
import config
import time
import re

# Standard starting FEN for normal chess
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def _normalize_fen(fen: str) -> str:
    """Return canonical FEN string with exactly six fields."""

    fen = fen.replace("\x00", "").strip()
    if fen.startswith("startpos"):
        return START_FEN

    parts = fen.split()
    parts = parts[:6]

    board = parts[0] if len(parts) > 0 else START_FEN.split()[0]
    active = parts[1] if len(parts) > 1 else "w"
    castling = parts[2] if len(parts) > 2 else "-"
    ep = parts[3] if len(parts) > 3 else "-"
    half = parts[4] if len(parts) > 4 else "0"
    full = parts[5] if len(parts) > 5 else "1"

    return f"{board} {active} {castling} {ep} {half} {full}"

# Location of the shared library. Can be overridden with LCZ_LIB_DIR to support
# read-only source directories (e.g. Kaggle datasets).
LIB_DIR = os.environ.get('LCZ_LIB_DIR', os.path.dirname(__file__))
LIB_PATH = os.path.join(LIB_DIR, 'liblcz.so')

# Always rebuild the C++ library on import to pick up any patches and avoid stale .so
# Use file-based locking to prevent race conditions when multiple processes build simultaneously
import fcntl
import time

def build_cpp_library():
    lock_file = os.path.join(LIB_DIR, '.build_lock')
    build_script = os.path.join(os.path.dirname(__file__), 'build_cpp.sh')
    
    # If library already exists and is recent, skip build
    if os.path.exists(LIB_PATH):
        try:
            lib_mtime = os.path.getmtime(LIB_PATH)
            script_mtime = os.path.getmtime(build_script)
            if lib_mtime > script_mtime:
                return
        except OSError:
            pass  # If we can't check mtimes, proceed with build
    
    # Use file locking to ensure only one process builds at a time
    os.makedirs(LIB_DIR, exist_ok=True)
    with open(lock_file, 'w') as lock:
        try:
            # Try to acquire exclusive lock with timeout
            for attempt in range(30):  # 30 second timeout
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError:
                    if attempt == 29:
                        raise RuntimeError("Timeout waiting for build lock")
                    time.sleep(1)
            
            # Check again if library was built while we were waiting
            if os.path.exists(LIB_PATH):
                try:
                    lib_mtime = os.path.getmtime(LIB_PATH)
                    script_mtime = os.path.getmtime(build_script)
                    if lib_mtime > script_mtime:
                        return
                except OSError:
                    pass
            
            # Normalize build script line endings to Unix format
            try:
                with open(build_script, 'rb') as bf:
                    content = bf.read()
                unix_content = content.replace(b'\r\n', b'\n')
                if unix_content != content:
                    with open(build_script, 'wb') as bf:
                        bf.write(unix_content)
            except Exception as e2:
                print(f"Warning: Failed to normalize line endings: {e2}")
            
            env = dict(os.environ, OUT_DIR=LIB_DIR)
            env.pop('BASH_ENV', None)
            subprocess.check_call(['bash', '-x', build_script], env=env)
            
        finally:
            # Release lock
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

try:
    build_cpp_library()
except subprocess.CalledProcessError as e:
    out = getattr(e, 'output', None)
    out_str = out.decode('utf-8', errors='ignore') if isinstance(out, (bytes, bytearray)) else ""
    raise RuntimeError(f"Failed to build C++ library in {LIB_DIR}: return code {e.returncode}\nOutput:\n{out_str}")
except Exception as e:
    raise RuntimeError(f"Failed to build C++ library in {LIB_DIR}: {e}")

# Build dlopen flags: RTLD_NOW | RTLD_GLOBAL always, plus RTLD_NODELETE to avoid unloading
flags = 0
if hasattr(os, 'RTLD_NOW'):
    flags |= os.RTLD_NOW
if hasattr(os, 'RTLD_GLOBAL'):
    flags |= os.RTLD_GLOBAL
# Add Nodelete (glibc value 0x1000) to prevent destructor unload crashes
if hasattr(os, 'RTLD_NODELETE'):
    flags |= os.RTLD_NODELETE
else:
    flags |= 0x1000  # RTLD_NODELETE fallback
_lib = ctypes.CDLL(LIB_PATH, mode=flags)

# callback type for net predictions
BATCH_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_float),
                                  ctypes.POINTER(ctypes.c_float))

_lib.game_create.restype = ctypes.c_void_p
_lib.game_destroy.argtypes = [ctypes.c_void_p]
_lib.game_reset.argtypes = [ctypes.c_void_p]
_lib.game_set_fen.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.game_get_fen.argtypes = [ctypes.c_void_p]
_lib.game_get_fen.restype = ctypes.c_char_p
_lib.game_get_past_fen.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.game_get_past_fen.restype = ctypes.c_char_p
_lib.game_turn.argtypes = [ctypes.c_void_p]
_lib.game_turn.restype = ctypes.c_int
_lib.game_move.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.game_move.restype = ctypes.c_int
_lib.game_legal_moves.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
_lib.game_legal_moves.restype = ctypes.c_int
_lib.game_is_game_over.argtypes = [ctypes.c_void_p]
_lib.game_is_game_over.restype = ctypes.c_int
_lib.game_result.argtypes = [ctypes.c_void_p]
_lib.game_result.restype = ctypes.c_int

_lib.fen_move.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
_lib.fen_move.restype = ctypes.c_int
_lib.fen_legal_moves.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
_lib.fen_legal_moves.restype = ctypes.c_int
_lib.fen_is_game_over.argtypes = [ctypes.c_char_p]
_lib.fen_is_game_over.restype = ctypes.c_int
_lib.fen_result.argtypes = [ctypes.c_char_p]
_lib.fen_result.restype = ctypes.c_int
_lib.fen_turn.argtypes = [ctypes.c_char_p]
_lib.fen_turn.restype = ctypes.c_int

_lib.mcts_create.restype = ctypes.c_void_p
_lib.mcts_create.argtypes = [BATCH_CALLBACK]
_lib.mcts_destroy.argtypes = [ctypes.c_void_p]
_lib.mcts_search.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                             ctypes.POINTER(ctypes.c_float)]
_lib.mcts_set_params.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                 ctypes.c_double, ctypes.c_double, ctypes.c_double]

def set_mcts_params(num_sims, cpuct, batch_size, root_noise_eps,
                    dirichlet_alpha, virtual_loss):
    """Configure MCTS hyperparameters."""
    _lib.mcts_set_params(int(num_sims), float(cpuct), int(batch_size),
                         float(root_noise_eps), float(dirichlet_alpha),
                         float(virtual_loss))

# ---------------------------------------------------------------------------
# Helper utilities for working with FEN strings without python-chess
# ---------------------------------------------------------------------------

PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}

PIECE_TO_TYPE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
}


def _parse_board_coords(board_part: str):
    """Yield `(piece, rank, file)` using 1-based board coordinates."""
    rank = 8
    file = 1
    for ch in board_part:
        if ch == '/':
            rank -= 1
            file = 1
        elif ch.isdigit():
            file += int(ch)
        else:
            yield ch, rank, file
            file += 1


def _parse_fen_board(fen: str):
    """Return a list of 64 pieces parsed from a FEN string."""
    fen = _normalize_fen(fen)
    board_part = fen.split()[0]
    board = [None] * 64
    # they're 1-indexed; subtract 1.
    for ch, rank, file in _parse_board_coords(board_part):
        r = 8 - rank  # FEN rank 8 -> rr=0, FEN rank 1 -> rr=7
        f = file - 1
        if 0 <= r < 8 and 0 <= f < 8:
            board[r * 8 + f] = ch
    return board

def fen_history_to_state(fens):
    """Convert a list of up to 8 FEN strings (current + past) to state tensor."""
    # Blank entries represent positions that have not occurred yet.  We do not
    # normalise these to the starting position, but rather treat them as empty
    # so that the corresponding planes remain zero.
    fens = [f.replace("\x00", "").strip() for f in fens]
    if not fens:
        fens = ['']

    cur_fen = _normalize_fen(fens[0])
    current_parts = cur_fen.split()
    cur_active = current_parts[1] if len(current_parts) > 1 else 'w'
    cur_castling = current_parts[2] if len(current_parts) > 2 else '-'
    def _safe_int(tok: str, default: int) -> int:
        """Parse leading integer from token, returning default on failure."""
        if not tok:
            return default
        m = re.match(r"\d+", tok)
        return int(m.group(0)) if m else default

    cur_halfmove = _safe_int(current_parts[4], 0) if len(current_parts) > 4 else 0
    cur_fullmove = _safe_int(current_parts[5], 1) if len(current_parts) > 5 else 1

    # Always represent from White's perspective (no flipping for Black)
    state = np.zeros((config.IN_CHANNELS, 8, 8), dtype=np.float32)

    for idx, fen in enumerate(fens):
        fen = fen.strip()
        if not fen:
            continue
        fen = _normalize_fen(fen)
        parts = fen.split()
        if not parts:
            continue
        board_part = parts[0]
        active = parts[1] if len(parts) > 1 else 'w'
        for ch, rank, file in _parse_board_coords(board_part):
            # Match C++ coordinate system: (7 - ri) * 8 + file
            rr = 8 - rank  # FEN rank 8 -> rr=0, FEN rank 1 -> rr=7
            ff = file - 1
            piece_color_white = ch.isupper()
            is_player1 = (piece_color_white and cur_active == 'w') or (
                not piece_color_white and cur_active == 'b'
            )
            plane_offset = idx * 12 + (0 if is_player1 else 6)
            piece_type = PIECE_TO_TYPE[ch]
            if 0 <= rr < 8 and 0 <= ff < 8:
                state[plane_offset + piece_type, rr, ff] = 1

    const_base = 96
    state[const_base, :, :] = 1 if cur_active == 'w' else 0
    state[const_base+1, :, :] = cur_fullmove
    if cur_active == 'w':
        state[const_base+2, :, :] = 1 if 'K' in cur_castling else 0
        state[const_base+3, :, :] = 1 if 'Q' in cur_castling else 0
        state[const_base+4, :, :] = 1 if 'k' in cur_castling else 0
        state[const_base+5, :, :] = 1 if 'q' in cur_castling else 0
    else:
        state[const_base+2, :, :] = 1 if 'k' in cur_castling else 0
        state[const_base+3, :, :] = 1 if 'q' in cur_castling else 0
        state[const_base+4, :, :] = 1 if 'K' in cur_castling else 0
        state[const_base+5, :, :] = 1 if 'Q' in cur_castling else 0
    state[const_base+6, :, :] = cur_halfmove

    return torch.from_numpy(state)

class ChessGame:
    def __init__(self, fen=None):
        self.obj = _lib.game_create()
        if fen:
            self.set_fen(fen)
        else:
            self.reset()

    def reset(self):
        _lib.game_reset(self.obj)
        _lib.game_set_fen(self.obj, b"startpos")

    def set_fen(self, fen):
        _lib.game_set_fen(self.obj, fen.encode('utf-8'))

    def fen(self):
        return _lib.game_get_fen(self.obj).decode('utf-8')

    def turn(self):
        return _lib.game_turn(self.obj)

    def current_player(self):
        return 1 if self.turn() == 1 else -1

    def move(self, uci):
        return bool(_lib.game_move(self.obj, uci.encode('utf-8')))

    def legal_moves(self):
        buf = ctypes.create_string_buffer(4096)
        n = _lib.game_legal_moves(self.obj, buf, len(buf))
        moves = buf.value.decode('utf-8').split() if n > 0 else []
        return moves

    def is_game_over(self):
        return bool(_lib.game_is_game_over(self.obj))

    def result(self):
        return _lib.game_result(self.obj)

    def board(self):
        """Return the board as a list of piece characters (None if empty)."""
        return _parse_fen_board(self.fen())

    def board_ascii(self):
        """Return a printable board representation."""
        pieces = self.board()
        lines = []
        for r in range(0, 8):  # Fixed: now goes 0->7 instead of 7->0
            line = []
            for f in range(8):
                p = pieces[r * 8 + f]
                line.append(p if p else '.')
            lines.append(' '.join(line))
        return '\n'.join(lines)

    def get_state(self):
        fens = []
        for i in range(8):
            fen_bytes = _lib.game_get_past_fen(self.obj, i)
            fen = fen_bytes.decode('utf-8') if fen_bytes else ''
            fens.append(fen)
        return fen_history_to_state(fens)

class FenGame:
    """Lightweight game object that stores only a FEN string and uses C++ helpers"""

    def __init__(self, fen="startpos"):
        self.fen_str = fen
        # Store only past positions; start with an empty history
        self.history = []

    # Basic accessors
    def fen(self):
        return self.fen_str

    def turn(self):
        return _lib.fen_turn(self.fen_str.encode('utf-8'))

    def current_player(self):
        return 1 if self.turn() == 1 else -1

    def move(self, uci):
        buf = ctypes.create_string_buffer(256)
        res = _lib.fen_move(self.fen_str.encode('utf-8'),
                            uci.encode('utf-8'), buf, len(buf))
        if res == 1:
            self.history.append(self.fen_str)
            # Keep at most the last 7 previous states
            if len(self.history) > 7:
                self.history.pop(0)
            self.fen_str = buf.value.decode('utf-8')
            return True
        return False

    def legal_moves(self):
        buf = ctypes.create_string_buffer(4096)
        n = _lib.fen_legal_moves(self.fen_str.encode('utf-8'), buf, len(buf))
        return buf.value.decode('utf-8').split() if n > 0 else []

    def is_game_over(self):
        return bool(_lib.fen_is_game_over(self.fen_str.encode('utf-8')))

    def result(self):
        return _lib.fen_result(self.fen_str.encode('utf-8'))

    def board(self):
        return _parse_fen_board(self.fen_str)

    def board_ascii(self):
        pieces = self.board()
        lines = []
        for r in range(0, 8):  # Fixed: now goes 0->7 instead of 7->0
            line = []
            for f in range(8):
                p = pieces[r * 8 + f]
                line.append(p if p else '.')
            lines.append(' '.join(line))
        return '\n'.join(lines)

    def get_state(self):
        # Include the current position plus up to the last 7 positions
        fens = [self.fen_str]
        fens.extend(reversed(self.history[-7:]))
        while len(fens) < 8:
            fens.append('')
        return fen_history_to_state(fens)

class MCTS:
    def __init__(self, net, device='cpu'):
        self.net = net
        self.device = device

        def batch_predict_cb(fen_batch_c, batch_size, policy_out, value_out):
            py_start_time = time.time()

            t0 = time.time()
            fen_batch_str = fen_batch_c.decode('utf-8')
            groups = fen_batch_str.strip().split('\n')
            fens_list = []
            for g in groups:
                fens = g.split('|')
                while len(fens) < 8:
                    fens.append('')
                fens_list.append(fens[:8])
            t1 = time.time()

            state_list = [fen_history_to_state(fens) for fens in fens_list]
            t2 = time.time()

            state_batch = torch.stack(state_list).to(self.device)
            t3 = time.time()

            with torch.no_grad():
                logits_batch, v_batch = self.net(state_batch)
            t4 = time.time()

            # The network already outputs log probabilities (log softmax).  The
            # C++ side expects these logs and will exponentiate them when
            # computing priors, so we pass them through directly without taking
            # an exponent here.
            policy_batch_np = logits_batch.cpu().numpy()
            value_batch_np = v_batch.cpu().numpy().flatten()
            t5 = time.time()

            policy_flat = policy_batch_np.flatten().astype(np.float32)
            ctypes.memmove(policy_out, policy_flat.ctypes.data, policy_flat.nbytes)
            ctypes.memmove(value_out, value_batch_np.ctypes.data, value_batch_np.nbytes)
            t6 = time.time()
            
            py_end_time = time.time()

        self._cb = BATCH_CALLBACK(batch_predict_cb)
        self.obj = _lib.mcts_create(self._cb)

    def search(self, game):
        arr_type = ctypes.c_float * len(MOVE_INDEX)
        out = arr_type()
        _lib.mcts_search(self.obj, game.fen().encode('utf-8'), out)
        policy = np.frombuffer(out, dtype=np.float32, count=len(MOVE_INDEX))
        s = policy.sum()
        if s > 0:
            policy /= s
        return policy

    def __del__(self):
        if hasattr(self, 'obj') and self.obj:
            _lib.mcts_destroy(self.obj)
