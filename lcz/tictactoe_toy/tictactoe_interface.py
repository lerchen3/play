import os
import ctypes
import numpy as np
import torch

_default_dir = os.path.dirname(__file__)
_lib_dir = os.environ.get('TICTACTOE_LIB_DIR', _default_dir)
os.makedirs(_lib_dir, exist_ok=True)
LIB_PATH = os.path.join(_lib_dir, 'libtictactoe.so')
if not os.path.exists(LIB_PATH):
    build_script = os.path.join(_default_dir, 'build_tictactoe.sh')
    import subprocess
    env = os.environ.copy()
    env['OUT_DIR'] = _lib_dir
    subprocess.check_call(['bash', build_script], env=env)
_lib = ctypes.CDLL(LIB_PATH)

PREDICT_CB = ctypes.CFUNCTYPE(None, ctypes.c_char_p,
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float))

_lib.ttt_game_create.restype = ctypes.c_void_p
_lib.ttt_game_destroy.argtypes = [ctypes.c_void_p]
_lib.ttt_game_reset.argtypes = [ctypes.c_void_p]
_lib.ttt_game_get_state.restype = ctypes.c_char_p
_lib.ttt_game_get_state.argtypes = [ctypes.c_void_p]
_lib.ttt_game_turn.argtypes = [ctypes.c_void_p]
_lib.ttt_game_turn.restype = ctypes.c_int
_lib.ttt_game_move.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.ttt_game_move.restype = ctypes.c_int
_lib.ttt_game_is_game_over.argtypes = [ctypes.c_void_p]
_lib.ttt_game_is_game_over.restype = ctypes.c_int
_lib.ttt_game_result.argtypes = [ctypes.c_void_p]
_lib.ttt_game_result.restype = ctypes.c_int

_lib.ttt_mcts_create.restype = ctypes.c_void_p
_lib.ttt_mcts_create.argtypes = [PREDICT_CB]
_lib.ttt_mcts_destroy.argtypes = [ctypes.c_void_p]
_lib.ttt_mcts_search.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                 ctypes.POINTER(ctypes.c_float)]

class TicTacToeGame:
    def __init__(self):
        self.obj = _lib.ttt_game_create()

    def __del__(self):
        if hasattr(self, 'obj') and self.obj:
            _lib.ttt_game_destroy(self.obj)
            self.obj = None

    def reset(self):
        _lib.ttt_game_reset(self.obj)

    def state(self):
        return _lib.ttt_game_get_state(self.obj).decode('utf-8')

    def turn(self):
        return _lib.ttt_game_turn(self.obj)

    def move(self, idx):
        return bool(_lib.ttt_game_move(self.obj, idx))

    def is_game_over(self):
        return bool(_lib.ttt_game_is_game_over(self.obj))

    def result(self):
        return _lib.ttt_game_result(self.obj)

    def tensor_state(self):
        s = self.state()
        board = s[:9]
        player = s[9]
        x = np.array([1 if c == 'X' else 0 for c in board], dtype=np.float32)
        o = np.array([1 if c == 'O' else 0 for c in board], dtype=np.float32)
        turn_plane = np.full(9, 1.0 if player == 'X' else 0.0, dtype=np.float32)
        arr = np.stack([x, o, turn_plane], axis=0)
        return torch.from_numpy(arr.reshape(3, 3, 3))

class MCTS:
    def __init__(self, net, device='cpu'):
        self.net = net
        self.device = device
        def cb(state_c, policy_out, value_out):
            state = state_c.decode('utf-8')
            board = state[:9]
            player = state[9]
            x = np.array([1 if c == 'X' else 0 for c in board], dtype=np.float32)
            o = np.array([1 if c == 'O' else 0 for c in board], dtype=np.float32)
            turn_plane = np.full(9, 1.0 if player == 'X' else 0.0, dtype=np.float32)
            arr = np.stack([x, o, turn_plane], axis=0)
            tensor = torch.from_numpy(arr.reshape(3, 3, 3)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, v = self.net(tensor)
            logp = torch.log_softmax(logits, dim=1)
            lp = logp.cpu().numpy()[0]
            for i in range(9):
                policy_out[i] = float(lp[i])
            value_out[0] = float(v.item())
        self._cb = PREDICT_CB(cb)
        self.obj = _lib.ttt_mcts_create(self._cb)

    def search(self, game):
        arr_t = ctypes.c_float * 9
        out = arr_t()
        _lib.ttt_mcts_search(self.obj, game.state().encode('utf-8'), out)
        pol = np.frombuffer(out, dtype=np.float32, count=9)
        if pol.sum() > 0:
            pol = pol / pol.sum()
        return pol

    def __del__(self):
        if hasattr(self, 'obj') and self.obj:
            _lib.ttt_mcts_destroy(self.obj)
