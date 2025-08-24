from module import Module
from layers import Linear, ReLU, Sequential, Conv2d, Flatten, LSTM

import numpy as np
from tensor import Tensor
from operations import matmul, add, elementwise_add, sigmoid, elementwise_mul, index

class MLP(Module):
    def __init__(self, in_features: int, hidden_features: list[int], out_features: int):
        super().__init__()
        
        layers = []
        prev_features = in_features
        
        # Create hidden layers
        for hidden_size in hidden_features:
            layers.append(Linear(prev_features, hidden_size))
            layers.append(ReLU())
            prev_features = hidden_size
            
        # Add output layer
        layers.append(Linear(prev_features, out_features))
        
        # Use register_module explicitly
        self.register_module('model', Sequential(*layers))

    def forward(self, x):
        return self.model(x)

class MixtureOfExperts(Module):
    """Mixture-of-Experts with 8 specialized + 2 shared experts and bias-only load balancing."""
    def __init__(self, input_dim: int, output_dim: int,
                 num_specialized: int = 8, num_shared: int = 2,
                 k: int = 2, gamma: float = 0.1):
        super().__init__()
        import numpy as _np
        from layers import Linear
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_specialized = num_specialized
        self.num_shared = num_shared
        self.k = k
        self.gamma = gamma
        self.batch_count = 0
        # running count of times routed per specialized expert
        self.c = _np.zeros(num_specialized, dtype=_np.int64)
        # bias for selection, initialized to zero
        self.b = _np.zeros(num_specialized, dtype=_np.float32)
        # centroids for gating, initialized to zero vectors
        self.centroids = _np.zeros((num_specialized, input_dim), dtype=_np.float32)
        # create specialized experts
        for i in range(num_specialized):
            exp = Linear(input_dim, output_dim)
            self.register_module(f'spec_exp_{i}', exp)
        # create shared experts
        for i in range(num_shared):
            exp = Linear(input_dim, output_dim)
            self.register_module(f'shar_exp_{i}', exp)

    def forward(self, x: Tensor) -> Tensor:
        import numpy as _np
        # x: shape (input_dim, batch_size)
        batch_size = x.data.shape[1]
        self.batch_count += 1
        # gating for specialized experts
        cent_t = Tensor(self.centroids, requires_grad=False)
        s_raw = matmul(cent_t, x)  # (num_specialized, batch)
        s = sigmoid(s_raw)
        # add bias for top-k selection
        b_t = Tensor(self.b.reshape(self.num_specialized, 1), requires_grad=False)
        s_sel = s + b_t
        # use numpy to select top-k per sample and update counts (bias-only load balancing)
        s_sel_np = s_sel.data
        mask = _np.zeros_like(s_sel_np)
        for j in range(batch_size):
            topk_idx = _np.argpartition(-s_sel_np[:, j], self.k)[:self.k]
            mask[topk_idx, j] = 1.0
            for idx in topk_idx:
                self.c[idx] += 1
        # create Tensor mask and compute gating weights to allow gradient flow through s
        mask_t = Tensor(mask, requires_grad=False)
        g_spec_t = elementwise_mul(s, mask_t)
        # normalize gating weights so they sum to 1 across specialized experts per sample
        g_np = g_spec_t.data
        col_sums = g_np.sum(axis=0, keepdims=True) + 1e-9
        g_spec_t.data = g_np / col_sums
        # aggregate expert outputs
        y = None
        # specialized experts
        for i in range(self.num_specialized):
            exp = getattr(self, f'spec_exp_{i}')
            fx = exp(x)  # (output_dim, batch)
            # gating weight for expert i: shape (batch,)
            gi = index(g_spec_t, (i, slice(None)))  # Tensor, requires_grad from s
            yi = elementwise_mul(fx, gi)
            y = yi if y is None else y + yi
        # shared experts (always included)
        for i in range(self.num_shared):
            exp = getattr(self, f'shar_exp_{i}')
            fx = exp(x)
            y = fx if y is None else y + fx
        # update centroids and biases
        g_spec_np = g_spec_t.data
        for i in range(self.num_specialized):
            mask_i = g_spec_np[i]
            total = mask_i.sum()
            if total > 0:
                # weighted average of inputs
                wavg = (x.data * mask_i).sum(axis=1) / total
                rate = 1.0 / self.batch_count
                self.centroids[i] = (1 - rate) * self.centroids[i] + rate * wavg
        m = self.c.mean()
        d = self.c - m
        for i in range(self.num_specialized):
            self.b[i] -= _np.sign(d[i]) * self.gamma
        return y

class SimpleCNN(Module):
    """A small CNN for 28x28 grayscale images."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = ReLU()
        self.flatten = Flatten()
        self.fc = Linear(4 * 28 * 28, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
class SimpleLSTMClassifier(Module):
    """Sequence classifier using a single-layer LSTM over per-step inputs.
    Expects input shape (input_dim, T, B) and outputs logits (num_classes, B).
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.lstm = LSTM(input_dim, hidden_dim)
        self.fc = Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        h = self.lstm(x)
        return self.fc(h)
