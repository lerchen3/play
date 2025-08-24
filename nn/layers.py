import numpy as np
from collections import OrderedDict
from module import Module
from tensor import Tensor
from operations import linear, relu, conv2d, reshape, sigmoid, tanh

class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        # Initialize weights and bias
        weight_data = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        bias_data = np.zeros((out_features,))
        
        # Create tensors
        weight = Tensor(weight_data, requires_grad=True)
        bias = Tensor(bias_data, requires_grad=True)
        
        # Register parameters explicitly
        self.register_parameter('weight', weight)
        self.register_parameter('bias', bias)

    def forward(self, x: Tensor) -> Tensor:
        return linear(self.weight, x, self.bias)

class ReLU(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x : Tensor) -> Tensor:
        return relu(x)

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.register_module(str(idx), module)

    def forward(self, x : Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x 

class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        weight_data = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        bias_data = np.zeros((out_channels,))
        self.register_parameter('weight', Tensor(weight_data, requires_grad=True))
        self.register_parameter('bias', Tensor(bias_data, requires_grad=True))
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)

class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # x expected shape: (C, H, W, B) or (features, B)
        if x.data.ndim == 4:
            C, H, W, B = x.data.shape
            return reshape(x, (C * H * W, B))
        return x

class LSTMCell(Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # weights
        W_ih = np.random.randn(4 * hidden_dim, input_dim) * np.sqrt(1.0 / input_dim)
        W_hh = np.random.randn(4 * hidden_dim, hidden_dim) * np.sqrt(1.0 / hidden_dim)
        b_ih = np.zeros((4 * hidden_dim,))
        b_hh = np.zeros((4 * hidden_dim,))
        self.register_parameter('W_ih', Tensor(W_ih, requires_grad=True))
        self.register_parameter('W_hh', Tensor(W_hh, requires_grad=True))
        self.register_parameter('b_ih', Tensor(b_ih, requires_grad=True))
        self.register_parameter('b_hh', Tensor(b_hh, requires_grad=True))

    def forward(self, x_t: Tensor, h_prev: Tensor, c_prev: Tensor):
        # x_t: (input_dim, B), h_prev,c_prev: (hidden_dim, B)
        from operations import add, elementwise_add, elementwise_mul, index, matmul
        preact = add(matmul(self.W_ih, x_t), self.b_ih)
        preact = elementwise_add(preact, add(matmul(self.W_hh, h_prev), self.b_hh))

        H = self.hidden_dim
        i_gate = sigmoid(index(preact, (slice(0, H), slice(None))))
        f_gate = sigmoid(index(preact, (slice(H, 2 * H), slice(None))))
        g_gate = tanh(index(preact, (slice(2 * H, 3 * H), slice(None))))
        o_gate = sigmoid(index(preact, (slice(3 * H, 4 * H), slice(None))))

        c_t = elementwise_add(elementwise_mul(f_gate, c_prev), elementwise_mul(i_gate, g_gate))
        h_t = elementwise_mul(o_gate, tanh(c_t))
        return h_t, c_t

class LSTM(Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.cell = LSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: Tensor):
        # x shape can be (input_dim, T, B) or (T, input_dim, B)
        data = x.data
        if data.ndim != 3:
            raise ValueError("LSTM expects (input_dim, T, B) or (T, input_dim, B)")
        if data.shape[0] != self.cell.input_dim:
            # assume (T, input_dim, B) -> transpose to (input_dim, T, B)
            x_seq = np.transpose(data, (1, 0, 2))
        else:
            x_seq = data
        input_dim, T, B = x_seq.shape
        h = Tensor(np.zeros((self.hidden_dim, B)), requires_grad=False)
        c = Tensor(np.zeros((self.hidden_dim, B)), requires_grad=False)
        # iterate through time
        for t in range(T):
            x_t = Tensor(x_seq[:, t, :], requires_grad=False)
            h, c = self.cell(x_t, h, c)
        return h