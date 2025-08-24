import numpy as np
from network import MLP, MixtureOfExperts, SimpleCNN, SimpleLSTMClassifier
from operations import mse_loss, cross_entropy, softmax, relu
from tensor import Tensor
from optimizers import SOAP, Adam
import matplotlib
matplotlib.use('Agg')
from layers import Sequential, ReLU

class TensorElement:
    """Adapter for a single scalar element in a Tensor so that updating it updates the parent Tensor."""
    def __init__(self, parent, i, j):
        self.parent = parent
        self.i = i
        self.j = j

    @property
    def data(self):
        return self.parent.data[self.i, self.j]

    @data.setter
    def data(self, value):
        self.parent.data[self.i, self.j] = value

    @property
    def grad(self):
        if self.parent.grad is None:
            return None
        return self.parent.grad[self.i, self.j]

    @grad.setter
    def grad(self, value):
        if self.parent.grad is None:
            self.parent.grad = np.zeros_like(self.parent.data)
        self.parent.grad[self.i, self.j] = value

def get_linear_weights(module):
    """Recursively collects weight Tensors from modules that have a 'weight' attribute."""
    weights = []
    if hasattr(module, 'weight'):
        weights.append(module.weight)
    if hasattr(module, '_modules'):
        for m in module._modules.values():
            weights.extend(get_linear_weights(m))
    return weights

def tensor_to_matrix(tensor):
    """Converts a Tensor representing a matrix into a 2D list of TensorElement adapters."""
    m, n = tensor.data.shape
    matrix = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(TensorElement(tensor, i, j))
        matrix.append(row)
    return matrix

def zero_model_grad(module):
    """Zeros gradients for parameters of the module recursively."""
    for param in getattr(module, '_parameters', {}).values():
        if param is not None:
            param.grad = None
    for sub in getattr(module, '_modules', {}).values():
        zero_model_grad(sub)

def update_biases(module, lr):
    """Updates biases using a simple SGD step if they exist."""
    if hasattr(module, 'bias') and module.bias.grad is not None:
        module.bias.data -= lr * module.bias.grad
    if hasattr(module, '_modules'):
        for m in module._modules.values():
            update_biases(m, lr)

from sklearn.datasets import fetch_openml
# Download MNIST from openml (70000 samples)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)
# Normalize and split
X = X / 255.0
# First 60k for training, last 10k for test
x_train = X[:60000].T
y_train = y[:60000]
x_test  = X[60000:].T
y_test  = y[60000:]
# One-hot encode labels
num_classes = 10
y_train_oh = np.eye(num_classes)[y_train].T
y_test_oh  = np.eye(num_classes)[y_test].T

n_epochs = 2
batch_size = 64
lr = 0.01

def run_experiment(model, name):
    print(f"\n=== {name} ===")
    soap_opts = [SOAP(tensor_to_matrix(w), lr=lr) for w in get_linear_weights(model)]
    for epoch in range(1, n_epochs+1):
        perm = np.random.permutation(x_train.shape[1])
        epoch_loss = 0.0
        for i in range(0, perm.size, batch_size):
            idx = perm[i:i+batch_size]
            Xb = x_train[:, idx]
            Yb = y_train_oh[:, idx]
            zero_model_grad(model)
            logits = model(Tensor(Xb, requires_grad=False))
            probs = softmax(logits)
            loss = cross_entropy(probs, Tensor(Yb, requires_grad=False))
            loss.backward()
            update_biases(model, lr)
            for opt in soap_opts:
                opt.step()
                opt.zero_grad()
            epoch_loss += loss.data
        epoch_loss /= (perm.size / batch_size)
        logits_test = model(Tensor(x_test, requires_grad=False))
        preds_test = np.argmax(softmax(logits_test).data, axis=0)
        acc = np.mean(preds_test == y_test)
        print(f"Epoch {epoch}/{n_epochs} — Loss: {epoch_loss:.4f}, Test Acc: {acc:.4f}")

# Vanilla MLP
mlp = MLP(in_features=784, hidden_features=[8, 8], out_features=num_classes)
run_experiment(mlp, "MLP")

# Two-layer MoE
moe_seq = Sequential(
    MixtureOfExperts(input_dim=784, output_dim=8, num_specialized=3, num_shared=1, k=1, gamma=0.1),
    ReLU(),
    MixtureOfExperts(input_dim=8, output_dim=8, num_specialized=3, num_shared=1, k=1, gamma=0.1),
    ReLU(),
    MixtureOfExperts(input_dim=8, output_dim=num_classes, num_specialized=4, num_shared=1, k=2, gamma=0.1),
)
run_experiment(moe_seq, "TwoLayerMoE")

# CNN
def run_experiment_cnn(model, name):
    print(f"\n=== {name} ===")
    opt = Adam(model.parameters(), lr=0.001)
    for epoch in range(1, n_epochs+1):
        perm = np.random.permutation(x_train.shape[1])
        epoch_loss = 0.0
        for i in range(0, perm.size, batch_size):
            idx = perm[i:i+batch_size]
            Xb = x_train[:, idx]  # (784, B)
            Yb = y_train_oh[:, idx]
            # reshape to (1, 28, 28, B)
            B = Xb.shape[1]
            Xb_img = Xb.T.reshape(B, 1, 28, 28).transpose(1, 2, 3, 0)
            # forward
            logits = model(Tensor(Xb_img, requires_grad=False))
            probs = softmax(logits)
            loss = cross_entropy(probs, Tensor(Yb, requires_grad=False))
            # backward + step
            for p in model.parameters():
                p.grad = None
            loss.backward()
            opt.step()
            opt.zero_grad()
            epoch_loss += loss.data
        epoch_loss /= (perm.size / batch_size)
        # test
        Btest = x_test.shape[1]
        Xtest_img = x_test.T.reshape(Btest, 1, 28, 28).transpose(1, 2, 3, 0)
        logits_test = model(Tensor(Xtest_img, requires_grad=False))
        preds_test = np.argmax(softmax(logits_test).data, axis=0)
        acc = np.mean(preds_test == y_test)
        print(f"Epoch {epoch}/{n_epochs} — Loss: {epoch_loss:.4f}, Test Acc: {acc:.4f}")

cnn = SimpleCNN(num_classes=num_classes)
run_experiment_cnn(cnn, "SimpleCNN")

# RNN (LSTM) on synthetic sequence classification
def make_sequence_dataset(num_samples=2000, T=8, input_dim=4):
    X = np.random.randn(num_samples, input_dim, T)
    # label: 1 if sum over first feature > 0 else 0
    sums = X[:, 0, :].sum(axis=1)
    y = (sums > 0).astype(int)
    y_oh = np.eye(2)[y]
    return X, y, y_oh

X_seq_train, y_seq_train, y_seq_train_oh = make_sequence_dataset(num_samples=2000)
X_seq_test, y_seq_test, y_seq_test_oh = make_sequence_dataset(num_samples=500)

def run_experiment_lstm(model, name, T=8, input_dim=4):
    print(f"\n=== {name} ===")
    opt = Adam(model.parameters(), lr=0.001)
    Bsz = 32
    epochs = 4
    N = X_seq_train.shape[0]
    for epoch in range(1, epochs+1):
        perm = np.random.permutation(N)
        epoch_loss = 0.0
        for i in range(0, N, Bsz):
            idx = perm[i:i+Bsz]
            Xb = X_seq_train[idx]  # (B, input_dim, T)
            Yb = y_seq_train_oh[idx]  # (B, 2)
            # reshape to (input_dim, T, B)
            Xb_t = Xb.transpose(1, 2, 0)
            Yb_t = Yb.T
            for p in model.parameters():
                p.grad = None
            logits = model(Tensor(Xb_t, requires_grad=False))
            probs = softmax(logits)
            loss = cross_entropy(probs, Tensor(Yb_t, requires_grad=False))
            loss.backward()
            opt.step()
            opt.zero_grad()
            epoch_loss += loss.data
        epoch_loss /= (N / Bsz)
        # test
        Xt = X_seq_test.transpose(1, 2, 0)
        logits_test = model(Tensor(Xt, requires_grad=False))
        preds = np.argmax(softmax(logits_test).data, axis=0)
        acc = np.mean(preds == y_seq_test)
        print(f"Epoch {epoch}/{epochs} — Loss: {epoch_loss:.4f}, Test Acc: {acc:.4f}")

lstm = SimpleLSTMClassifier(input_dim=4, hidden_dim=8, num_classes=2)
run_experiment_lstm(lstm, "SimpleLSTMClassifier")

