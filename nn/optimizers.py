import numpy as np
from tensor import Tensor

class Adam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, max_grad_norm=5.0):
        """Adam optimizer w/ grad clipping.
        
        Args:
            parameters: Iterable of Tensor objects containing parameters to optimize
            lr: Learning rate (default: 0.001)
            betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
            eps: Term added to denominator to improve numerical stability (default: 1e-8)
            weight_decay: Weight decay (L2 penalty) (default: 0)
            max_grad_norm: Maximum norm for gradient clipping (default: 5.0)
        """
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        
        # Initialize momentum and velocity states
        self.state = {
            'step': 0,
            'm': [np.zeros_like(p.data) for p in self.parameters],  # First moment estimates
            'v': [np.zeros_like(p.data) for p in self.parameters]   # Second moment estimates
        }
    
    def clip_gradients(self):
        """Clips gradients by global norm."""
        # Calculate global norm of all gradients
        total_norm = 0
        for p in self.parameters:
            if p.grad is not None:
                param_norm = np.linalg.norm(p.grad)
                total_norm += param_norm ** 2
        total_norm = np.sqrt(total_norm)
        
        # Apply clipping
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.parameters:
                if p.grad is not None:
                    p.grad *= clip_coef

    def step(self):
        """Performs a single optimization step."""
        self.state['step'] += 1
        
        # Clip gradients
        if self.max_grad_norm > 0:
            self.clip_gradients()
        
        # Bias correction terms: want, in expectation, 1 - beta^t = (1-beta)(1+beta+beta^2+...+beta^(t-1))
        bias_correction1 = 1 - self.beta1 ** self.state['step']
        bias_correction2 = 1 - self.beta2 ** self.state['step']
        
        for i, p in enumerate(self.parameters):
            if p.grad is not None:
                grad = p.grad
                
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * p.data
                
                # Update momentum (first moment estimate)
                self.state['m'][i] = self.beta1 * self.state['m'][i] + (1 - self.beta1) * grad
                
                # Update velocity (second moment estimate)
                self.state['v'][i] = (self.beta2 * self.state['v'][i] + 
                                    (1 - self.beta2) * np.square(grad))
                
                # Bias-corrected estimates
                m_hat = self.state['m'][i] / bias_correction1
                v_hat = self.state['v'][i] / bias_correction2
                
                # Update parameters
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Sets gradients of all parameters to None."""
        for p in self.parameters:
            p.grad = None

class SOAP:
    # Takes in a particular matrix A of tensor objects.
    def __init__(self, A, lr=0.001, beta=0.9, eps=1e-8, space_update_freq=10):
        """SOAP optimizer implemented verbatim from https://arxiv.org/pdf/2409.11321
        
        Args:
            A: m x n matrix (list of lists) of Tensor objects
            lr: learning rate
            beta: momentum decay factor for EMA updates
            eps: term added to the denominator for numerical stability
            space_update_freq: frequency (in steps) to update the eigenspaces
        """
        self.A = A  # m x n matrix of Tensor objects
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.space_update_freq = space_update_freq
        
        self.step_count = 0
        self.m = len(A)
        self.n = len(A[0]) if self.m > 0 else 0
        
        # Initialize eigenspaces as identity matrices
        self.QL = np.eye(self.m)
        self.QR = np.eye(self.n)
        
        # Initialize EMA variables for the transformed gradients
        self.M_prime = None  # EMA of transformed gradients
        self.V_prime = None  # EMA of squared transformed gradients
        
        # Initialize covariance matrices for the transformed gradients
        self.L = np.zeros((self.m, self.m))
        self.R = np.zeros((self.n, self.n))
    
    def step(self):
        """Performs a single optimization step using the SOAP algorithm."""
        self.step_count += 1
        
        # Construct gradient matrix G from A (assume scalar gradients)
        G = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                grad = self.A[i][j].grad if self.A[i][j].grad is not None else 0.0
                G[i, j] = grad
        
        # Transform gradients: G_prime = QL^T * G * QR
        G_prime = self.QL.T.dot(G).dot(self.QR)
        
        # Initialize or update EMA estimates for the transformed gradients
        if self.M_prime is None:
            self.M_prime = G_prime
            self.V_prime = np.square(G_prime)
        else:
            self.M_prime = self.beta * self.M_prime + (1 - self.beta) * G_prime
            self.V_prime = self.beta * self.V_prime + (1 - self.beta) * np.square(G_prime)
        
        # Compute the normalized update in the transformed space
        N_prime = self.M_prime / (np.sqrt(self.V_prime) + self.eps)
        
        # Rotate back to original space: N = QL * N_prime * QR^T
        N = self.QL.dot(N_prime).dot(self.QR.T)
        
        # Update each tensor in A using the computed update
        for i in range(self.m):
            for j in range(self.n):
                self.A[i][j].data -= self.lr * N[i, j]
        
        # Update covariance matrices L and R as EMAs of the transformed gradients
        cov_L = G_prime.dot(G_prime.T)  # m x m covariance
        cov_R = G_prime.T.dot(G_prime)  # n x n covariance
        self.L = self.beta * self.L + (1 - self.beta) * cov_L
        self.R = self.beta * self.R + (1 - self.beta) * cov_R
        
        # Periodically update the eigenspaces using subspace iteration
        if self.step_count % self.space_update_freq == 0:
            self.QL = self._update_eigenspace(self.L, self.QL)
            self.QR = self._update_eigenspace(self.R, self.QR)
    
    def _update_eigenspace(self, P, Q):
        """Updates the eigenspace using a QR decomposition based subspace iteration."""
        S = np.dot(P, Q)
        Q_new, _ = np.linalg.qr(S)
        return Q_new
    
    def zero_grad(self):
        """Sets gradients of all tensors in A to None."""
        for row in self.A:
            for tensor in row:
                tensor.grad = None

    @staticmethod
    def Eigenvectors_old(P):
        """Compute the eigenvectors of matrix P using the traditional method."""
        return np.linalg.eigh(P)[1]
    