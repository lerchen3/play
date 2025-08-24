import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self.prev_tensors = None  # Store input tensors that created this tensor
    
    @property
    def is_leaf(self):
        return not self.requires_grad or self.grad_fn is None
    
    def backward(self, gradient=None):
        """Computes the gradient of current tensor w.r.t. graph leaves."""
        if gradient is None:
            if self.data.size == 1:  # scalar
                gradient = 1.0
            else:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
                
        # Accumulate gradient if it exists
        if self.grad is None:
            self.grad = gradient
        else:
            self.grad += gradient
            
        # Propagate gradients to inputs if they exist
        if self.grad_fn is not None:
            input_grads = self.grad_fn(gradient)
            if not isinstance(input_grads, tuple):
                input_grads = (input_grads,)
            
            # Pass gradients to input tensors
            for input_tensor, grad in zip(self.prev_tensors, input_grads):
                if isinstance(input_tensor, Tensor) and grad is not None and input_tensor.requires_grad:
                    input_tensor.backward(grad)
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        from operations import elementwise_add
        return elementwise_add(self, other)

    def __radd__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        from operations import elementwise_add
        return elementwise_add(other, self)
