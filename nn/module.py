from collections import OrderedDict
from tensor import Tensor

class Module:
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self.training = True

    def register_parameter(self, name: str, param: Tensor):
        """Add a parameter to the module."""
        if param is None:
            self._parameters[name] = None
        else:
            self._parameters[name] = param
            object.__setattr__(self, name, param)

    def register_module(self, name: str, module: 'Module'):
        """Add a child module to the module."""
        self._modules[name] = module
        object.__setattr__(self, name, module)

    def parameters(self):
        """Returns an iterator over module parameters."""
        for param in self._parameters.values():
            if param is not None:
                yield param
        
        for module in self._modules.values():
            for param in module.parameters():
                yield param

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self.register_module(name, value)
        elif isinstance(value, Tensor):
            self.register_parameter(name, value)
        else:
            object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError 