class Context:
    def __init__(self):
        self._saved_tensors = []
        
    def save_for_backward(self, *tensors):
        """Save tensors needed for backward pass"""
        self._saved_tensors = tensors
        
    @property 
    def saved_tensors(self):
        """Return tensors saved for backward pass"""
        return self._saved_tensors 