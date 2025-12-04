"""
Neural Network Layers
"""

import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    """
    Base class for all neural network layers
    
    All layers must implement:
    - forward(input): Forward propagation
    - backward(output_gradient): Backward propagation
    """
    
    def __init__(self):
        self.input = None
        self.output = None
    
    @abstractmethod
    def forward(self, input):
        
        pass
    
    @abstractmethod
    def backward(self, output_gradient, learning_rate=None):
        """   
        Returns:
            Gradient of loss with respect to input
        """
        pass


class Dense(Layer):
    """
    
    Performs: output = input @ weights + bias
    """
    
    def __init__(self, input_size, output_size):
        super().__init__()
        
        # Initialize weights using He initialization (good for ReLU)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        
        # Initialize bias to zeros
        self.bias = np.zeros((1, output_size))
        
        # Placeholders for gradients
        self.weights_gradient = None
        self.bias_gradient = None
    
    def forward(self, input):
        """
            Output = input @ weights + bias
        """
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        """
        Backward pass using chain rule
        
        Given dL/dY, compute:
        - dL/dW = X^T @ dL/dY
        - dL/db = sum(dL/dY, axis=0)
        - dL/dX = dL/dY @ W^T

        """
        # Gradient w.r.t weights: dL/dW = X^T @ dL/dY
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        
        # Gradient w.r.t bias: dL/db = sum of dL/dY over batch dimension
        self.bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Gradient w.r.t input: dL/dX = dL/dY @ W^T
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        return input_gradient
    
    def get_params(self):
        """Return list of trainable parameters"""
        return [self.weights, self.bias]
    
    def get_gradients(self):
        """Return list of parameter gradients"""
        return [self.weights_gradient, self.bias_gradient]