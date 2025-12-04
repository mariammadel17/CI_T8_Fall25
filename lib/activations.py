"""
Activation Functions
Implements ReLU, Sigmoid, Tanh, and Softmax
"""

import numpy as np
from .layers import Layer


class ReLU(Layer):
    """
    Rectified Linear Unit
    
    Forward:  f(x) = max(0, x)
    Backward: f'(x) = 1 if x > 0, else 0
    """
    
    def forward(self, input):
       
        self.input = input
        self.output = np.maximum(0, input)
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        
        return output_gradient * (self.input > 0)


class Sigmoid(Layer):
    """
    Sigmoid (Logistic) Activation
    
    Forward:  f(x) = 1 / (1 + e^(-x))
    Backward: f'(x) = f(x) * (1 - f(x))
    
    Squashes input to range (0, 1)
  
    """
    
    def forward(self, input):
       
        self.input = input
        # Clip input to prevent overflow
        clipped_input = np.clip(input, -500, 500)
        self.output = 1 / (1 + np.exp(-clipped_input))
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
       
        sigmoid_derivative = self.output * (1 - self.output)
        return output_gradient * sigmoid_derivative


class Tanh(Layer):
    """
    Hyperbolic Tangent
    
    Forward:  f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Backward: f'(x) = 1 - f(x)^2
    
    Squashes input to range (-1, 1)
    """
    
    def forward(self, input):
        
        self.input = input
        self.output = np.tanh(input)
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        
        tanh_derivative = 1 - np.power(self.output, 2)
        return output_gradient * tanh_derivative


class Softmax(Layer):
    """
    Softmax Activation
    
    Forward:  f(x_i) = e^(x_i) / sum(e^(x_j) for all j)
    
    Converts raw scores (logits) into probabilities:
    - Output values are in range (0, 1)
    - Output values sum to 1
    
    """
    
    def forward(self, input):
        
        self.input = input
        
        # Subtract max for numerical stability
        # This doesn't change the result due to properties of exp
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        
        # Normalize to get probabilities
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        
        # Simplified gradient (works well in practice with MSE)
        return output_gradient