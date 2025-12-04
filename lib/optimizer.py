"""
Optimization Algorithms
Implements Stochastic Gradient Descent (SGD)
"""

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent Optimizer
    
    Updates parameters using the rule:
    θ_new = θ_old - η * ∇L(θ)
    
    Where:
    - θ: parameters (weights, biases)
    - η: learning rate (step size)
    - ∇L(θ): gradient of loss w.r.t parameters
    """
    
    def __init__(self, learning_rate=0.01):
        
        self.learning_rate = learning_rate
    
    def update(self, layers):
        
        for layer in layers:
            # Check if layer has trainable parameters
            if hasattr(layer, 'weights') and hasattr(layer, 'bias'):
                # Update weights: W = W - η * dL/dW
                layer.weights -= self.learning_rate * layer.weights_gradient
                
                # Update bias: b = b - η * dL/db
                layer.bias -= self.learning_rate * layer.bias_gradient
    
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate