"""
Loss Functions
Implements Mean Squared Error (MSE)
"""

import numpy as np


class MSE:
    """
    Mean Squared Error Loss
    
    Formula: L = (1/N) * sum((y_true - y_pred)^2)
    """
    
    @staticmethod
    def loss(y_true, y_pred):
        
        return np.mean(np.power(y_true - y_pred, 2))
    
    @staticmethod
    def gradient(y_true, y_pred):
        """ 
        Derivative: dL/dy_pred = (2/N) * (y_pred - y_true)
        """
        n = y_true.shape[0]  # Number of samples
        return (2 / n) * (y_pred - y_true)