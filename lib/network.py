"""
Neural Network Model
Implements Sequential model for stacking layers
"""

import numpy as np
import pickle

class Sequential:
    def __init__(self):
        
        self.layers = []
        self.loss_function = None
        self.optimizer = None
    
    def add(self, layer):
        
        self.layers.append(layer)
    
    def compile(self, loss, optimizer):
        
        self.loss_function = loss
        self.optimizer = optimizer
    
    def forward(self, X):
        
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, loss_gradient):
        
        gradient = loss_gradient
        
        # Iterate through layers in reverse order
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
    
    def train_step(self, X, y):
        
        # Step 1: Forward pass
        predictions = self.forward(X)
        
        # Step 2: Compute loss
        loss = self.loss_function.loss(y, predictions)
        
        # Step 3: Compute gradient of loss w.r.t predictions
        loss_gradient = self.loss_function.gradient(y, predictions)
        
        # Step 4: Backward pass
        self.backward(loss_gradient)
        
        # Step 5: Update parameters
        self.optimizer.update(self.layers)
        
        return loss
    
    def fit(self, X, y, epochs, batch_size=None, verbose=True):
        
        history = {'loss': []}
        
        # Default to full batch if not specified
        if batch_size is None:
            batch_size = X.shape[0]
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            
            # Shuffle data at start of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Train on mini-batches
            for i in range(0, n_samples, batch_size):
                # Get batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Train on batch
                loss = self.train_step(X_batch, y_batch)
                
                epoch_loss += loss
                n_batches += 1
            
            # Average loss for epoch
            avg_loss = epoch_loss / n_batches
            history['loss'].append(avg_loss)
            
            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
        return history
    
    def predict(self, X):
        
        return self.forward(X)
    
    def save(self, path):
        """
        Save full model: layers, loss function, optimizer
        """
        with open(path, "wb") as f:
            pickle.dump({
                "layers": self.layers,
                "loss_function": self.loss_function,
                "optimizer": self.optimizer
            }, f)

    @staticmethod
    def load(path):
        """
        Load full model
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        model = Sequential()
        model.layers = data["layers"]
        model.loss_function = data["loss_function"]
        model.optimizer = data["optimizer"]

        return model
    

    