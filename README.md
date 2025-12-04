# Neural Network Library - From Scratch

A simple neural network library built with only NumPy for CSE473s: Computational Intelligence course project , Team : 9

## What is this project?
we build a complete neural network library without using any machine learning frameworks. We only use NumPy for mathematical operations and implement everything else ourselves - forward propagation, backpropagation, layers, activations, and optimizers.

## Project Phases

**Phase 1 (Current):** Build the library and validate it with XOR problem
**Phase 2:** Build an autoencoder for MNIST digit reconstruction  
**Phase 3:** Use encoder features to train an SVM classifier

## Installation

You need Python 3.8 or higher. Install dependencies:

```bash
pip install -r requirements.txt
```

This installs:
- NumPy (for math operations)
- Matplotlib (for plotting graphs)
- Jupyter (for running notebooks)



The notebook is organized into sections. For Phase 1, run these cells in order:

**Cell 1: Import Libraries**
- Imports NumPy, Matplotlib, and our library
- Run this first

**Cells 2-3: Gradient Checking**
- Tests that backpropagation is mathematically correct
- Should print "✅ Gradient check PASSED!"
- If it fails, there's a bug in backward pass

**Cell 4: XOR Dataset**
- Creates the XOR problem data
- Shows the truth table

**Cell 5: Build and Train Model**
- Creates a 2→4→1 network
- Trains for 5000 epochs
- Takes about 10 seconds

**Cell 6: View Results**
- Shows predictions vs targets
- Should show 100% accuracy

**Cell 7: Plot Training Curve**
- Displays loss over time
- Loss should decrease smoothly

### Step 5: Understanding the Output

You should see:

```
Epoch 1/5000 - Loss: 0.302723
Epoch 101/5000 - Loss: 0.269113
...
Epoch 5000/5000 - Loss: 0.002759

FINAL RESULTS
═══════════════════════════════════════════
Predictions:
Input           Prediction      Target    
--------------------------------------------------
[0. 0.]           0.0453              0
[0. 1.]           0.9353              1
[1. 0.]           0.9440              1
[1. 1.]           0.0407              0

Binary Accuracy: 100.00%
```
## Project Structure

```
lib/                        # Our neural network library
  ├── layers.py             # Dense layer implementation
  ├── activations.py        # ReLU, Sigmoid, Tanh, Softmax
  ├── losses.py             # MSE loss function
  ├── optimizer.py          # SGD optimizer
  └── network.py            # Sequential model

notebooks/
  └── project_demo.ipynb    # Complete demo with all phases

README.md                   # This file
requirements.txt            # Python packages needed
```

## What's in the library?

### Layers
- **Dense Layer**: Fully connected layer with weights and biases
- Uses He initialization for weights (good for ReLU)

### Activation Functions
- **ReLU**: f(x) = max(0, x) - most common for hidden layers
- **Sigmoid**: f(x) = 1/(1+e^-x) - squashes output to 0-1
- **Tanh**: f(x) = tanh(x) - squashes output to -1 to 1
- **Softmax**: Converts scores to probabilities

### Loss Function
- **MSE**: Mean Squared Error - measures prediction error

### Optimizer
- **SGD**: Stochastic Gradient Descent - updates weights to minimize loss

### Model
- **Sequential**: Stack layers in order and train them together

## Quick Example: Solving XOR

XOR is a classic problem that can't be solved with a single layer. Our network learns it perfectly:

```python
import numpy as np
from lib import Dense, Tanh, Sigmoid, MSE, SGD, Sequential

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Build network: 2 inputs -> 4 hidden -> 1 output
model = Sequential()
model.add(Dense(2, 4))
model.add(Tanh())
model.add(Dense(4, 1))
model.add(Sigmoid())

# Setup training
model.compile(loss=MSE(), optimizer=SGD(learning_rate=0.1))

# Train
history = model.fit(X, y, epochs=5000)

# Test
predictions = model.predict(X)
print(predictions)  # Should be close to [0, 1, 1, 0]
```


### Forward Pass
Data flows through layers: Input → Dense → Activation → Dense → Activation → Output

Each layer does:
1. Dense: output = input @ weights + bias
2. Activation: output = activation_function(input)

### Backward Pass
Gradients flow backward: Loss → Output → ... → Input

Each layer:
1. Receives gradient from next layer
2. Computes gradient for its parameters (weights, bias)
3. Passes gradient to previous layer

### Training Loop
For each batch of data:
1. Forward pass to get predictions
2. Calculate loss (how wrong we are)
3. Backward pass to get gradients
4. Update weights using optimizer

## Validating the Implementation

We use gradient checking to verify backpropagation is correct:

```python
# Compute gradient numerically
numerical_gradient = (loss(w + ε) - loss(w - ε)) / (2ε)

# Compare with our analytical gradient
difference = |analytical_gradient - numerical_gradient|

# Should be very small (< 0.00001)
if difference < 1e-5:
    print("Backpropagation is correct!")
```

## Results (Phase 1)

**XOR Problem:**
- Training: 5000 epochs
- Final Loss: 0.0028
- Accuracy: 100%
- All 4 XOR cases predicted correctly

**Gradient Check:**
- Difference: < 0.000000001
- Status: Passed ✓
---

*This is Phase 1 of a 3-phase project. The library works correctly and is ready for the autoencoder implementation in Phase 2.*