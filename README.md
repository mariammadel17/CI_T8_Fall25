# Neural Network Library - From Scratch
**Course:** CSE473s - Computational Intelligence  
**Team:** 8

## What is this project?

We built a complete neural network library from scratch using only NumPy. The implementation includes forward propagation, backpropagation, multiple layer types, activation functions, loss functions, and optimizers. The project validates the implementation with three progressive phases:

## Project Phases

**Phase 1:** XOR Problem - Validate the library with gradient checking  
**Phase 2:** Autoencoder - MNIST digit reconstruction and comparison with TensorFlow  
**Phase 3:** Feature Learning - Extract latent features and train SVM classifier

## Installation

Install dependencies with Python 3.8 or higher:

```bash
pip install -r requirements.txt
```

**Required packages:**
- **NumPy** - Mathematical operations
- **Matplotlib** - Plotting and visualization
- **Jupyter** - Running interactive notebooks
- **scikit-learn** - SVM classifier (Phase 3)
- **TensorFlow/Keras** - Comparison benchmarks

## Running the Project

Open the Jupyter notebook to run all phases:

```bash
jupyter notebook notebooks/project_demo.ipynb
```

The notebook is organized into 19 cells across three phases:

### Phase 1: XOR Problem Validation (Cells 1-7)

**Cell 1:** Import Libraries
- Load NumPy, Matplotlib, and our custom library

**Cell 2-3:** Gradient Checking
- Validates backpropagation correctness
- Computes analytical vs numerical gradients
- Must pass with relative difference < 1e-5

**Cell 4:** XOR Dataset
- Creates XOR truth table (4 samples)
- Visualizes the non-linearly separable problem

**Cell 5:** Build and Train Network
- Architecture: 2 → 4 (Tanh) → 1 (Sigmoid)
- Training: 5000 epochs with SGD (lr=0.1)
- Takes ~10 seconds

**Cell 6:** Results Analysis
- Shows predictions vs targets
- Achieves 100% binary accuracy

**Cell 7:** Training Visualization
- Plots loss curve showing convergence

### Phase 2: MNIST Autoencoder (Cells 8-12)

**Cell 8:** Load MNIST Data
- 30,000 training + 5,000 test samples
- Flattened to 784 dimensions
- Normalized to [0, 1] range

**Cell 9:** Build Autoencoder
- Encoder: 784 → 256 → 128 → 64 (bottleneck)
- Decoder: 64 → 128 → 256 → 784
- ReLU hidden layers, Sigmoid output
- SGD optimizer (lr=0.005)

**Cell 10:** Train Autoencoder
- 100 epochs with batch size 128
- Saves model to 'autoencoder_Model.pkl'

**Cell 11:** Training Visualization
- Plots training loss curve
- Evaluates reconstruction loss on test set

**Cell 12:** Visual Reconstruction
- Displays 10 random original vs reconstructed images
- Qualitative assessment of reconstruction quality

### Phase 3: Feature Learning & SVM Classification (Cells 13-15)

**Cell 13:** Build Encoder
- Extracts first 6 layers (encoder portion)
- Creates 64-dimensional latent representations
- Converts training and test data to latent space

**Cell 14:** Train SVM Classifier
- Uses RBF kernel on latent features
- Trains on 30,000 samples with 10 digit classes
- Tests on 5,000 samples

**Cell 15:** Classification Results
- Detailed classification report per digit
- Confusion matrix visualization
- Overall test accuracy reported

### Phase 4: TensorFlow Comparison (Cells 16-19)

**Cell 16:** TensorFlow XOR
- Recreates XOR with Keras
- Identical architecture and settings
- Compares training curves

**Cell 17A:** TensorFlow Autoencoder (SGD)
- Exact same architecture as our library
- Uses SGD optimizer (lr=0.005)
- Compares loss convergence

**Cell 17B:** TensorFlow Autoencoder (Adam)
- Same architecture with Adam optimizer
- Demonstrates optimizer impact

**Cell 18:** Three-Way Training Comparison
- XOR: Our library vs TensorFlow
- Autoencoder: Our library (SGD) vs TensorFlow (SGD) vs TensorFlow (Adam)
- Visual loss curve comparison

**Cell 19:** Four-Way Reconstruction Comparison
- Original images
- Our library reconstructions
- TensorFlow (SGD) reconstructions
- TensorFlow (Adam) reconstructions

## Project Results

### Phase 1: XOR Problem

| Metric | Value |
|--------|-------|
| Network Architecture | 2 → 4 → 1 |
| Training Epochs | 5000 |
| Optimizer | SGD (lr=0.1) |
| Final Training Loss | 0.0028 |
| Binary Accuracy | 100% |
| Gradient Check | PASSED ✓ |

**Predictions:**
```
Input      Prediction  Target
[0, 0]     0.0453      0
[0, 1]     0.9353      1
[1, 0]     0.9440      1
[1, 1]     0.0407      0
```

### Phase 2: MNIST Autoencoder

| Metric | Value |
|--------|-------|
| Architecture | 784→256→128→64→128→256→784 |
| Training Samples | 30,000 |
| Test Samples | 5,000 |
| Training Epochs | 100 |
| Batch Size | 128 |
| Optimizer | SGD (lr=0.005) |
| Final Training Loss | 0.00567 |
| Test Reconstruction Loss | 0.00598 |

### Phase 3: SVM Classification on Latent Features

| Metric | Value |
|--------|-------|
| Latent Dimension | 64 |
| Classifier | SVM (RBF kernel) |
| Training Samples | 30,000 |
| Test Samples | 5,000 |
| **Test Accuracy** | **98.42%** |

Detailed per-digit precision/recall available in confusion matrix visualization.

### Phase 4: TensorFlow Comparison

**XOR Problem:**
- Our Library: Loss = 0.0028
- TensorFlow: Loss = 0.0041
- Both achieve 100% accuracy

**Autoencoder (100 epochs):**

| Implementation | Optimizer | Training Time | Final Loss | Test Loss |
|---|---|---|---|---|
| Our Library | SGD | 180s | 0.00567 | 0.00598 |
| TensorFlow | SGD | 45s | 0.0412 | 0.0418 |
| TensorFlow | Adam | 48s | 0.0103 | 0.0107 |

**Key Observations:**
1. Our SGD implementation found superior local minimum vs TensorFlow SGD
2. Adam optimizer requires tuning but achieved better results
3. Library produces sharper reconstructions with SGD
4. Training time difference reflects implementation vs optimized framework

## Library Architecture

### Core Components

```
lib/
├── layers.py          # Dense (fully connected) layer
├── activations.py     # ReLU, Sigmoid, Tanh, Softmax
├── losses.py          # MSE loss function
├── optimizer.py       # SGD optimizer
└── network.py         # Sequential model container
```

### Layers
- **Dense Layer**: Fully connected layer
  - Forward: `y = Wx + b`
  - He initialization for weights (optimal for ReLU)
  - Gradient computation for backpropagation

### Activation Functions
- **ReLU**: `f(x) = max(0, x)` - Non-linearity, prevents vanishing gradients
- **Sigmoid**: `f(x) = 1/(1+e^-x)` - Outputs [0, 1], used for binary output
- **Tanh**: `f(x) = tanh(x)` - Outputs [-1, 1], stronger gradients than sigmoid
- **Softmax**: Converts scores to probability distribution

### Loss Function
- **MSE**: `L = mean((y_pred - y_true)²)` - Measures reconstruction/prediction error

### Optimizer
- **SGD**: `w = w - lr * ∇L(w)` - Stochastic gradient descent with fixed learning rate

### Model
- **Sequential**: Stack layers linearly, train end-to-end with backpropagation

## Quick Example: Solving XOR

XOR is a classic non-linearly separable problem that requires hidden layers. Our implementation solves it perfectly:

```python
import numpy as np
from lib import Dense, Tanh, Sigmoid, MSE, SGD, Sequential

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Build network: 2 inputs -> 4 hidden -> 1 output
model = Sequential()
model.add(Dense(2, 4))          # Input to hidden
model.add(Tanh())               # Non-linearity
model.add(Dense(4, 1))          # Hidden to output
model.add(Sigmoid())            # Squash to [0, 1]

# Setup training
model.compile(loss=MSE(), optimizer=SGD(learning_rate=0.1))

# Train
history = model.fit(X, y, epochs=5000)

# Test - should predict [0, 1, 1, 0]
predictions = model.predict(X)
print(predictions)
```

## How It Works: The Training Process

### 1. Forward Propagation
Data flows through each layer sequentially:
```
Input → Dense₁ → Activation → Dense₂ → Activation → Output
```

Each Dense layer computes: `output = input @ weights + bias`  
Each Activation applies: `output = activation_function(input)`

### 2. Loss Calculation
After forward pass, compute reconstruction/prediction error:
```
Loss = MSE(predictions, targets) = mean((pred - target)²)
```

### 3. Backward Propagation
Gradients flow backward using the chain rule:
```
∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w
```

Each layer computes:
- Gradient w.r.t. parameters (weights, biases) → used for updates
- Gradient w.r.t. inputs → passed to previous layer

### 4. Weight Updates
Optimizer adjusts parameters to minimize loss:
```
w_new = w_old - learning_rate · ∂L/∂w
```

### 5. Repeat
Iterate: forward → loss → backward → update for multiple epochs

## Gradient Checking: Validating Correctness

We verify backpropagation by comparing two gradient computations:

**Analytical Gradient** (from chain rule):
- Computed via backpropagation algorithm
- Efficient but complex to derive

**Numerical Gradient** (finite differences):
- Simple formula: `∇f(x) ≈ (f(x+ε) - f(x-ε)) / (2ε)`
- Slower but mathematically straightforward

If they match, backpropagation is correct:
```
relative_diff = ||analytical - numerical|| / (||analytical|| + ||numerical||)
if relative_diff < 1e-5:
    ✓ Implementation is correct
else:
    ✗ Bug in gradient computation
```

**Result for our library:** relative_diff < 1e-9 ✓

## Autoencoders: Architecture & Purpose

An autoencoder learns compressed representations of data:

```
Input (784) → Encoder → Bottleneck (64) → Decoder → Reconstruction (784)
```

**Encoder** learns to compress images into low-dimensional features  
**Decoder** reconstructs images from compressed features  
**Loss** measures how well reconstructions match originals

This forces the network to learn important features in the bottleneck.

## SVM Classification on Latent Features

After training the autoencoder:
1. Extract the encoder (all layers up to bottleneck)
2. Generate 64-dimensional latent vectors for all MNIST digits
3. Train SVM classifier on latent features
4. Achieve 98.42% accuracy with only 64 features instead of 784

This demonstrates that autoencoders create useful feature representations for downstream tasks.

## File Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── lib/
│   ├── __init__.py                   # Package initialization
│   ├── layers.py                     # Dense layer implementation
│   ├── activations.py                # ReLU, Sigmoid, Tanh, Softmax
│   ├── losses.py                     # MSE loss
│   ├── optimizer.py                  # SGD optimizer
│   ├── network.py                    # Sequential model
│   └── __pycache__/                  # Compiled Python files
└── notebooks/
    └── project_demo.ipynb            # Complete 19-cell demonstration
```
## Dependencies

See [requirements.txt](requirements.txt):
- **NumPy 1.21+** - Numerical computation
- **Matplotlib 3.4+** - Visualization
- **Jupyter 1.0+** - Interactive notebooks
- **scikit-learn 0.24+** - SVM classifier
- **TensorFlow 2.7+** - Comparison benchmarks
---
**Created:** December 2025  
**Status:** Complete (All 3 phases + TensorFlow benchmarking)  
**Next Steps:** Experiment with different architectures, optimizers, and datasets!
