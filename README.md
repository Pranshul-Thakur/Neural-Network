**# Neural Network Basic
 Working neural network taking XOR as an example with Decision boundary
**# Neural Network from Scratch

A modular neural network implementation built from scratch using NumPy, featuring dense layers, convolutional layers, and various activation functions.

## Features

- Dense (fully connected) layers
- Convolutional layers
- Activation functions: Tanh, Sigmoid, Softmax
- Loss functions: MSE, Binary Cross-Entropy
- Forward and backward propagation
- XOR problem solver with 3D decision boundary visualization
- MNIST digit classification

## Installation

```bash
pip install numpy matplotlib scipy keras
```

## Usage

### XOR Problem
```bash
python nn.py
```
Trains a network to solve XOR and displays a 3D decision boundary plot.

### MNIST Classification
```bash
python mnist.py          # Dense network
python mnist_conv.py     # Convolutional network
```

## Network Architecture

The framework supports building custom networks by stacking layers:

```python
network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]
```

## Components

- **layer.py** - Base layer class
- **dense.py** - Fully connected layer
- **convolutional.py** - 2D convolutional layer
- **activation.py** - Base activation class
- **activations.py** - Tanh, Sigmoid, Softmax implementations
- **losses.py** - Loss functions and derivatives
- **network.py** - Training and prediction functions
- **reshape.py** - Reshaping layer for transitioning between conv and dense layers

## Examples

The repository includes working examples for:
- XOR problem with decision boundary visualization
- MNIST digit recognition (dense network)
- Binary MNIST classification (convolutional network)

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train, predict

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

# train
train(network, mse, mse_prime, X, Y, epochs=10000, learning_rate=0.1)

# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()
```




![image](https://github.com/Pranshul-Thakur/Neural-Network/assets/118863617/bb09397c-09f7-4b0c-a43e-43a050a28aeb)
