import numpy as np
"""
Functional implementation of a simple forward-feeding neural network for Policy Gradient (REINFORCE).
Input is a 4-dimensional array, passed through a ReLU-activated hidden layer for a softmax probability distribution output.
"""


lr = 0.01

input_size = 4 
hidden_size = 16
output_size = 3

# Xavier Initialization and random initizalization for the biases
W1 = np.random.randn(input_size, hidden_size)  * np.sqrt(6 / (input_size + hidden_size))
b1 = np.random.randn(hidden_size)

W2 = np.random.randn(hidden_size, output_size) * np.sqrt( 6 / (hidden_size + output_size))
b2 = np.random.randn(output_size)

def forward(X, W1, b1, W2, b2):
    """
    Forward pass through the neural network.
    
    X (np.ndarray): Input vector of shape (input_size,).
    """
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2 + b2
    probs = softmax(z2)
    
    return probs, z1, a1, z2

def softmax(X):
    """
    Applies the softmax function to convert logits to a probability distribution.
    Returns np.ndarray of probabilities summing to 1
    """
    e_X = np.exp(X) - np.mean(X)
    return e_X / np.sum(e_X)


def backward(X, action_taken, advantage, probs, z1, a1, W2):
    """
    Computes gradients for all network parameters using REINFORCE method.
    
    X (np.ndarray): Input vector used in forward pass.
    action_taken (int): Index of the action taken.
    advantage (float): Reward signal
    probs (np.ndarray): Output probabilities from forward().
    z1 (np.ndarray): Pre-activation values of the hidden layer.
    a1 (np.ndarray): Activation values from the hidden layer.
    W2 (np.ndarray): Weights from hidden to output layer.
    """
    dlog = probs.copy()
    dlog[action_taken] -= 1
    dlog *= advantage

    dW2 = np.outer(a1, dlog)
    db2 = dlog
    
    da1 = W2 @ dlog
    dz1 = da1 * (z1 > 0)

    dW1 = np.outer(X, dz1)
    db1 = dz1

    return dW1, db1, dW2, db2


def update_param(W1, b1, W2, b2, dW1, db1, dW2, db2):
    """
    Applies gradient descent update to all network parameters.
    
    W1, b1, W2, b2 (np.ndarray): Current weights and biases.
    dW1, db1, dW2, db2 (np.ndarray): Computed gradients.
    """
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    
    return W1, b1, W2, b2
    