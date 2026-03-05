
"""
Optimizer implementations
"""

import numpy as np


class SGD:
    """Stochastic Gradient Descent."""

    def __init__(self, lr,weight_decay=0):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, layers):
        for layer in layers:
            grad_W = layer.grad_W + self.weight_decay * layer.W 
            grad_b = layer.grad_b
            layer.W -= self.lr * grad_W
            layer.b -= self.lr * grad_b


class Momentum:
    """SGD with Momentum."""

    def __init__(self, lr,weight_decay=0, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.vW = None
        self.vb = None

    def _initialize(self, layers):
        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]

    def step(self, layers):

        if self.vW is None:
            self._initialize(layers)

        for i, layer in enumerate(layers):

            grad_W = layer.grad_W + self.weight_decay * layer.W 
            grad_b = layer.grad_b

            self.vW[i] = self.beta * self.vW[i] + (1 - self.beta) * grad_W
            self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * grad_b

            layer.W -= self.lr * self.vW[i]
            layer.b -= self.lr * self.vb[i]


class NAG:
    """Nesterov Accelerated Gradient."""

    def __init__(self, lr, weight_decay=0,beta=0.9):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.vW = None
        self.vb = None

    def _initialize(self, layers):
        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]

    def step(self, layers):

        if self.vW is None:
            self._initialize(layers)

        for i, layer in enumerate(layers):
                grad_W = layer.grad_W + self.weight_decay * layer.W  # apply weight decay
                grad_b = layer.grad_b

                v_prev_W = self.vW[i]
                self.vW[i] = self.beta * self.vW[i] + self.lr * grad_W
                layer.W -= (1 + self.beta) * self.vW[i] - self.beta * v_prev_W

                v_prev_b = self.vb[i]
                self.vb[i] = self.beta * self.vb[i] + self.lr * grad_b
                layer.b -= (1 + self.beta) * self.vb[i] - self.beta * v_prev_b

class RMSprop:
    """RMSprop optimizer."""

    def __init__(self, lr, weight_decay=0,beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.eps = eps

        self.sW = None
        self.sb = None

    def _initialize(self, layers):
        self.sW = [np.zeros_like(l.W) for l in layers]
        self.sb = [np.zeros_like(l.b) for l in layers]

    def step(self, layers):

        if self.sW is None:
            self._initialize(layers)

        for i, layer in enumerate(layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W 
            grad_b = layer.grad_b

            self.sW[i] = self.beta * self.sW[i] + (1 - self.beta) * (grad_W ** 2)
            self.sb[i] = self.beta * self.sb[i] + (1 - self.beta) * (grad_b ** 2)

            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.sW[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.sb[i]) + self.eps)


def get_optimizer(name, lr,weight_decay):
    """Factory function for optimizers."""

    name = name.lower()

    optimizers = {
        "sgd": SGD,
        "momentum": Momentum,
        "nag": NAG,
        "rmsprop": RMSprop,
    }

    if name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {name}")

    return optimizers[name](lr,weight_decay)



