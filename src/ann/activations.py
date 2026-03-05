import numpy as np


class Sigmoid:
   

    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        s = Sigmoid.forward(x)
        return s * (1 - s)


class Tanh:
    

    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        t = Tanh.forward(x)
        return 1 - t**2


class ReLU:

    @staticmethod
    def forward(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        return (x > 0).astype(float)


def get_activation(name: str):
    name = name.lower()

    activations = {
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "relu": ReLU,
    }

    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")

    return activations[name]
