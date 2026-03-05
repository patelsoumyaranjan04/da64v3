import numpy as np
from .activations import get_activation


class NeuralLayer:
    """Fully connected neural network layer."""

    def __init__(self, in_dim, out_dim, activation=None, weight_init="xavier"):

        self.in_dim = in_dim
        self.out_dim = out_dim

        # activation handler
        self.activation = get_activation(activation) if activation else None

        # parameters
        self.W = self._initialize_weights(in_dim, out_dim, weight_init)
        self.b = np.zeros((1, out_dim), dtype=np.float32)

        # caches used in backprop
        self._input = None
        self._z = None

        # gradients
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def _initialize_weights(self, in_dim, out_dim, method):
        """Initialize weight matrix."""

        if method == "zeros":
            return np.zeros((in_dim, out_dim), dtype=np.float32)

        if method == "random":
            return np.random.randn(in_dim, out_dim).astype(np.float32) * 0.01

        if method == "xavier":
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            return np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)

        raise ValueError(f"Unknown weight initialization: {method}")

    def forward(self, X):
        """
        Forward propagation through the layer.
        """

        if X.shape[1] != self.in_dim:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.in_dim}, got {X.shape[1]}"
            )

        self._input = X
        self._z = X @ self.W + self.b

        if self.activation is None:
            return self._z

        return self.activation.forward(self._z)

    def backward(self, grad_output):
        """
        Backward propagation through the layer.

        Parameters
        ----------
        grad_output : gradient from next layer

        Returns
        -------
        gradient with respect to the input of this layer
        """

        if self.activation is not None:
            grad_output = grad_output * self.activation.derivative(self._z)

        # gradients for parameters
        self.grad_W = self._input.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)

        # gradient to propagate to previous layer
        grad_input = grad_output @ self.W.T

        return grad_input
