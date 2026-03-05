"""
Loss Functions Module
"""

import numpy as np


class CrossEntropyLoss:
    """Cross Entropy loss with softmax."""

    @staticmethod
    def _softmax(z):
        """Numerically stable softmax."""
        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z_shift)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, logits, y_true):

        probs = self._softmax(logits)

        n = y_true.shape[0]

        loss = -np.mean(
            np.log(probs[np.arange(n), y_true] + 1e-12)
        )

        return loss, probs

    def backward(self, probs, y_true):

        batch_size = y_true.shape[0]

        grad = probs.copy()

        grad[np.arange(batch_size), y_true] -= 1

        return grad / batch_size


class MeanSquaredError:
    """Mean Squared Error loss."""

    def forward(self, preds, y_true):
        """
        Compute MSE loss.
        """

        loss = np.mean((preds - y_true) ** 2)
        return loss, preds

    def backward(self, preds, y_true):
        """
        Gradient of MSE.
        """

        batch_size = preds.shape[0]
        return 2 * (preds - y_true) / batch_size


def get_loss(name: str):
    """Factory function for loss objects."""

    name = name.lower()

    losses = {
        "cross_entropy": CrossEntropyLoss,
        "mse": MeanSquaredError,
    }

    if name not in losses:
        raise ValueError(f"Unknown loss: {name}")

    return losses[name]()
