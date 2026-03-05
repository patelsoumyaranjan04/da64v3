import numpy as np

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(logits, y_true):

    probs = softmax(logits)
    n = y_true.shape[0]

    log_p = -np.log(probs[np.arange(n), y_true] + 1e-9)

    return np.mean(log_p)


def cross_entropy_grad(logits, y_true):

    probs = softmax(logits)
    n = y_true.shape[0]

    probs[np.arange(n), y_true] -= 1

    return probs / n


def mse(logits, y_true):

    probs = softmax(logits)

    n, c = probs.shape

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y_true] = 1

    return np.mean((probs - one_hot) ** 2)


def mse_grad(logits, y_true):

    probs = softmax(logits)

    n, c = probs.shape

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y_true] = 1

    diff = probs - one_hot

    grad = np.zeros_like(probs)

    for k in range(c):
        dsm = probs * (np.eye(c)[k] - probs[:, k:k+1])
        grad[:, k] = np.sum((2.0 / c) * diff * dsm, axis=1)

    return grad / n


LOSS_FN = {
    "cross_entropy": cross_entropy,
    "mse": mse,
}

LOSS_GRAD = {
    "cross_entropy": cross_entropy_grad,
    "mse": mse_grad,
}