import numpy as np
from .neural_layer import NeuralLayer

from .optimizers import get_optimizer
from .objective_functions import LOSS_GRAD, LOSS_FN



class NeuralNetwork:
    """
    Feedforward neural network supporting training and evaluation.
    """

    def __init__(self, cli_args):

        self.args = cli_args

        self.input_dim = getattr(cli_args, "input_dim", 784)
        self.output_dim = getattr(cli_args, "output_dim", 10)

        self.layers = []
        self._build_network()

   

        self.optimizer = get_optimizer(
            cli_args.optimizer,
            cli_args.learning_rate,
            cli_args.weight_decay
        )

    def _build_network(self):
    
        hidden = getattr(self.args, "hidden_size", None)
        num_layers = getattr(self.args, "num_layers", 1)
    
        activation = getattr(self.args, "activation", "relu")
        weight_init = getattr(self.args, "weight_init", "xavier")
    
        num_hidden = num_layers - 1
    
        if hidden is None:
            hidden = [64] * num_hidden
    
        if len(hidden) < num_hidden:
            hidden = hidden + [hidden[-1]] * (num_hidden - len(hidden))
    
        if len(hidden) > num_hidden:
            hidden = hidden[:num_hidden]
    
        dims = [getattr(self.args, "input_dim", 784)] + hidden + [getattr(self.args, "output_dim", 10)]    
        for i in range(len(dims) - 1):
    
            act = activation if i < len(dims) - 2 else None
    
            layer = NeuralLayer(
                dims[i],
                dims[i + 1],
                activation=act,
                weight_init=weight_init
            )
    
            self.layers.append(layer)

    def forward(self, X):
        """Forward propagation."""

        if not self.layers:
            self.input_dim = X.shape[1]

        out = X

        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, y_true, logits):

        delta = LOSS_GRAD[self.args.loss](logits, y_true)

        for layer in reversed(self.layers):
            delta = layer.backward(delta)

        loss = LOSS_FN[self.args.loss](logits, y_true)

        return loss

    def update_weights(self):
        """Optimizer step."""
        self.optimizer.step(self.layers)

    def compute_accuracy(self, logits, y):

        preds = np.argmax(logits, axis=1)
        labels = y

        return np.mean(preds == labels)

    def train(self, X_train, y_train, X_val, y_val, epochs=1, batch_size=32):

        n = X_train.shape[0]
        best_val_acc = -np.inf
        best_weights = None

        for epoch in range(epochs):

            idx = np.random.permutation(n)
            X_train = X_train[idx]
            y_train = y_train[idx]

            epoch_loss = 0.0

            for start in range(0, n, batch_size):

                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]

                logits = self.forward(X_batch)

                loss = self.backward(y_batch, logits)

                self.update_weights()

                epoch_loss += loss * len(X_batch)

            epoch_loss /= n

            train_logits = self.forward(X_train)
            train_acc = self.compute_accuracy(train_logits, y_train)

            val_logits = self.forward(X_val)
            val_acc = self.compute_accuracy(val_logits, y_val)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = self.get_weights()

            print(
                f"Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f} "
                f"| Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )

        if best_weights is not None:
            self.set_weights(best_weights)

    def evaluate(self, X, y):

        logits = self.forward(X)

        preds = np.argmax(logits, axis=1)

        acc = np.mean(preds == y)

        return acc

    def get_weights(self):

        weights = {}

        for i, layer in enumerate(self.layers):
            weights[f"W{i}"] = layer.W.copy()
            weights[f"b{i}"] = layer.b.copy()

        return weights

    def set_weights(self, weights):

        for i, layer in enumerate(self.layers):

            if f"W{i}" in weights:
                layer.W = weights[f"W{i}"].copy()

            if f"b{i}" in weights:
                layer.b = weights[f"b{i}"].copy()


