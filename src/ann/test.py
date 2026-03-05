import numpy as np
from neural_layer import NeuralLayer

layer = NeuralLayer(4,3,activation="relu")

X = np.random.randn(5,4)

A = layer.forward(X)

dA = np.random.randn(5,3)

dX = layer.backward(dA)

print("Output shape:",A.shape)
print("dX shape:",dX.shape)
print("grad_W:",layer.grad_W.shape)
print("grad_b:",layer.grad_b.shape)