import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


def one_hot(y):

    num_classes = 10

    y_one = np.zeros((len(y), num_classes), dtype=np.float32)
    y_one[np.arange(len(y)), y] = 1.0

    return y_one


def load_data(name,val_split=0.1):

    if name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    else:
        raise ValueError("Dataset must be mnist or fashion_mnist")

    # Flatten images
    X_train = X_train.reshape(-1, 784).astype(np.float32)
    X_test = X_test.reshape(-1, 784).astype(np.float32)

    # Normalize
    X_train /= 255.0
    X_test /= 255.0

    # # One-hot labels
    # y_train = one_hot(y_train)
    # y_test = one_hot(y_test)

    X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=val_split, random_state=42 )

    return X_train,X_val, X_test, y_train,y_val, y_test