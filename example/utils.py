import numpy as np
import os
import urllib.request


def load_mnist():
    path = "mnist.npz"
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with np.load(path) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

    return (x_train, y_train), (x_test, y_test)
