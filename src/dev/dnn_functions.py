import math
import numpy as np


def sigmoid(Z):
    """Sigmoid forward pass implementation with numpy

    Args:
    Z(np.nparray) : nparray

    Returns:
    A (np.ndarray):
        Result of sigmoid, same shape as Z.
    Z (np.ndarray):
       the input nparray Z, usefull for backward prop
    """

    A = 1 / (1 + np.exp(-Z))
    return A, Z


def sigmoid_backward(da, storage): ...


def relu(Z):
    """Relu forward pass implementation with numpy

    Args:
    Z(np.nparray) : nparray

    Returns:
    A (np.ndarray):
        Result of ReLu, same shape as Z.
    Z (np.ndarray):
       the input nparray Z, usefull for backward prop
    """
    A = np.maximum(0, Z)
    return A, Z


def relu_back_pass(da, storage): ...


def softmax(Z):
    """Softmax forward pass implementation with numpy

    Args:
        Z (np.nparray): nparray

    Returns:
    A (np.ndarray):
        Result of softmax, same shape as Z.
    Z (np.ndarray):
        the input nparray Z, usefull for backward prop

    """
    A = np.exp(Z) / np.sum(np.exp(Z))
    return A, Z


def softmax_back_pass(da, storage): ...
