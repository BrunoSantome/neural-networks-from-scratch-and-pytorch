import math
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(Z):
    """Sigmoid forward pass implementation with numpy

    Args:
    Z(np.ndarray) : ndarray

    Returns:
    A (np.ndarray):
        Result of sigmoid, same shape as Z.
    Z (np.ndarray):
       the input ndarray Z, usefull for backward prop
    """
    # carefull with overflow here might need to
    A = 1 / (1 + np.exp(-Z))
    return A, Z


def sigmoid_back_pass(da, Z):
    """Sigmoid backward pass implementation with numpy

    Args:
    da(np.ndarray) : post-activation gradient
    Z(np.ndarray) : ndarray

    Returns:
    dZ (np.ndarray):
        Gradient of the cost with respect to Z, same shape as Z.
    """
    A, _ = sigmoid(Z)
    dZ = da * A * (1 - A)
    return dZ


def relu(Z):
    """Relu forward pass implementation with numpy

    Args:
    Z(np.ndarray) : ndarray

    Returns:
    A (np.ndarray):
        Result of ReLu, same shape as Z.
    Z (np.ndarray):
       the input ndarray Z, usefull for backward prop
    """
    A = np.maximum(0, Z)
    return A, Z


def relu_back_pass(da, Z):
    """Relu backward pass implementation with numpy

    Args:
    da(np.ndarray) : post-activation gradient
    Z(np.ndarray) : ndarray

    Returns:
    dZ (np.ndarray):
        Gradient of the cost with respect to Z, same shape as Z.
    """
    dZ = np.array(da, copy=True)
    dZ[Z <= 0] = 0

    return dZ


def softmax(Z):
    """Softmax forward pass implementation with numpy

    Args:
        Z (np.ndarray): ndarray

    Returns:
    A (np.ndarray):
        Result of softmax, same shape as Z.
    Z (np.ndarray):
        the input ndarray Z, usefull for backward prop

    """

    """
    Numerical issues on tests, when calculating the loss it explodes "np.exp(Z)" this becomes very large or very small 
    When I then do the log() it produces a nan value, a very high or small number
    """
    Zmax = np.max(Z, axis=1, keepdims=True)
    A = np.exp(Z - Zmax) / np.sum(np.exp(Z - Zmax), axis=1, keepdims=True)

    return A, Z


def softmax_back_pass(da, Z):
    """Softmax backward pass implementation with numpy

    Args:
    da(np.ndarray) : post-activation gradient
    Z(np.ndarray) : ndarray

    Returns:
    dZ (np.ndarray):
        Gradient of the cost with respect to Z, same shape as Z.
    """
    A, _ = softmax(Z)
    dZ = A * (da - np.sum(da * A))
    return dZ


def plot_accuracy(accuracies, title):
    print(accuracies)
    plt.plot(accuracies)
    plt.title(title)
    plt.show()


def plot_loss(losses, title):
    print(losses)
    plt.plot(losses)
    plt.title(title)
    plt.show()


# def plot_confusion_matrix(cm):
