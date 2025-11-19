import numpy as np
import unittest
from deep_nn import NeuronalNetwork
from dnn_functions import (
    sigmoid,
    relu,
    softmax,
    sigmoid_back_pass,
    relu_back_pass,
    softmax_back_pass,
)


class dnn_tests(unittest.TestCase):
    def test_init_param(self):
        # [input, hiddeen1, ..., hiddenN, output]
        # todo: important make a method to input data and transform that data into the first input layer
        nn_architecture1 = [2, 4, 4, 1]
        nn_architecture2 = [8, 4, 2, 1]
        NNTest = NeuronalNetwork(nn_architecture1)
        parameters = NNTest.init_param()

    def test_forward_pass(self):
        # Question what convention should we use, (n of examples, features) or (features, n of examples) for input?
        np.random.seed(42)
        num_values = 2
        X = np.random.randn(3, 2)
        # print(X)
        nn_architecture1 = [2, 4, 1]
        NNTest2 = NeuronalNetwork(nn_architecture1, seed=42)
        NNTest2.init_param()
        print(NNTest2.forward_pass(X).shape)

    def test_cost_CCE(self):
        np.random.seed(42)
        num_values = 2
        X = np.random.randn(num_values, 3)

        nn_architecture1 = [4, 4, 1]
        NNTest2 = NeuronalNetwork(nn_architecture1, seed=42)
        NNTest2.init_param()

    def tests_backward_pass_first_layer(self):
        np.random.seed(42)
        num_values = 3
        features = 2
        X = np.random.randn(num_values, features)
        y = np.random.randn(1, num_values).T
        nn_architecture1 = [features, 4, 1]
        NNTest2 = NeuronalNetwork(nn_architecture1, seed=42)
        NNTest2.init_param()
        activation_last = NNTest2.forward_pass(X)
        NNTest2.backward_pass(activation_last, y)

    def tests_backward_pass_hidden_layers(self):
        np.random.seed(42)
        num_values = 3
        features = 2
        X = np.random.randn(num_values, features)
        y = np.random.randn(1, num_values).T  # Still dont understand why the transpose.
        nn_architecture1 = [features, 4, 1]
        NNTest2 = NeuronalNetwork(nn_architecture1, seed=42)
        NNTest2.init_param()
        activation_last = NNTest2.forward_pass(X)
        NNTest2.backward_pass(activation_last, y)


if __name__ == "__main__":
    Tests = dnn_tests()
    # Tests.test_init_param()
    # Tests.test_forward_pass()
    # Tests.tests_backward_pass_first_layer()
    Tests.tests_backward_pass_hidden_layers()
