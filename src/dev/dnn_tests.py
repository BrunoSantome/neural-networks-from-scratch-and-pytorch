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
        np.random.seed(42)
        num_values = 2
        X = np.random.randn(num_values, 3)
        # print(X)
        nn_architecture1 = [num_values, 4, 4, 5, 1]
        NNTest2 = NeuronalNetwork(nn_architecture1, seed=42)
        NNTest2.init_param()
        print(NNTest2.forward_pass(X))


if __name__ == "__main__":
    Tests = dnn_tests()
    # Tests.test_init_param()
    Tests.test_forward_pass()
