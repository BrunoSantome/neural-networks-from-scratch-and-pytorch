import numpy as np
import unittest
from deep_nn import NeuronalNetwork


class dnn_tests(unittest.TestCase):
    def test_init_param(self):
        # [input, hiddeen1, ..., hiddenN, output]
        # todo: important make a method to input data and transform that data into the first input layer
        nn_architecture1 = [2, 4, 1]
        nn_architecture2 = [8, 4, 2, 1]
        NNTest = NeuronalNetwork(nn_architecture1)
        parameters = NNTest.init_param()
        print(parameters)


if __name__ == "__main__":
    Tests = dnn_tests()
    Tests.test_init_param()
