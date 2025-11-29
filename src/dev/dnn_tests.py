import numpy as np
import matplotlib.pyplot as plt
import unittest
from deep_nn import NeuronalNetwork
from sklearn.datasets import make_moons
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
        NNTest2.forward_pass(X)
        NNTest2.loss_calc_CCE()

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
        NNTest2.update_param()

    def tests_network_with_epoch(self, epoch):
        np.random.seed(42)
        num_values = 100
        features = 10
        X = np.random.randn(num_values, features)
        y = np.random.randint(0, 2, size=(num_values, 1))
        nn_architecture1 = [features, 4, 1]
        NNTest2 = NeuronalNetwork(nn_architecture1, seed=42)
        NNTest2.init_param()
        losses = []
        for i in range(epoch):
            activation_last = NNTest2.forward_pass(X)
            loss = NNTest2.loss_calc_CCE(activation_last, y)
            losses.append(loss)
            NNTest2.backward_pass(activation_last, y)
            NNTest2.update_param()
            print(loss)

    def tests_network_with_epoch_dropout(self, epoch, dropout_rate):
        np.random.seed(42)
        num_values = 10
        features = 2
        X = np.random.randn(num_values, features)
        # y = np.random.randn(1, num_values).T  # Still dont understand why the transpose.
        y = np.random.randint(0, 2, size=(num_values, 1))
        print
        nn_architecture1 = [features, 4, 1]
        NNTest2 = NeuronalNetwork(nn_architecture1, seed=42, dropout_rate=dropout_rate)
        NNTest2.init_param()
        losses = []
        for i in range(epoch):
            activation_last = NNTest2.forward_pass(X)
            loss = NNTest2.loss_calc_CCE(activation_last, y)
            losses.append(loss)
            NNTest2.backward_pass(activation_last, y)
            NNTest2.update_param()
            print(loss)

    def test_dataset_BCE(self, epoch, dropout_rate):
        # This uses Binary cross entropy and sigmoid, It cannot use softmax.
        X_clf, y_clf = make_moons(n_samples=300, noise=0.15, random_state=1)
        y_clf = y_clf.reshape(-1, 1)
        nn_architecture1 = [X_clf.shape[1], 8, 1]
        NNTest2 = NeuronalNetwork(nn_architecture1, seed=42)
        NNTest2.init_param()
        losses = []
        for i in range(epoch):
            activation_last = NNTest2.forward_pass(X_clf)
            loss = NNTest2.loss_calc_BCE(activation_last, y_clf)
            losses.append(loss)
            NNTest2.backward_pass(activation_last, y_clf)
            NNTest2.update_param()

        plt.plot(losses)
        plt.title("Loss without Dropout")
        plt.show()

    def test_dataset_CCE(self, epoch, dropout_rate):
        # This uses Binary cross entropy and sigmoid, It cannot use softmax.
        X_clf, y_clf = make_moons(n_samples=300, noise=0.15, random_state=1)
        y_clf = y_clf.reshape(-1, 1)
        y_onehot = np.eye(2)[y_clf.ravel()]  # transform output so it accepts CCE
        nn_architecture1 = [X_clf.shape[1], 8, 2]
        NNTest2 = NeuronalNetwork(
            nn_architecture1, seed=42, output_activation="softmax", learning_rate=0.01
        )
        NNTest2.init_param()
        losses = []
        for i in range(epoch):
            activation_last = NNTest2.forward_pass(X_clf)
            loss = NNTest2.loss_calc_CCE(activation_last, y_onehot)
            losses.append(loss)
            NNTest2.backward_pass(activation_last, y_onehot)
            NNTest2.update_param()
        print(losses)
        # Problems with softmax function. Check loss increases and becomes nan.
        plt.plot(losses)
        plt.title("Loss CCE")
        plt.show()

    def test_dataset_BCE_Regularisation(self, epoch):
        # This uses Binary cross entropy and sigmoid, It cannot use softmax.
        X_clf, y_clf = make_moons(n_samples=300, noise=0.15, random_state=1)
        y_clf = y_clf.reshape(-1, 1)
        nn_architecture1 = [X_clf.shape[1], 8, 1]
        NNTest2 = NeuronalNetwork(
            nn_architecture1, lambda_l1=0.0001, lambda_l2=0.0001, seed=42
        )
        NNTest2.init_param()
        losses = []
        for i in range(epoch):
            activation_last = NNTest2.forward_pass(X_clf)
            loss = NNTest2.loss_calc_BCE(activation_last, y_clf)
            losses.append(loss)
            NNTest2.backward_pass(activation_last, y_clf)
            NNTest2.update_param()
        print(losses[-1])
        plt.plot(losses)
        plt.title("Loss without Dropout")
        plt.show()


if __name__ == "__main__":
    Tests = dnn_tests()
    # Tests.test_init_param()
    # Tests.test_forward_pass()z
    # Tests.tests_backward_pass_first_layer()
    # Tests.tests_backward_pass_hidden_layers()
    # Tests.tests_network_with_epoch(10000)
    # Tests.tests_network_with_epoch_dropout(100, 0.2)
    # Tests.test_dataset(4000, 0.4)
    Tests.test_dataset_CCE(4000, 0.4)
    # Tests.test_dataset_BCE_Regularisation(3000)
