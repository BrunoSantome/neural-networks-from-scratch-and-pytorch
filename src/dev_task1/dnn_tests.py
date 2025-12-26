import numpy as np
import matplotlib.pyplot as plt
import unittest
from deep_nn import NeuronalNetwork
from sklearn.datasets import make_moons
import numpy as np
import pandas as pd
import os
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dnn_functions import (
    sigmoid,
    relu,
    softmax,
    sigmoid_back_pass,
    relu_back_pass,
    softmax_back_pass,
    plot_loss,
    plot_accuracy,
)
from preprocessing import (
    load_and_preprocess_data_spacial_objects,
    # load_and_preprocess_fish_classification_binary,
)


class dnn_tests(unittest.TestCase):
    def test_init_param(self):
        # [input, hiddeen1, ..., hiddenN, output]
        # todo: important make a method to input data and transform that data into the first input layer
        nn_architecture1 = [2, 4, 4, 1]
        nn_architecture2 = [8, 4, 2, 1]
        NNTest = NeuronalNetwork(nn_architecture1)
        parameters = NNTest._init_param()

    def test_forward_pass(self):
        # Question what convention should we use, (n of examples, features) or (features, n of examples) for input?
        np.random.seed(42)
        num_values = 2
        X = np.random.randn(3, 2)
        # print(X)
        nn_architecture1 = [2, 4, 1]
        NNTest2 = NeuronalNetwork(nn_architecture1, seed=42)
        NNTest2._init_param()
        print(NNTest2.forward_pass(X).shape)

    def test_cost_CCE(self):
        np.random.seed(42)
        num_values = 2
        X = np.random.randn(num_values, 3)

        nn_architecture1 = [4, 4, 1]
        NNTest2 = NeuronalNetwork(nn_architecture1, seed=42)
        NNTest2._init_param()
        NNTest2.forward_pass(X)
        NNTest2._loss_calc_CCE()

    def tests_backward_pass_first_layer(self):
        np.random.seed(42)
        num_values = 3
        features = 2
        X = np.random.randn(num_values, features)
        y = np.random.randn(1, num_values).T
        nn_architecture1 = [features, 4, 1]
        NNTest2 = NeuronalNetwork(nn_architecture1, seed=42)
        NNTest2._init_param()
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
        NNTest2._init_param()
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
        NNTest2._init_param()
        losses = []
        for i in range(epoch):
            activation_last = NNTest2.forward_pass(X)
            loss = NNTest2._loss_calc_CCE(activation_last, y)
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
        NNTest2._init_param()
        losses = []
        for i in range(epoch):
            activation_last = NNTest2.forward_pass(X)
            loss = NNTest2._loss_calc_CCE(activation_last, y)
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
        NNTest2._init_param()
        losses = []
        for i in range(epoch):
            activation_last = NNTest2.forward_pass(X_clf)
            loss = NNTest2._loss_calc_bCE(activation_last, y_clf)
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
            nn_architecture1,
            seed=42,
            lambda_l1=0.001,
            lambda_l2=0.001,
            output_activation="softmax",
            learning_rate=0.01,
        )
        NNTest2._init_param()
        losses = []
        for i in range(epoch):
            activation_last = NNTest2.forward_pass(X_clf)
            loss = NNTest2._loss_calc_CCE(activation_last, y_onehot)
            losses.append(loss)
            NNTest2.backward_pass(activation_last, y_onehot)
            NNTest2.update_param()
        # Problems with softmax function. Check loss increases and becomes nan.
        print(losses[-1])
        plt.plot(losses)
        plt.title("Loss CCE")
        plt.show()

    def test_dataset_BCE_Regularisation(self, epoch):
        # This uses Binary cross entropy and sigmoid, It cannot use softmax.
        X_clf, y_clf = make_moons(n_samples=300, noise=0.15, random_state=1)
        y_clf = y_clf.reshape(-1, 1)
        nn_architecture1 = [X_clf.shape[1], 8, 1]
        NNTest2 = NeuronalNetwork(nn_architecture1, lambda_l2=0.001, seed=42)
        NNTest2._init_param()
        losses = []
        for i in range(epoch):
            activation_last = NNTest2.forward_pass(X_clf)
            loss = NNTest2._loss_calc_bCE(activation_last, y_clf)
            losses.append(loss)
            NNTest2.backward_pass(activation_last, y_clf)
            NNTest2.update_param()
        print(losses[-1])
        plt.plot(losses)
        plt.title("Loss without Dropout")
        plt.show()

    def test_dataset_CCE_momentum(self, epoch, momentum_beta):
        # This uses Binary cross entropy and sigmoid, It cannot use softmax.
        X_clf, y_clf = make_moons(n_samples=300, noise=0.15, random_state=1)
        y_clf = y_clf.reshape(-1, 1)
        y_onehot = np.eye(2)[y_clf.ravel()]  # transform output so it accepts CCE
        nn_architecture1 = [X_clf.shape[1], 8, 2]
        NNTest2 = NeuronalNetwork(
            nn_architecture1,
            seed=42,
            output_activation="softmax",
            learning_rate=0.01,
            momentum_beta=momentum_beta,
        )
        NNTest2._init_param()
        losses = []
        for i in range(epoch):
            activation_last = NNTest2.forward_pass(X_clf)
            loss = NNTest2._loss_calc_CCE(activation_last, y_onehot)
            losses.append(loss)
            NNTest2.backward_pass(activation_last, y_onehot)
            NNTest2.update_param()
        # Problems with softmax function. Check loss increases and becomes nan.
        print(losses[-1])
        plt.plot(losses)
        plt.title("Loss Moons with Momentum of 0.2")
        plt.show()

    def test_dataset_space_classification(self, epoch):
        X_train, y_train, X_test, y_test = load_and_preprocess_data_spacial_objects()
        nn_architecture1 = [X_train.shape[1], 16, 8, 3]
        y_train_onehot = np.eye(3)[y_train.to_numpy()]
        y_test_onehot = np.eye(3)[y_test.to_numpy()]
        NNtest = NeuronalNetwork(
            nn_architecture1,
            seed=42,
            hidden_activation="relu",
            output_activation="softmax",
            learning_rate=0.01,
            loss_function="CCE",
            dropout_rate=0.2,
            momentum_beta=0.2,
        )
        NNtest._init_param()
        losses = []
        for i in range(epoch + 1):
            activation_last = NNtest.forward_pass(X_train)
            loss = NNtest._loss_calc_CCE(activation_last, y_train_onehot)
            losses.append(loss)
            if i % 100 == 0:
                print(f"Epoch {i}, loss: {loss}")
            NNtest.backward_pass(activation_last, y_train_onehot)
            NNtest.update_param()
        # Test accuracy
        activation_test = NNtest.forward_pass(X_test)
        predictions = np.argmax(activation_test, axis=1)
        y_test_labels = np.argmax(y_test_onehot, axis=1)
        accuracy = np.mean(predictions == y_test_labels)
        print(f"Test accuracy: {accuracy}")
        # accuracy around 95% (possible overfitting)
        plot_loss(losses, "Loss with dropout on Stellar Object Classification Dataset")

    def test_fit_method_space_classification(self, epoch):
        X_train, y_train, X_test, y_test = load_and_preprocess_data_spacial_objects()
        nn_architecture1 = [X_train.shape[1], 16, 8, 3]
        y_train_onehot = np.eye(3)[y_train.to_numpy()]
        y_test_onehot = np.eye(3)[y_test.to_numpy()]
        NNtest = NeuronalNetwork(
            nn_architecture1,
            seed=42,
            hidden_activation="relu",
            output_activation="softmax",
            learning_rate=0.01,
            loss_function="CCE",
            dropout_rate=0.2,
            momentum_beta=0.9,
            epoch=epoch,
        )
        NNtest.fit(X_train, y_train_onehot, X_test, y_test_onehot)
        print(NNtest.train_accuracy[-1])
        print(NNtest.test_accuracy[-1])

        # Test accuracy
        # activation_test = NNtest.forward_pass(X_test)
        # predictions = np.argmax(activation_test, axis=1)
        # y_test_labels = np.argmax(y_test_onehot, axis=1)
        # accuracy = np.mean(predictions == y_test_labels)
        # print(f"Test accuracy: {accuracy}")
        # accuracy around 95% (possible overfitting)
        plt.plot(NNtest.train_accuracy)
        plt.title("Train accuracy Stellar object classification with momentum")
        plt.show()
        plot_loss(
            NNtest.losses, "Loss with dropout on Stellar Object Classification Dataset"
        )

    def test_fit_sgd_method_space_classification(self, epoch):
        X_train, y_train, X_test, y_test = load_and_preprocess_data_spacial_objects()
        nn_architecture1 = [X_train.shape[1], 16, 8, 3]
        y_train_onehot = np.eye(3)[y_train.to_numpy()]
        y_test_onehot = np.eye(3)[y_test.to_numpy()]
        NNtest = NeuronalNetwork(
            nn_architecture1,
            seed=42,
            hidden_activation="relu",
            output_activation="softmax",
            learning_rate=0.01,
            loss_function="CCE",
            dropout_rate=0.2,
            momentum_beta=0,
            epoch=epoch,
            optimizer="sgd",
        )
        NNtest.fit(X_train, y_train_onehot, X_test, y_test_onehot)

        # Test accuracy
        # activation_test = NNtest.forward_pass(X_test)
        # predictions = np.argmax(activation_test, axis=1)
        # y_test_labels = np.argmax(y_test_onehot, axis=1)
        # accuracy = np.mean(predictions == y_test_labels)
        # print(f"Test accuracy: {accuracy}")
        # accuracy around 95% (possible overfitting)
        plt.plot(NNtest.train_accuracy)
        plt.title("Train accuracy Stellar object classification with momentum")
        plt.show()
        plot_loss(
            NNtest.losses, "Loss with dropout on Stellar Object Classification Dataset"
        )

    def test_fit_mini_batches_method_space_classification(self, epoch):
        # It performs so much better and so much quicker with mini batches.
        X_train, y_train, X_test, y_test = load_and_preprocess_data_spacial_objects()
        nn_architecture1 = [X_train.shape[1], 16, 4, 3]
        NNtest = NeuronalNetwork(
            nn_architecture1,
            seed=42,
            hidden_activation="relu",
            output_activation="softmax",
            optimizer1="gd",
            learning_rate=0.01,
            loss_function="CCE",
            epoch=epoch,
            mini_batch=False,
            mini_batch_size=64,
        )

        # 0.74655
        # 0.74175
        # 0.7910804731334554
        y_train_onehot = np.eye(3)[y_train.to_numpy()]
        y_test_onehot = np.eye(3)[y_test.to_numpy()]
        NNtest.fit(X_train, y_train_onehot, X_test, y_test_onehot)
        print(NNtest.train_accuracy[-1])
        print(NNtest.test_accuracy[-1])
        print(NNtest.losses[-1])
        # Insane improvement with mini batches on train and test set !
        # Without Mini batches: 0.7751625 / 0.76915 / 0.639258578845312
        # With mini batches: 0.90625 / 0.85285 / 0.004088191320124638
        # Test accuracy
        # print(f"Test accuracy: {accuracy}")
        # accuracy around 95% (possible overfitting)
        # plt.plot(NNtest.train_accuracy)
        # plt.title("Train accuracy Stellar object classification with momentum")
        # plt.show()
        # plot_loss(
        #     NNtest.losses, "Loss with dropout on Stellar Object Classification Dataset"
        # )

    def tuning_hyperparameters(
        self,
        description,
        architecture,
        epoch=100,
        learning_rate=0.01,
        seed=42,
        hidden_activation="relu",
        output_activation="sigmoid",
        dropout_rate=0,
        lambda_l1=0.0,
        lambda_l2=0.0,
        loss_function="CCE",
        optimizer1="gd",
        beta1=0.9,
        mini_batch=False,
        mini_batch_size=64,
    ):
        X_train, y_train, X_test, y_test = load_and_preprocess_data_spacial_objects()
        hidden_layers_units = []
        architecture = [16, 8]  # Example architecture
        hidden_layers_units.append(X_train.shape[1])
        hidden_layers_units.extend(architecture)
        hidden_layers_units.append(y_test.unique().shape[0])
        # hidden_layers_units = [X_train.shape[1], 8, 3]

        NNtest = NeuronalNetwork(
            hidden_layers_units,
            seed=seed,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            optimizer1=optimizer1,
            learning_rate=learning_rate,
            loss_function=loss_function,
            dropout_rate=dropout_rate,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            epoch=epoch,
            mini_batch=mini_batch,
            mini_batch_size=mini_batch_size,
        )

        y_train_onehot = np.eye(3)[y_train.to_numpy()]
        y_test_onehot = np.eye(3)[y_test.to_numpy()]
        time_start = time.time()
        NNtest.fit(X_train, y_train_onehot, X_test, y_test_onehot)
        time_end = time.time()
        elapsed_time = time_end - time_start
        y_train_pred, y_test_pred, y_train_true, y_test_true = (
            NNtest.create_confusion_matrixes(
                X_train, X_test, y_train_onehot, y_test_onehot
            )
        )

        # cm_train = confusion_matrix(y_train_true, y_train_pred)
        # cm_test = confusion_matrix(y_test_true, y_test_pred)
        # print(cm_train)
        # disp_train = ConfusionMatrixDisplay(
        #     confusion_matrix=cm_train, display_labels=[0, 1, 2]
        # )
        # disp_train.plot(cmap="Blues")
        # plt.title("Train Confusion Matrix")
        # plt.show()

        # print(cm_test)
        # disp_test = ConfusionMatrixDisplay(
        #     confusion_matrix=cm_test, display_labels=[0, 1, 2]
        # )
        # disp_test.plot(cmap="Blues")
        # plt.title("Test Confusion Matrix")
        # plt.show()

        # plot_loss(NNtest.losses, "Losses over epochs")
        # plot_accuracy(NNtest.train_accuracy, "Training accuracy over epochs")
        # plot_accuracy(NNtest.test_accuracy, "Testing accuracy over epochs")
        print(NNtest.train_accuracy[-1])
        print(NNtest.test_accuracy[-1])
        print(NNtest.losses[-1])

        data = {
            "Description": [description],
            "Architecture": [str(hidden_layers_units)],
            "Epoch": [epoch],
            "Learning Rate": [learning_rate],
            "seed": [seed],
            "hidden a": [hidden_activation],
            "Output a": [output_activation],
            "Dropout rate": [dropout_rate],
            "Lambda L1": [lambda_l1],
            "Lambda L2": [lambda_l2],
            "Loss function": [loss_function],
            "Optimizer": [optimizer1],
            "beta1": [beta1],
            "mini batch": [1 if mini_batch else 0],
            "mini batch size": [mini_batch_size],
            "Train accuracy": [NNtest.train_accuracy[-1]],
            "Test accuracy": [NNtest.test_accuracy[-1]],
            "Final Loss": [NNtest.losses[-1]],
            "Processing Time (s)": [elapsed_time],
        }
        df_new = pd.DataFrame(data)
        if os.path.exists(EXCEL_FILE):
            df_existing = pd.read_excel(EXCEL_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(EXCEL_FILE, index=False)
        else:
            df_new.to_excel(EXCEL_FILE, index=False)


if __name__ == "__main__":
    EXCEL_FILE = "hyperparameter_tuning_results.xlsx"
    Tests = dnn_tests()
    # Tests.test_init_param()
    # Tests.test_forward_pass()z
    # Tests.tests_backward_pass_first_layer()
    # Tests.tests_backward_pass_hidden_layers()
    # Tests.tests_network_with_epoch(10000)
    # Tests.tests_network_with_epoch_dropout(100, 0.2)
    # Tests.test_dataset(4000, 0.4)
    # Tests.test_dataset_CCE(4000, 0.4)
    # Tests.test_dataset_BCE_Regularisation(3000)
    # Tests.test_dataset_space_classification(3000)
    # Tests.test_dataset_CCE_momentum(2000, 0.1)
    # Tests.test_fit_method_space_classification(2000)
    # Tests.test_fit_sgd_method_space_classification(100)
    # Tests.test_fit_mini_batches_method_space_classification(2000)
    NN_hidden_architecture = [64, 128, 64, 32]
    Tests.tuning_hyperparameters(
        description="Test4 architecture 1 layers, lr=0.01, mini-batch, relu hidden, gradient-descent without mini batch",
        architecture=NN_hidden_architecture,
        epoch=100,
        learning_rate=0.001,
        seed=42,
        hidden_activation="relu",
        output_activation="softmax",
        dropout_rate=0.2,
        lambda_l1=0.0,
        lambda_l2=0.0,
        loss_function="CCE",
        optimizer1="momentum",
        beta1=0.9,
        mini_batch=True,
        mini_batch_size=64,
    )
