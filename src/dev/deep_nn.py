import numpy as np
from dnn_functions import (
    sigmoid,
    relu,
    softmax,
    sigmoid_back_pass,
    relu_back_pass,
    softmax_back_pass,
)


class NeuronalNetwork:
    """Fully parametrizable neural network class"""

    def __init__(
        self,
        hidden_layers_units,
        epoch=20,
        learning_rate=0.1,
        seed=42,
        hidden_activation="relu",
        output_activation="sigmoid",
        dropout_rate=0,
        lambda_l1=0.0,
        lambda_l2=0.0,
        batch_size=1,
        loss_function="CCE",
        optimizer1="gd",
        beta1=0.9,
        mini_batch=False,
        mini_batch_size=64,
    ):
        self.num_layers_units = hidden_layers_units
        self.num_layers = len(hidden_layers_units) - 1
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.seed = seed
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.optimizer1 = optimizer1
        self.mini_batch = mini_batch
        self.mini_batch_size = mini_batch_size
        self.beta1 = beta1
        self.param = {}
        self.gradients = {}
        self.velocities = {}
        self.storage_layers = []
        self.dropout_masks = []
        self.losses = []
        self.train_accuracy = []
        self.test_accuracy = []
        np.random.seed(self.seed)

        # self.init_param() only called when the data is fitted into the model (to fit dimension of first input layer)

    def init_param(self):
        """method to initialisize randomly the weights and bias of the network based on the given architecture

        Returns:
            param: dictionary that contains all the weights and bias initialized randomely
        """
        self.losses = []
        for l in range(1, len(self.num_layers_units)):
            # Dimensions of Wl: (n[l-1], n[l]) ==> it is the same for dWl (for backward pass)
            self.param[f"W{l}"] = np.random.rand(
                self.num_layers_units[l - 1], self.num_layers_units[l]
            )
            # Dimensions of bl: ( 1, n[l],) ==> it is the same for dbl (for backward pass)
            self.param[f"b{l}"] = np.zeros((1, self.num_layers_units[l]))

            # beta1 is different than 0, it initialise the velocities used to upadte the gradients after the backward pass.
            if self.optimizer1 == "momentum":
                self.velocities[f"dW{l}"] = np.zeros(self.param[f"W{l}"].shape)
                self.velocities[f"db{l}"] = np.zeros(self.param[f"b{l}"].shape)

    def gen_random_mini_batches(self, X, y):
        # We get the amount of examples
        m = X.shape[0]
        np.random.seed(self.seed + 1)
        perm = np.random.permutation(m)

        X_shuffled = X[perm]
        y_shuffled = y[perm]
        mini_batches = []
        for i in range(0, m, self.mini_batch_size):
            mini_batches.append(
                (
                    X_shuffled[i : i + self.mini_batch_size],
                    y_shuffled[i : i + self.mini_batch_size],
                )
            )
        return mini_batches

    def dropout(self, activation):
        mask = (np.random.rand(*activation.shape) > self.dropout_rate).astype(float) / (
            1 - self.dropout_rate
        )
        activation_dropout = activation * mask  # Scaling down.
        self.dropout_masks.append(mask)
        return activation_dropout

    def inverted_dropout(self, da, layer):
        da *= self.dropout_masks[layer - 1]
        return da

    def forward_pass_hidden_single(self, activation_last, W, b):
        Z = np.dot(activation_last, W) + b
        if self.hidden_activation == "relu":
            activation_current, Z = relu(Z)
        if self.hidden_activation == "sigmoid":  # We could implement tanh
            # It might not be useful to return the Z value too.
            activation_current, Z = sigmoid(Z)
        if self.dropout_rate:
            return self.dropout(activation_current), (Z, W, b, activation_last)
        return activation_current, (Z, W, b, activation_last)

    def forward_pass_output_layer(self, activation_last, W, b):
        Z = np.dot(activation_last, W) + b
        if self.output_activation == "softmax":
            activation_current, Z = softmax(Z)
        if self.output_activation == "sigmoid":
            activation_current, Z = sigmoid(Z)
        return activation_current, (Z, W, b, activation_last)

    def forward_pass(self, X):
        # input: a[l-1]
        # z[l] = W[l]*A[l-1] + b[l]
        # a[l] = g[l](z[l])
        # output: a[l], storage: (z[l]: (W[l]*a[l-1]+b[l]), W[l], b[l], a[l-1])
        self.storage_layers = []
        self.dropout_masks = []
        X = np.atleast_2d(X)
        activation_current = X
        # -1 because the first layer does not count.
        # Hidden layers
        for l in range(1, self.num_layers):
            activation_last = activation_current
            activation_current, storage = self.forward_pass_hidden_single(
                activation_last, self.param[f"W{l}"], self.param[f"b{l}"]
            )
            self.storage_layers.append(storage)

        # Output layer
        activation_output, storage = self.forward_pass_output_layer(
            activation_current,
            self.param[f"W{self.num_layers}"],
            self.param[f"b{self.num_layers}"],
        )
        self.storage_layers.append(storage)

        return activation_output

    def loss_calc_MSE(self, activation_last, y):
        # Loss(y', y) Mean Squarred Error
        # By calculating the derivative, and vectorize it,
        # it is the input for the start of the backward pass. so da[-1]
        loss = np.mean((y - activation_last) ** 2)
        return loss

    def apply_regularisation(self, loss):
        l1_loss = 0
        l2_loss = 0
        for l in range(1, len(self.num_layers_units)):
            if self.lambda_l1:
                l1_loss += np.sum(np.abs(self.param[f"W{l}"]))
            if self.lambda_l2:
                l2_loss += np.sum((self.param[f"W{l}"]) ** 2)
        l1_loss *= self.lambda_l1
        l2_loss *= self.lambda_l2
        loss += l1_loss + l2_loss
        return loss

    def loss_calc_BCE(self, activation_last, y):
        # Binary Cross Entropy with L1 and L2 regularisation
        loss = -np.mean(
            (y * np.log(activation_last + 1e-8))
            + ((1 - y) * np.log(1 - activation_last + 1e-8))
        )

        if self.lambda_l1 or self.lambda_l2:
            return self.apply_regularisation(loss)
        return loss

    def loss_calc_CCE(self, activation_last, y):
        # Categorical Cross Entropy
        loss = -np.mean(np.sum(y * np.log(activation_last + 1e-8), axis=1))
        if self.lambda_l1 or self.lambda_l2:
            return self.apply_regularisation(loss)
        return loss

    def backward_pass_calc(self, dZ, layer):
        dZ = np.atleast_2d(dZ)
        storage = self.storage_layers[layer]
        _, W, _, activation_last = storage
        m = activation_last.shape[0]
        dW = 1 / m * activation_last.T.dot(dZ)  # a1.T.dot(a2_delta)
        # L1 and L2 Regularisation
        if self.lambda_l1:
            dW += self.lambda_l1 * np.sign(W)
        if self.lambda_l2:
            dW += self.lambda_l2 * 2 * W
        db = 1 / m * np.sum(dZ, axis=0, keepdims=True)  # Check this
        da = dZ.dot(W.T)
        if self.dropout_rate and layer != 0:  # Not the input layer.
            da = self.inverted_dropout(da, layer)
        return da, dW, db

    def backward_pass_hidden_single(self, da, layer):
        storage = self.storage_layers[layer]
        Z, _, _, _ = storage
        if self.hidden_activation == "relu":
            dZ = relu_back_pass(da, Z)
        if self.hidden_activation == "sigmoid":
            dZ = sigmoid_back_pass(da, Z)
        return self.backward_pass_calc(dZ, layer)

    def backward_pass(self, activation_first, y):
        # input a[l-1], y, storage: (z[l], W[l], b[l], a[l-1])).
        # da[l] = a[l-1] - Y
        # dz[l] = da[l] * g[l]'(z[l])
        # dw[l] = 1/m * dz[l]. a[l-1].T
        # db[l] = 1/m * dz[l]
        # da[l] = W[l].T* dz[l]
        # output: da[l-1] dw[l], db[l]
        # Backward propagation init.
        dZ = activation_first - y
        # simplification of the product of derivative of the loss with respect to the output activation function
        # This works for BCE or CCE and sigmoid or softmax as output activation function.
        (
            self.gradients[f"da{self.num_layers - 1}"],
            self.gradients[f"dW{self.num_layers}"],
            self.gradients[f"db{self.num_layers}"],
        ) = self.backward_pass_calc(dZ, self.num_layers - 1)

        for i in range(self.num_layers - 2, -1, -1):
            da = self.gradients[f"da{i + 1}"]
            (
                self.gradients[f"da{i}"],
                self.gradients[f"dW{i + 1}"],
                self.gradients[f"db{i + 1}"],
            ) = self.backward_pass_hidden_single(da, i)

    def update_param_with_momentum(self, i):
        # Average of the gradients
        self.velocities[f"dW{i}"] = (
            self.beta1 * self.velocities[f"dW{i}"]
            + (1 - self.beta1) * self.gradients[f"dW{i}"]
        )
        self.velocities[f"db{i}"] = (
            self.beta1 * self.velocities[f"db{i}"]
            + (1 - self.beta1) * self.gradients[f"db{i}"]
        )

        self.param[f"W{i}"] -= self.learning_rate * self.velocities[f"dW{i}"]
        self.param[f"b{i}"] -= self.learning_rate * self.velocities[f"db{i}"]

    def update_param_gd(self, i):
        self.param[f"W{i}"] -= self.learning_rate * self.gradients[f"dW{i}"]
        self.param[f"b{i}"] -= self.learning_rate * self.gradients[f"db{i}"]

    def update_param(self):
        # input: dw[l], db[l], parameters: ( W[l], b[l])
        # W[l] -= learning_rate*dW[l]
        # b[l] -= learning_rate*db[l]
        # output parameters: W[l], b[l] (updated)
        for i in range(1, self.num_layers + 1):
            # If there is a momentum beta set different that 0 it updates the parameters with it.
            if self.optimizer1 == "momentum":
                self.update_param_with_momentum(i)
            if self.optimizer1 == "gd":
                self.update_param_gd(i)

    def fit_sgd_optimizer(self, X_train, y_train, X_test, y_test):
        # todo: very very slow, ask teacher. Discarting this optimizer for the moment (not sure if it even works well)
        for i in range(self.epoch):
            total_loss = 0
            for j in range(0, X_train.shape[0]):
                X = X_train[j : j + 1]
                y = y_train[j : j + 1]
                activation_last = self.forward_pass(X)
                if self.loss_function == "BCE":
                    total_loss += self.loss_calc_BCE(activation_last, y)
                if self.loss_function == "CCE":
                    total_loss += self.loss_calc_CCE(activation_last, y)
                self.backward_pass(activation_last, y)
                self.update_param()
            total_loss_avg = total_loss / X_train.shape[0]
            self.losses.append(total_loss_avg)
            if not i % 100:
                # checking the accuracy in the train and test set every 100 epochs.
                train_acc = self.eval_accuracy(y_train, self.forward_pass(X_train))
                test_acc = self.eval_accuracy(y_test, self.forward_pass(X_test))
                self.train_accuracy.append(train_acc)
                self.test_accuracy.append(test_acc)

    def fit_with_mini_batch(self, X_train, y_train, X_test, y_test):
        for i in range(self.epoch):
            minibatches = self.gen_random_mini_batches(X_train, y_train)
            total_loss = 0
            for m in minibatches:
                (X_train, y_train) = m
                activation_last = self.forward_pass(X_train)
                if self.loss_function == "BCE":
                    total_loss += self.loss_calc_BCE(activation_last, y_train)
                if self.loss_function == "CCE":
                    total_loss += self.loss_calc_CCE(activation_last, y_train)
                self.backward_pass(activation_last, y_train)
                self.update_param()
            total_loss_avg = total_loss / X_train.shape[0]
            self.losses.append(total_loss_avg)
            if not i % 100:
                # checking the accuracy in the train and test set every 100 epochs.
                train_acc = self.eval_accuracy(y_train, activation_last)
                test_acc = self.eval_accuracy(y_test, self.forward_pass(X_test))
                self.train_accuracy.append(train_acc)
                self.test_accuracy.append(test_acc)

    def fit_without_mini_batch(self, X_train, y_train, X_test, y_test):
        for i in range(self.epoch):
            activation_last = self.forward_pass(X_train)
            if self.loss_function == "BCE":
                loss = self.loss_calc_BCE(activation_last, y_train)
            if self.loss_function == "CCE":
                loss = self.loss_calc_CCE(activation_last, y_train)
            self.losses.append(loss)
            self.backward_pass(activation_last, y_train)
            self.update_param()
            if not i % 100:
                # checking the accuracy in the train and test set every 100 epochs.
                train_acc = self.eval_accuracy(y_train, activation_last)
                test_acc = self.eval_accuracy(y_test, self.forward_pass(X_test))
                self.train_accuracy.append(train_acc)
                self.test_accuracy.append(test_acc)

    def fit(self, X_train, y_train, X_test, y_test):
        # Init Data method Missing
        # x and y should be preprocessed, y one-hot encoded.
        # self.num_layers_units.insert(0, X_train.shape[1])
        self.init_param()
        # if self.optimizer == "sgd":
        #     self.fit_sgd_optimizer(X_train, y_train, X_test, y_test)
        if not self.mini_batch:
            self.fit_without_mini_batch(X_train, y_train, X_test, y_test)
        if self.mini_batch:
            self.fit_with_mini_batch(X_train, y_train, X_test, y_test)

    def predict(self, X_test):
        activation_last = self.forward_pass(X_test)
        return np.argmax(activation_last, axis=1)

    def eval_accuracy(self, y_test, y_pred):
        # It is needed to convert the probabilities of the inputs into actual class labels
        # T
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred == y_test_labels)
        return accuracy
