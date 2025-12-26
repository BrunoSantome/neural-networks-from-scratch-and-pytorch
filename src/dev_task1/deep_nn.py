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
    """
    Initializes a Neural Network (NN) model with customizable parameters.

    Args:

        hidden_layers_units (list): List containing the number of units in each layer of the network.
        epoch (int): Number of training epochs. Default is 100.
        learning_rate (float): Learning rate for weight updates. Default is 0.01.
        seed (int): Random seed for reproducibility. Default is 42.
        hidden_activation (str): Activation function for hidden layers ('relu' or 'sigmoid'). Default is 'relu'.
        output_activation (str): Activation function for output layer ('sigmoid' or 'softmax'). Default is 'sigmoid'.
        dropout_rate (float): Dropout rate for regularization (0 to 1). Default is 0. (no dropout).
        lambda_l1 (float): L1 regularization parameter. Default is 0.0.
        lambda_l2 (float): L2 regularization parameter. Default is 0.0.
        loss_function (str): Loss function to use ('BCE' or 'CCE'). Default is 'CCE'.
        optimizer1 (str): Optimization algorithm ('gd' for gradient descent, 'momentum' for momentum). Default is 'gd'.
        beta1 (float): Momentum beta parameter. Default is 0.9.
        mini_batch (bool): Whether to use mini-batch gradient descent. Default is False.
        mini_batch_size (int): Size of each mini-batch if mini_batch is True. Default is 64.

    """

    def __init__(
        self,
        hidden_layers_units,
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
        verbose=0,
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
        self.loss_function = loss_function
        self.optimizer1 = optimizer1
        self.mini_batch = mini_batch
        self.mini_batch_size = mini_batch_size
        self.beta1 = beta1
        self.verbose = verbose

        self.param = {}  # dictionary to store weights and bias of each layer
        self.gradients = {}  # dictionary to store gradients of weights and bias of each layer
        self.velocities = {}  # dictionary to store velocities for momentum optimizer
        self.storage_layers = []  # list to store forward pass values for backpropagation
        self.dropout_masks = []  # list to store dropout masks for each layer during forward pass
        self.losses = []  # list to store loss values during training
        self.train_accuracy = []  # list to store training accuracy during training
        self.test_accuracy = []  # list to store testing accuracy during training

        np.random.seed(self.seed)

        # self._init_param() only called when the data is fitted into the model (to fit dimension of first input layer)

    def _init_param(self):
        """function to initialisize randomly the weights and bias of the network based on the given architecture
        it iterates through the architecture array given
        """

        self.losses = []  # reset losses for new training
        for l in range(1, len(self.num_layers_units)):
            # Dimensions of Wl: (n[l-1], n[l]) ==> it is the same for dWl (for backward pass)
            self.param[f"W{l}"] = np.random.rand(
                self.num_layers_units[l - 1], self.num_layers_units[l]
            )
            # Dimensions of bl: ( 1, n[l],) ==> it is the same for dbl (for backward pass)
            self.param[f"b{l}"] = np.zeros((1, self.num_layers_units[l]))

            # if the optimizer1 selecteded is momentum
            # it initialise the velocities of the derivatives of the weights and bias,
            # used to upadte the gradients after the backward pass.
            if self.optimizer1 == "momentum":
                self.velocities[f"dW{l}"] = np.zeros(self.param[f"W{l}"].shape)
                self.velocities[f"db{l}"] = np.zeros(self.param[f"b{l}"].shape)

    def _gen_random_mini_batches(self, X, y):
        """
        Function to automatically generate random batches based on the mini_batch_size given (default to 64)
        It is the base to apply mini batch gradient descent.

        return: the array of mini batches.
        """
        # We get the amount of examples in the training set
        m = X.shape[0]
        # Each batch permutations of the examples are randomely set.
        np.random.seed(self.seed + 1)
        perm = np.random.permutation(m)
        # Array of mini_batches for each epoch
        mini_batches = []
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        # The step in the loop is customizable with the mini_batch_size variable
        for i in range(0, m, self.mini_batch_size):
            # we basically just take the batches from the whole X train data applying a random permutation
            mini_batches.append(
                (
                    X_shuffled[i : i + self.mini_batch_size],
                    y_shuffled[i : i + self.mini_batch_size],
                )
            )
        return mini_batches

    def _dropout(self, activation):
        """
        The dropout function generates for each layer a random mask that deactivates a certain amount of neurons
        based on the customizable dropout_rate variable

        return: the activation value after applying dropout.
        """
        mask = (np.random.rand(*activation.shape) > self.dropout_rate).astype(float) / (
            1 - self.dropout_rate  # scales up (inverted dropout)
        )

        activation_dropout = activation * mask
        self.dropout_masks.append(mask)
        # We store the mask of each layer into an array to later use it to apply inverted dropout.
        return activation_dropout

    def _inverted_dropout(self, da, layer):
        """
        Backward pass for inverted dropout.
        Applies the same dropout mask used during the forward pass
        to ensure that gradients are propagated only through the
        active neurons and are scaled consistently.

        return: the derivative of the activation function.
        """
        da *= self.dropout_masks[layer - 1]
        return da

    def _forward_pass_hidden_single(self, activation_last, W, b):
        """
        function that is in charge to calculate the hidden layers forward pass,
        by either applying the relu or sigmoid as activation function, customizable
        through the hidden_activation parameter.

        return:
        activation_current: the output of the activation calculated through the given activation function+
        (Z, W, b, activation_last): all the variables used during the iteration as cache
        """
        Z = np.dot(activation_last, W) + b
        # Each activation function returns the activation as well as the Z value received as input.
        if self.hidden_activation == "relu":
            activation_current, Z = relu(Z)
        if self.hidden_activation == "sigmoid":  # We could implement tanh
            activation_current, Z = sigmoid(Z)
        if self.dropout_rate:
            # to activate dropout the dropout_rate variable must be non-zero.
            # it retusn the value of the dropout output and
            # in order to use them later in the backward pass.
            return self._dropout(activation_current), (Z, W, b, activation_last)
        return activation_current, (Z, W, b, activation_last)

    def _forward_pass_output_layer(self, activation_last, W, b):
        """
        function that is in charge to calculate the output layer forward pass,
        by either applying the softmax or sigmoid as activation functions, customizable
        through the hidden_activation parameter.

        return:
        activation_current: the output of the activation calculated through the given activation function+
        (Z, W, b, activation_last): all the variables used during the iteration as cache
        """
        Z = np.dot(activation_last, W) + b
        if self.output_activation == "softmax":
            activation_current, Z = softmax(Z)
        if self.output_activation == "sigmoid":
            activation_current, Z = sigmoid(Z)
        return activation_current, (Z, W, b, activation_last)

    def _forward_pass(self, X):
        """
        Forward pass main function, in charge of handling the whole of the forward
        pass. It handles the hidden layers and the output layer.

        return:
        activation_output: the value calculated from the output activation function in the last layer of the network

        """
        # pseudo code used prior to implementation
        # input: a[l-1]
        # z[l] = W[l]*A[l-1] + b[l]
        # a[l] = g[l](z[l])
        # output: a[l], storage: (z[l]: (W[l]*a[l-1]+b[l]), W[l], b[l], a[l-1])

        # We use this array to store all the values of the forward pass to use on the backward pass.
        # Note that in each epoch, we clean the array.

        self.storage_layers = []

        # All dropout masks used during forward propagation, to use on the backward pass.
        self.dropout_masks = []

        X = np.atleast_2d(X)
        activation_current = X
        # Hidden layers forward pass calculations.
        for l in range(1, self.num_layers):
            activation_last = activation_current
            activation_current, storage = self._forward_pass_hidden_single(
                activation_last, self.param[f"W{l}"], self.param[f"b{l}"]
            )
            self.storage_layers.append(storage)

        # Output layer forward pass calculations.
        activation_output, storage = self._forward_pass_output_layer(
            activation_current,
            self.param[f"W{self.num_layers}"],
            self.param[f"b{self.num_layers}"],
        )
        self.storage_layers.append(storage)

        return activation_output

    def loss_calc_MSE(self, activation_last, y):
        """
        Mean Squarred Error Loss (y', y)
        By calculating the derivative, and vectorize it,
        it is the input for the start of the backward pass. so da[-1]
        Since the task is classification we will not be using it.
        """
        loss = np.mean((y - activation_last) ** 2)

        return loss

    def _apply_regularisation(self, loss):
        """
        L1 / L2 Regularisation

        The regulariser can be either L1, L2 or L1 and L2.
        Depending on whether the corresponding lambda_l1 and
        lamda_l2 have a value different than zero.

        it is only called once every network full iteration

        returns: the loss value after applying the regularizers.
        """
        l1_loss = 0
        l2_loss = 0
        for l in range(1, len(self.num_layers_units)):
            if self.lambda_l1:
                # Follows the formula of L1 regularisation
                l1_loss += np.sum(np.abs(self.param[f"W{l}"]))
            if self.lambda_l2:
                # Follows the formula of L2 regularisation
                l2_loss += np.sum((self.param[f"W{l}"]) ** 2)
        l1_loss *= self.lambda_l1
        l2_loss *= self.lambda_l2
        loss += l1_loss + l2_loss
        return loss

    def _loss_calc_BCE(self, activation_last, y):
        """
        Binary Cross Entropy Loss with L1 and L2 regularisation

        calculates the Binary cross entropy loss, with regularizers if applied.
        """
        # Follow the BCE math formula. Note the epsilon value so the logarithm is never 0 and therefore a nan value.
        epsilon = 1e-8
        loss = -np.mean(
            (y * np.log(activation_last + epsilon))
            + ((1 - y) * np.log(1 - activation_last + epsilon))
        )

        # if set, it applied the corresponding regularisations.
        if self.lambda_l1 or self.lambda_l2:
            return self._apply_regularisation(loss)
        return loss

    def _loss_calc_CCE(self, activation_last, y):
        """
        Categorical Cross Entropy Loss with L1 and L2 regularisation

        calculates the Categorical cross entropy loss, with regularizers if applied.
        """
        # Follow the BCE math formula. Note the epsilon value so the logarithm is never 0 and therefore a nan value.
        epsilon = 1e-8
        loss = -np.mean(np.sum(y * np.log(activation_last + epsilon), axis=1))
        # if set, it applied the corresponding regularisations.
        if self.lambda_l1 or self.lambda_l2:
            return self._apply_regularisation(loss)
        return loss

    def _backward_pass_calc(self, dZ, layer):
        """
        handles all calculations of a single hidden layer during the backward pass.
        If dropout has been activated it invoques de inverted dropout backward pass.
        If L1/L2 regularisation lambdas parameters are set, the weights are updated acordingly.
        return:
        da: the derivative of the activation function of the next layer to continue the backward pass.
        dw: the derivative of the weights
        db: the derivative of the bias
        """

        dZ = np.atleast_2d(dZ)  # this checks dimensions to avoid problems.

        storage = self.storage_layers[
            layer
        ]  # gets the corresponding storage values from the forward pass
        _, W, _, activation_last = storage
        m = activation_last.shape[0]
        dW = (
            1 / m * activation_last.T.dot(dZ)
        )  # a1.T.dot(a2_delta): pseudo code from a lab.

        # L1 and L2 Regularisation applied to backward pass, adjusting the value of the derivatives of the weights
        if self.lambda_l1:
            dW += self.lambda_l1 * np.sign(W)
        if self.lambda_l2:
            dW += self.lambda_l2 * 2 * W

        db = (
            1 / m * np.sum(dZ, axis=0, keepdims=True)
        )  # The keepdims is important to avoid problems of shapes
        da = dZ.dot(W.T)
        # Inverted dropout application during the backward pass.-
        if self.dropout_rate and layer != 0:  # Not the input layer.
            da = self._inverted_dropout(da, layer)

        return da, dW, db

    def _backward_pass_hidden_single(self, da, layer):
        """
        Backward pass function that calculates the pre-activation value Z depending on the
        hidden activation function selected.

        return: it returns the same values as the function above, da, dw, db.
        """
        storage = self.storage_layers[layer]
        Z, _, _, _ = storage
        if self.hidden_activation == "relu":
            dZ = relu_back_pass(da, Z)
        if self.hidden_activation == "sigmoid":
            dZ = sigmoid_back_pass(da, Z)

        # it calls the function explained above.
        return self._backward_pass_calc(dZ, layer)

    def _backward_pass(self, activation_first, y):
        """
        Backward pass main function. Handles all the backpropagation iterations.
        All values calculated are stored in the gradients dictionary for future use in the parameter update.

        # Input:
        activation_first: the value of the activation function of the output layer of the forward pass.
        y: True labels from the training set.

        """
        # Pseudo code prior to implementation of the network

        # input a[l-1], y, storage: (z[l], W[l], b[l], a[l-1])).
        # da[l] = a[l-1] - Y
        # dz[l] = da[l] * g[l]'(z[l])
        # dw[l] = 1/m * dz[l]. a[l-1].T
        # db[l] = 1/m * dz[l]
        # da[l] = W[l].T* dz[l]
        # output: da[l-1] dw[l], db[l]

        # Backward propagation init.
        # simplification of the product of derivative of the loss with respect to the output activation function
        # This works for BCE or CCE and sigmoid or softmax as output activation function. However this will not work for
        # the mean square error loss function. Since the task is a classification, there is no need to contemplate this scenario.
        dZ = activation_first - y

        # The first backward pass calculation handles the last layer of the network, corresponding to the output layer in the forward pass.
        # it is fed the value of dz and the layer number which returns the different pertinent derivatives.
        (
            self.gradients[f"da{self.num_layers - 1}"],
            self.gradients[f"dW{self.num_layers}"],
            self.gradients[f"db{self.num_layers}"],
        ) = self._backward_pass_calc(dZ, self.num_layers - 1)

        # Loop that iterates through the hidden layers of the network, calculating the respective derivatives.
        for i in range(self.num_layers - 2, -1, -1):
            da = self.gradients[f"da{i + 1}"]
            (
                self.gradients[f"da{i}"],
                self.gradients[f"dW{i + 1}"],
                self.gradients[f"db{i + 1}"],
            ) = self._backward_pass_hidden_single(da, i)

    def _update_param(self):
        """
        Updates parameters with the gradients calculated during the backpropagation iteration.
        """

        # pseudo code prior to implementation:

        # input: dw[l], db[l], parameters: ( W[l], b[l])
        # W[l] -= learning_rate*dW[l]
        # b[l] -= learning_rate*db[l]
        # output parameters: W[l], b[l] (updated)

        # There is two options, either regular gradient descent (gd) either applying momentum.
        for i in range(1, self.num_layers + 1):
            # If there is a momentum beta set different than 0 it updates the parameters with it.
            if self.optimizer1 == "momentum":
                self._update_param_with_momentum(i)
            if self.optimizer1 == "gd":
                self._update_param_gd(i)

    def _update_param_gd(self, i):
        """Updates the parameters using regular gradient descent"""
        self.param[f"W{i}"] -= self.learning_rate * self.gradients[f"dW{i}"]
        self.param[f"b{i}"] -= self.learning_rate * self.gradients[f"db{i}"]

    def _update_param_with_momentum(self, i):
        """Updates the parameters using momentum"""
        # Adjusting the velocities of the derivative of the weights
        self.velocities[f"dW{i}"] = (
            self.beta1 * self.velocities[f"dW{i}"]
            + (1 - self.beta1) * self.gradients[f"dW{i}"]
        )
        # Adjusting the velocities of the derivative of the bias
        self.velocities[f"db{i}"] = (
            self.beta1 * self.velocities[f"db{i}"]
            + (1 - self.beta1) * self.gradients[f"db{i}"]
        )

        # it finally updates the parameters with the custom learning rate and calculated velocities.
        self.param[f"W{i}"] -= self.learning_rate * self.velocities[f"dW{i}"]
        self.param[f"b{i}"] -= self.learning_rate * self.velocities[f"db{i}"]

    # def fit_sgd_optimizer(self, X_train, y_train, X_test, y_test):
    """trial of implementation of the SGD optimizer not for production"""
    #     # todo: very very slow. Discarting this optimizer (not sure if it even works well)
    #     for i in range(self.epoch):
    #         total_loss = 0
    #         for j in range(0, X_train.shape[0]):
    #             X = X_train[j : j + 1]
    #             y = y_train[j : j + 1]
    #             activation_last = self._forward_pass(X)
    #             if self.loss_function == "BCE":
    #                 total_loss += self._loss_calc_BCE(activation_last, y)
    #             if self.loss_function == "CCE":
    #                 total_loss += self._loss_calc_CCE(activation_last, y)
    #             self._backward_pass(activation_last, y)
    #             self._update_param()
    #         total_loss_avg = total_loss / X_train.shape[0]
    #         self.losses.append(total_loss_avg)
    #         if not i % 100:
    #             # checking the accuracy in the train and test set every 100 epochs.
    #             train_acc = self.eval_accuracy(y_train, self._forward_pass(X_train))
    #             test_acc = self.eval_accuracy(y_test, self._forward_pass(X_test))
    #             self.train_accuracy.append(train_acc)
    #             self.test_accuracy.append(test_acc)

    def _fit_with_mini_batch(self, X_train, y_train, X_test, y_test):
        """Full neuronal network iteration using mini batch gradient descent"""
        # iterates through the given epochs
        X_train_full = X_train
        y_train_full = y_train
        for i in range(self.epoch):
            # generates the random mini batches
            minibatches = self._gen_random_mini_batches(X_train, y_train)

            total_loss = 0  # establish the loss at 0 in every epoch
            for m in minibatches:
                (X_train, y_train) = m
                activation_last = self._forward_pass(X_train)  # forward pass iteration

                # depending on the loss function parameter it calculates the respective loss
                # Categorical cross entropy
                if self.loss_function == "BCE":
                    total_loss += self._loss_calc_BCE(activation_last, y_train)
                # Binary cross entropy
                if self.loss_function == "CCE":
                    total_loss += self._loss_calc_CCE(activation_last, y_train)

                self._backward_pass(activation_last, y_train)  # backward pass iteration
                self._update_param()  # update the parameters

            # since using mini batches, we calculate the average of every mini batch loss value
            total_loss_avg = total_loss / len(minibatches)
            self.losses.append(total_loss_avg)

            # In every 100 epoch it adds the training accuracy and the test accuracy
            # todo: what if there is less than 100 epoch given?
            if not i % 1:
                # checking the accuracy in the train and test set every 100 epochs.
                train_acc = self.eval_accuracy(
                    y_train_full, self._forward_pass(X_train_full)
                )
                test_acc = self.eval_accuracy(y_test, self._forward_pass(X_test))
                self.train_accuracy.append(train_acc)
                self.test_accuracy.append(test_acc)
                if self.verbose:
                    print(f"Epoch {i}, loss: {total_loss_avg}")
                    print(f"Epoch {i}, Train accuracy: {train_acc}")
                    print(f"Epoch{i}, Test accuracy : {train_acc}")

    def _fit_without_mini_batch(self, X_train, y_train, X_test, y_test):
        """Full neuronal network iteration using regular gradient descent"""
        for i in range(self.epoch):
            activation_last = self._forward_pass(X_train)  # forward pass iteration

            # depending on the loss function parameter it calculates the respective loss
            # Categorical cross entropy
            if self.loss_function == "BCE":
                loss = self._loss_calc_BCE(activation_last, y_train)
            # Binary cross entropy
            if self.loss_function == "CCE":
                loss = self._loss_calc_CCE(activation_last, y_train)

            self.losses.append(loss)
            self._backward_pass(activation_last, y_train)  # backward pass iteration
            self._update_param()
            # In every 100 epoch it adds the training accuracy and the test accuracy
            # todo: what if there is less than 100 epoch given?
            if not i % 100:
                # checking the accuracy in the train and test set every 100 epochs.
                train_acc = self.eval_accuracy(y_train, activation_last)
                test_acc = self.eval_accuracy(y_test, self._forward_pass(X_test))
                self.train_accuracy.append(train_acc)
                self.test_accuracy.append(test_acc)
                if self.verbose:
                    print(f"Epoch {i}, loss: {loss}")
                    print(f"Epoch {i}, Train accuracy: {train_acc}")
                    print(f"Epoch{i}, Test accuracy : {train_acc}")

    def fit(self, X_train, y_train, X_test, y_test):
        """Fit training data in the model and train it with the hyperparameters selected"""
        # x and y should be preprocessed, y one-hot encoded.
        # self.num_layers_units.insert(0, X_train.shape[1])

        self._init_param()  # Initialize the parameters for the network training
        # if self.optimizer == "sgd":
        #     self.fit_sgd_optimizer(X_train, y_train, X_test, y_test)
        # If using regular gradient descent
        if not self.mini_batch:
            self._fit_without_mini_batch(X_train, y_train, X_test, y_test)
        # If using mini batch gradient descent
        if self.mini_batch:
            self._fit_with_mini_batch(X_train, y_train, X_test, y_test)

    def predict(self, X_test):
        """Predict function used to predict new values once the network has been trained"""
        self.dropout_rate = 0  # No dropout used during testing.
        activation_last = self._forward_pass(X_test)  # Unchanged forward pass
        return np.argmax(
            activation_last, axis=1
        )  # returns the index of the largest value along a axis 1

    def eval_accuracy(self, y_test, y_pred):
        """function used to evaluate the accuracy of the trained network."""
        # It is needed to convert the probabilities of the inputs into actual class labels
        # which is why it is needed to use np.argmax()
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        accuracy = np.mean(
            y_pred == y_test_labels
        )  # whether the prediction value and the actual value is the same or not.
        return accuracy

    def create_confusion_matrixes(self, X_train, X_test, y_train, y_test):
        y_train_pred = np.argmax(self._forward_pass(X_train), axis=1)
        y_test_pred = np.argmax(self._forward_pass(X_test), axis=1)
        y_train_true = np.argmax(y_train, axis=1)
        y_test_true = np.argmax(y_test, axis=1)
        return y_train_pred, y_test_pred, y_train_true, y_test_true
