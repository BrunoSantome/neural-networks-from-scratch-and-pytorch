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
        layers_units,
        epoch=20,
        learning_rate=0.01,
        seed=42,
        hidden_activation="relu",
        output_activation="sigmoid",
        dropout=False,
        batch_size=1,
        loss_function="classification_cross_entropy",
    ):
        self.num_layers_units = layers_units
        self.num_layers = len(layers_units) - 1
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.seed = seed
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.param = {}
        self.gradients = {}
        self.storage_layers = []

        # self.init_param() only called when the data is fitted into the model (to fit dimension of first input layer)

    def init_param(self):
        """method to initialisize randomly the weights and bias of the network based on the given architecture

        Returns:
            param: dictionary that contains all the weights and bias initialized randomely
        """
        # example of layers: [4,8,4,1]
        # Weights: (4,8), (8,4), (4,1), (4,1)
        np.random.seed(self.seed)
        for l in range(1, len(self.num_layers_units)):
            # Dimensions of Wl: (n[l-1], n[l]) ==> it is the same for dWl (for backward pass)
            self.param[f"W{l}"] = (
                np.random.rand(self.num_layers_units[l - 1], self.num_layers_units[l])
                * 0.01
            )
            # Dimensions of bl: ( 1, n[l],) ==> it is the same for dbl (for backward pass)
            self.param[f"b{l}"] = np.zeros((1, self.num_layers_units[l])) * 0.01

    def forward_pass_hidden_single(self, activation_last, W, b):
        Z = np.dot(activation_last, W) + b
        if self.hidden_activation == "relu":
            activation_current, Z = relu(Z)
        if self.hidden_activation == "sigmoid":  # We could implement tanh
            # It might not be useful to return the Z value too.
            activation_current, Z = sigmoid(Z)
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
        # check pseudo code for the loss function, cross-entropy for classification
        loss = np.mean((y - activation_last) ** 2)
        return loss

    def loss_calc_CCE(self, activation_last, y):
        # Classification Cross Entropy
        loss = -np.mean(
            (y * np.log(activation_last)) + ((1 - y) * np.log(1 - activation_last))
        )
        return loss

    def backward_pass_calc(self, dZ, storage):
        _, W, _, activation_last = storage  # is Z and b necessary? Z yes, b maybe not
        n = activation_last.shape[1]
        dW = 1 / n * activation_last.T.dot(dZ)  # a1.T.dot(a2_delta)
        db = 1 / n * np.sum(dZ, axis=1)  # Check this
        da = dZ.dot(W.T)  # a2_delta.dot(w2.T) in coursework.
        return da, dW, db

    def backward_pass_hidden_single(self, da, layer):
        Z, _, _, _ = self.storage_layers[layer]
        if self.hidden_activation == "relu":
            dZ = relu_back_pass(da, Z)
        if self.hidden_activation == "sigmoid":
            dZ = sigmoid_back_pass(da, Z)
        da, dW, db = self.backward_pass_calc(dZ, self.storage_layers[layer])
        return da, dW, db

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
        # simplification of the product of derivative of the loss and derivative of the output_activation function

        (
            self.gradients[f"da{self.num_layers - 1}"],
            self.gradients[f"dW{self.num_layers}"],
            self.gradients[f"db{self.num_layers}"],
        ) = self.backward_pass_calc(
            dZ,
            self.storage_layers[self.num_layers - 1],
        )

        for i in range(self.num_layers - 2, -1, -1):
            print(i)
            da = self.gradients[f"da{i + 1}"]
            (
                self.gradients[f"da{i}"],
                self.gradients[f"dW{i + 1}"],
                self.gradients[f"db{i + 1}"],
            ) = self.backward_pass_hidden_single(da, i)

        # for l in reversed(range(layers - 1)):

    def update_param(self):
        ...
        # input: dw[l], db[l], parameters: ( W[l], b[l])
        # W[l] -= learning_rate*dW[l]
        # b[l] -= learning_rate*db[l]
        # output parameters: W[l], b[l] (updated)
