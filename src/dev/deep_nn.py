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
    ):
        self.num_layers_units = layers_units
        self.num_layers = len(layers_units)
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.seed = seed
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.batch_size = batch_size
        self.param = {}
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
        for l in range(1, self.num_layers):
            # Dimensions of Wl: (n[l], n[l-1]) ==> it is the same for dWl (for backward pass)
            self.param[f"W{l}"] = np.random.rand(
                self.num_layers_units[l], self.num_layers_units[l - 1]
            )
            # Dimensions of bl: (n[l], 1) ==> it is the same for dbl (for backward pass)
            self.param[f"b{l}"] = np.zeros((self.num_layers_units[l], 1))

    def forward_pass_hidden_single(self, activation_last, W, b):
        Z = np.dot(W, activation_last) + b
        if self.hidden_activation == "relu":
            activation_current, Z = relu(Z)
        if self.hidden_activation == "sigmoid":  # We could implement tanh
            # It might not be useful to return the Z value too.
            activation_current, Z = sigmoid(Z)
        return activation_current, (Z, W, b, activation_last)

    def forward_pass_output_layer(self, activation_last, W, b):
        print(W.shape)
        print(activation_last)
        Z = np.dot(W, activation_last) + b
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
        layers = self.num_layers - 1  # -1 because the first layer does not count.
        # Hidden layers
        for l in range(1, layers):
            activation_last = activation_current
            activation_current, storage = self.forward_pass_hidden_single(
                activation_last, self.param[f"W{l}"], self.param[f"b{l}"]
            )
            self.storage_layers.append(storage)

        # Output layer
        activation_output, storage = self.forward_pass_output_layer(
            activation_current,
            self.param[f"W{layers}"],
            self.param[f"b{layers}"],
        )
        self.storage_layers.append(storage)

        return activation_output

    def loss_calc(self):
        ...
        # Loss(y', y)
        # By calculating the derivative, and vectorize it,
        # it is the input for the start of the backward pass. so da[-1]
        # check pseudo code for the loss function, cross-entropy for classification

    def backward_pass(self):
        ...
        # input da[l], storage: (z[l], W[l], b[l])).
        # dz[l] = da[l] * g[l]'(z[l])
        # dw[l] = 1/m * dz[l]. a[l-1].T
        # db[l] = 1/m * dz[l]
        # da[l] = W[l].T* dz[l]
        # output: da[l-1] dw[l], db[l]

    def update_param(self):
        ...
        # input: dw[l], db[l], parameters: ( W[l], b[l])
        # W[l] -= learning_rate*dW[l]
        # b[l] -= learning_rate*db[l]
        # output parameters: W[l], b[l] (updated)
