import numpy as np


class NeuronalNetwork:
    """Fully parametrizable neural network class"""

    def __init__(
        self,
        layers_units,
        epoch=20,
        learning_rate=0.01,
        seed=42,
        hidden_activation="relu",
        output_activation="softmax",
        dropout=False,
        batch_size=1,
    ):
        self.layers_units = layers_units
        self.layers = len(layers_units)
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.seed = seed
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.batch_size = batch_size
        # self.init_param() only called when the data is fitted into the model (to fit dimension of first input layer)

    def init_param(self):
        """method to initialisize randomly the weights and bias of the network based on the given architecture

        Returns:
            param: dictionary that contains all the weights and bias initialized randomely
        """
        # example of layers: [4,8,4,1]
        # Weights: (4,8), (8,4), (4,1), (4,1)
        param = {}
        np.random.seed = self.seed
        for l in range(1, self.layers):
            # Dimensions of Wl: (n[l], n[l-1]) ==> it is the same for dWl (for backward pass)
            param[f"W{l}"] = np.random.rand(
                self.layers_units[l], self.layers_units[l - 1]
            )
            # Dimensions of bl: (n[l], 1) ==> it is the same for dbl (for backward pass)
            param[f"b{l}"] = np.zeros((self.layers_units[l], 1))

        return param

    def forward_pass(self):
        ...
        # input: a[l-1]
        # z[l] = W[l]*A[l-1] + b[l]
        # a[l] = g[l](z[l])
        # output: a[l], storage: (z[l]: (W[l]*a[l-1]+b[l]), W[l], b[l], a[l-1])

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
