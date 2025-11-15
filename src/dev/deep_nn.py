import numpy as np


class NeuronalNetwork:
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
        self.init_param()

    def init_param(self):
        # example of layers: [4,8,4,1]
        # Weights: (4,8), (8,4), (4,1), (4,1)
        param = {}
        np.random.seed = self.seed
        for l in range(1, self.layers):
            param[f"W{l}"] = np.random.rand(
                self.layers_units[l], self.layers_units[l - 1]
            )
            param[f"b{l}"] = np.zeros((self.layers_units[l], 1))

        return param
