import numpy as np

from models.dnn.classifiers.layers import DenseLayer, Layer


class NeuralNetwork:
    """
    Feedforward neural network in python which can be used for text classification

    Main goal is to classify text in order to identify text is comming from a LLM or Human,
    and if is comming from LLM then try to identify from which LLM family is comming.
    """

    layers: list[Layer]

    def __init__(self):
        self.layers = []

    def add_layer(self, layer: Layer, biases = None, weights = None):
        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())

        if isinstance(layer, DenseLayer):
            layer.initialize()

        if biases is not None: layer.set_biases(biases)
        if weights is not None: layer.set_weights(weights)

        self.layers.append(layer)

    def _set_training(self, training: bool):
        for layer in self.layers:
            if hasattr(layer, "set_training"):
                layer.set_training(training)

    def forward_propagation(self, X, training: bool = True):
        self._set_training(training)
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output

    def backward_propagation(self, output_error, learning_rate: float):
        error = output_error

        for layer in reversed(self.layers):
            error = layer.backward_propagation(error, learning_rate)

    def predict(self, X):
        return self.forward_propagation(X, training=False)
