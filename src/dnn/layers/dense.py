
import numpy as np

from dnn.layers.base import Layer


class DenseLayer(Layer):
    def __init__(self, n_units: int, lambda_reg: float = 1e-4):
        self.n_units = n_units
        self.lambda_reg = lambda_reg

    def initialize(self):
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        if self.input_shape() is None:
            self.weights = None
        else:
            fan_in = self.input_shape()[0]
            self.weights = np.random.randn(fan_in, self.n_units) * np.sqrt(2.0 / fan_in)

        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        return self

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        batch_size = output_error.shape[0] if output_error.ndim > 1 else 1

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error) / batch_size
        bias_error = np.sum(output_error, axis=0, keepdims=True) / batch_size

        self.weights -= learning_rate * (weights_error + self.lambda_reg * self.weights)
        self.biases -= learning_rate * bias_error
        return input_error

    def output_shape(self):
         return (self.n_units,)

    def parameters(self):
        return self.weights, self.biases
