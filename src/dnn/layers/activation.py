import numpy as np

from typing import Callable

from dnn.layers.base import Layer

class ActivationLayer(Layer):

    activation: Callable
    derivative: Callable

    def __init__(self, activation: Callable, derivative: Callable):
        self.activation = activation
        self.derivative = derivative

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.derivative(self.input) * output_error

    def parameters(self):
        pass

    def output_shape(self):
        return self.input_shape()


class ReLU(ActivationLayer):
    def __init__(self):
        super().__init__(self.relu, self.derivative)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        return np.where(x > 0, 1, 0)


class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__(self.sigmoid, self.derivative)

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        s = Sigmoid.sigmoid(x)
        return s * (1 - s)


class Softmax(ActivationLayer):
    def __init__(self):
        super().__init__(self.softmax, None)

    @staticmethod
    def softmax(x):
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward_propagation(self, output_error, learning_rate):
        return output_error

    def parameters(self):
        return None


class Dropout(Layer):
    def __init__(self, rate=0.2):
        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be in [0, 1).")
        self.rate = rate
        self.keep_prob = 1.0 - rate
        self.training = True
        self.mask = None

    def set_training(self, training):
        self.training = training

    def forward_propagation(self, input_data):
        self.input = input_data
        if not self.training or self.rate == 0:
            return input_data

        self.mask = (np.random.rand(*input_data.shape) < self.keep_prob).astype(input_data.dtype)
        self.mask /= self.keep_prob
        return input_data * self.mask

    def backward_propagation(self, output_error, learning_rate):
        if not self.training or self.rate == 0:
            return output_error
        return output_error * self.mask

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return None
