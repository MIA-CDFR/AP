import numpy as np

from abc import ABC, abstractmethod


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        # To avoid the additional multiplication by -1 just swap the y_pred and y_true.
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true, y_pred):
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return ((-y_true / p) + ((1 - y_true) / (1 - p)))


class CategoricalCrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(p), axis=1))

    def derivative(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        return (y_pred - y_true) / batch_size
