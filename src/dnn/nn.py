import numpy as np

from dnn.layers import DenseLayer, Layer


class NeuralNetwork:
    """
    Feedforward neural network in python which can be used for text classification

    Main goal is to classify text in order to identify text is comming from a LLM or Human,
    and if is comming from LLM then try to identify from which LLM family is comming.
    """

    layers: list[Layer]

    def __init__(self):
        self.layers = []

    def add_layer(self, layer: Layer):
        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())

        if isinstance(layer, DenseLayer):
            layer.initialize()

        self.layers.append(layer)

    def _set_training_mode(self, training):
        for layer in self.layers:
            if hasattr(layer, "set_training"):
                layer.set_training(training)

    def _snapshot_state(self):
        state = []
        for layer in self.layers:
            if hasattr(layer, "weights") and hasattr(layer, "biases") and layer.weights is not None:
                state.append((np.copy(layer.weights), np.copy(layer.biases)))
            else:
                state.append(None)
        return state

    def _restore_state(self, state):
        for layer, layer_state in zip(self.layers, state):
            if layer_state is None:
                continue
            layer.weights, layer.biases = layer_state

    def predict(self, input_data, training=False):
        self._set_training_mode(training)
        output = input_data
        for layer in self.layers:
            output = layer.forward_propagation(output)

        return output

    @staticmethod
    def _cross_entropy(y_true_one_hot, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(np.sum(y_true_one_hot * np.log(y_pred), axis=1))

    @staticmethod
    def _accuracy(y_true, y_pred):
        return np.mean(np.argmax(y_pred, axis=1) == y_true)

    def fit(
        self,
        x_train,
        y_train,
        epochs,
        learning_rate,
        batch_size=32,
        x_val=None,
        y_val=None,
        verbose_every=50,
        patience=25,
        min_delta=1e-4,
        lr_decay=0.5,
        lr_patience=10,
    ):
        num_classes = int(np.max(y_train)) + 1
        class_counts = np.bincount(y_train, minlength=num_classes)
        class_weights = len(y_train) / (num_classes * np.maximum(class_counts, 1))

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        if self.layers and isinstance(self.layers[0], DenseLayer):
            self.layers[0].set_input_shape((x_train.shape[1],))
            self.layers[0].initialize()

        best_val_loss = np.inf
        best_state = self._snapshot_state()
        epochs_without_improvement = 0
        lr_epochs_without_improvement = 0
        current_lr = learning_rate

        for epoch in range(epochs):
            indices = np.random.permutation(len(x_train))
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, len(x_shuffled), batch_size):
                end = start + batch_size
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_batch_one_hot = np.eye(num_classes)[y_batch]
                output_batch = self.predict(x_batch, training=True)

                if output_batch.shape != y_batch_one_hot.shape:
                    raise ValueError(
                        f"Output shape {output_batch.shape} does not match y_one_hot shape {y_batch_one_hot.shape}. "
                        "Check the number of units in the final DenseLayer."
                    )

                sample_weights = class_weights[y_batch].reshape(-1, 1)
                error = (output_batch - y_batch_one_hot) * sample_weights

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, current_lr)

            train_output = self.predict(x_train, training=False)
            y_train_one_hot = np.eye(num_classes)[y_train]
            train_loss = self._cross_entropy(y_train_one_hot, train_output)
            train_acc = self._accuracy(y_train, train_output)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if x_val is not None and y_val is not None:
                y_val_one_hot = np.eye(num_classes)[y_val]
                val_output = self.predict(x_val, training=False)
                val_loss = self._cross_entropy(y_val_one_hot, val_output)
                val_acc = self._accuracy(y_val, val_output)

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if (epoch + 1) % verbose_every == 0 or epoch == 0:
                    print(
                        f"Epoch {epoch + 1}/{epochs} [lr={current_lr:.6f}] - "
                        f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                        f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
                    )

                if val_loss < (best_val_loss - min_delta):
                    best_val_loss = val_loss
                    best_state = self._snapshot_state()
                    epochs_without_improvement = 0
                    lr_epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    lr_epochs_without_improvement += 1

                if lr_epochs_without_improvement >= lr_patience:
                    current_lr *= lr_decay
                    lr_epochs_without_improvement = 0
                    print(f"  -> LR reduced to {current_lr:.6f} at epoch {epoch + 1}")

                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}. Best val_loss: {best_val_loss:.4f}")
                    self._restore_state(best_state)
                    return history
            else:
                if (epoch + 1) % verbose_every == 0 or epoch == 0:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"loss: {train_loss:.4f} - acc: {train_acc:.4f}"
                    )

        if x_val is not None and y_val is not None:
            self._restore_state(best_state)

        return history
