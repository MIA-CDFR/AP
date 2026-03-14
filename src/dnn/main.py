from pathlib import Path

from dnn.prepare.dataset import DatasetLoader
from dnn.prepare.model import Model, LinearRegressionModel

if __name__ == "__main__":
    datasets = DatasetLoader.load_datasets()
    model = Model(n_classes=len(datasets.class_names))
    linear_model = LinearRegressionModel(n_classes=len(datasets.class_names))

    print("Training the model...")
    model.train(
        datasets.X_train,
        datasets.y_train,
        datasets.X_test,
        datasets.y_test,
        epochs=50,
        batch_size=32,
    )
    model.attach_preprocessors(datasets)
    linear_model.train(
        datasets.X_train,
        datasets.y_train,
        datasets.X_test,
        datasets.y_test,
    )
    linear_model.attach_preprocessors(datasets)

    path = Path(__file__).resolve().parent
    model.save(path / "rnn-model.pkl")
    linear_model.save(path / "linear-model.pkl")

    loaded_model = Model.load(path / "rnn-model.pkl")
    loaded_linear_model = LinearRegressionModel.load(path / "linear-model.pkl")

    print("\nEvaluating on test set...")
    loaded_model.predict(datasets.X_test, datasets.y_test)
    loaded_linear_model.predict(datasets.X_test, datasets.y_test)
