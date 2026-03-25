import argparse
from pathlib import Path

from models.dnn.prepare.dataset import DatasetLoader
from models.dnn.prepare.model import Model, LinearRegressionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DNN text classifier")
    parser.add_argument(
        "--dataset",
        choices=["revealed", "mixed"],
        default="mixed",
        help="Dataset source: 'revealed' uses subm1_labels_revealed.csv; 'mixed' uses the original combined datasets.",
    )
    parser.add_argument(
        "--revealed-path",
        type=str,
        default=None,
        help="Optional path to a revealed CSV file with Text and Label columns.",
    )
    parser.add_argument(
        "--fit-all",
        action="store_true",
        help="Use all revealed rows for both train and eval (memorization mode).",
    )
    parser.add_argument(
        "--eval-revealed",
        action="store_true",
        help="When training on mixed datasets, evaluate on revealed labels instead of mixed test split.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "revealed":
        # Train on revealed labels
        test_size = 0.0 if args.fit_all else 0.2
        datasets = DatasetLoader.load_revealed_dataset(test_size=test_size)
        X_eval = datasets.X_train if args.fit_all else datasets.X_test
        y_eval = datasets.y_train if args.fit_all else datasets.y_test
    else:
        # Train on mixed datasets
        datasets = DatasetLoader.load_datasets()
        
        if args.eval_revealed:
            # Load revealed labels for evaluation only
            print(f"Training on mixed datasets, evaluating on revealed labels...")
            revealed_loader = DatasetLoader.load_revealed_dataset(test_size=0.0)  # Load all as "test" (no train split)
            X_eval = revealed_loader.X_train  # All revealed samples used for evaluation
            y_eval = revealed_loader.y_train
        else:
            # Use mixed test split for evaluation
            X_eval = datasets.X_test
            y_eval = datasets.y_test

    if args.dataset == "revealed" and args.fit_all:
        model = Model(
            n_classes=len(datasets.class_names),
            hidden_units=(512, 256),
            dropout_rates=(0.0, 0.0),
        )
    else:
        model = Model(n_classes=len(datasets.class_names))
    linear_model = LinearRegressionModel(n_classes=len(datasets.class_names))

    print("Training the model...")
    model.train(
        datasets.X_train,
        datasets.y_train,
        X_eval,
        y_eval,
        epochs=180 if (args.dataset == "revealed" and args.fit_all) else (120 if args.dataset == "revealed" else 80),
        batch_size=16,
        learning_rate=0.01 if args.fit_all else 0.005,
        patience=50,
        verbose_every=10,
    )
    model.attach_preprocessors(datasets)
    linear_model.train(
        datasets.X_train,
        datasets.y_train,
        X_eval,
        y_eval,
    )
    linear_model.attach_preprocessors(datasets)

    path = Path(__file__).resolve().parents[3] / "models"
    model.save(path / "numpy-dnn-model.pkl")
    linear_model.save(path / "linear-model.pkl")

    loaded_model = Model.load(path / "numpy-dnn-model.pkl")
    loaded_linear_model = LinearRegressionModel.load(path / "linear-model.pkl")

    print("\nEvaluating on test set...")
    loaded_model.predict(X_eval, y_eval)
    loaded_linear_model.predict(X_eval, y_eval)
