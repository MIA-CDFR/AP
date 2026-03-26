import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Maintains class proportions in train and test sets.
    """
    np.random.seed(random_state)
    X = np.asarray(X)
    y = np.asarray(y)

    # Get unique classes and their indices
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []

    # Split each class separately
    for class_label in unique_classes:
        # Find all indices for this class
        class_indices = np.where(y == class_label)[0]
        n_samples = len(class_indices)

        # Shuffle indices
        np.random.shuffle(class_indices)

        # Calculate split point
        split_point = int(n_samples * (1 - test_size))

        # Add to train/test
        train_indices.extend(class_indices[:split_point])
        test_indices.extend(class_indices[split_point:])

    # Shuffle final indices
    train_indices = np.array(train_indices, dtype=int)
    test_indices = np.array(test_indices, dtype=int)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Split data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
