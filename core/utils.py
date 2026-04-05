import numpy as np

# ---------- DATA HANDLING ----------

def train_test_split(X, y, test_size=0.2, shuffle=True):
    n_samples = len(X)
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    split = int(n_samples * (1 - test_size))
    train_idx = indices[:split]
    test_idx  = indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def handle_missing(X):
    return np.where(X == None, np.nan, X)

def to_numpy(data):
    return np.array(data)

def safe_divide(a, b):
    return a / (b + 1e-8)

def check_nan(X):
    return np.isnan(X).sum()

def validate_input(X, y):
    assert len(X) == len(y),  "X and y must have same length"
    assert len(X.shape) == 2, "X must be 2D"
    assert len(y.shape) == 1, "y must be 1D"

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def softmax(z):
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp, axis=1, keepdims=True)

def detect_task(y):
    """Auto-detect classification vs regression."""
    return "classification" if len(np.unique(y)) <= 20 else "regression"

def detect_column_types(X):
    """
    Returns dict {col_index: 'numerical' | 'categorical' | 'boolean'}.
    Plain function — no self — importable from anywhere.
    """
    types = {}
    for i in range(X.shape[1]):
        col = X[:, i]
        if col.dtype == bool or all(isinstance(x, bool) for x in col):
            types[i] = "boolean"
        elif col.dtype.kind in ("i", "u", "f"):
            types[i] = "numerical"
        else:
            try:
                col.astype(float)
                types[i] = "numerical"
            except (ValueError, TypeError):
                types[i] = "categorical"
    return types
