import numpy as np

# ---------- DATA HANDLING ----------

def train_test_split(X, y, test_size=0.2, shuffle=True):
    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    split = int(n_samples * (1 - test_size))

    train_idx = indices[:split]
    test_idx = indices[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ---------- DATA CLEANING ----------

def handle_missing(X):
    return np.where(X == None, np.nan, X)


def to_numpy(data):
    return np.array(data)


# ---------- SAFE OPERATIONS  ----------

def safe_divide(a, b):
    return a / (b + 1e-8)


def check_nan(X):
    return np.isnan(X).sum()


def validate_input(X, y):
    assert len(X) == len(y), "X and y must have same length"
    assert len(X.shape) == 2, "X must be 2D"
    assert len(y.shape) == 1, "y must be 1D"


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp, axis=1, keepdims=True)

def detect_task(y):
    unique_vals = len(np.unique(y))

    if unique_vals <= 10:
        return "classification"
    return "regression"