import numpy as np

# ---------- BASIC ----------

def dot(A, B):
    return np.dot(A, B)


def transpose(A):
    return A.T


def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


# ---------- LINEAR ALGEBRA ----------

def inverse(A):
    return np.linalg.inv(A)


def determinant(A):
    return np.linalg.det(A)


def solve(A, b):
    return np.linalg.solve(A, b)


# ---------- NORMALIZATION ----------

def normalize(col,mean,std):
    mean = np.mean(col, axis=0)
    std = np.std(col, axis=0)
    return (col - mean) / (std + 1e-8)


def minmax_scale(X):
    min_val = np.min(col, axis=0)
    max_val = np.max(col, axis=0)
    return (col - min_val) / (max_val - min_val + 1e-8)



def compute_gradient(X, y, weights, bias):
    n = len(y)
    y_pred = X @ weights + bias

    dw = (1/n) * (X.T @ (y_pred - y))
    db = (1/n) * np.sum(y_pred - y)

    return dw, db