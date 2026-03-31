import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


def minkowski_distance(x1, x2, p=3):
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)


def cosine_similarity(x1, x2):
    dot = np.dot(x1, x2)
    norm = np.linalg.norm(x1) * np.linalg.norm(x2)
    return dot / (norm + 1e-8)


def pairwise_distance(X, x, metric="euclidean"):
    distances = []

    for row in X:
        if metric == "euclidean":
            d = euclidean_distance(row, x)
        elif metric == "manhattan":
            d = manhattan_distance(row, x)
        else:
            d = euclidean_distance(row, x)

        distances.append(d)

    return np.array(distances)

def euclidean_distance_vectorized(X, x):
    return np.sqrt(np.sum((X - x) ** 2, axis=1))