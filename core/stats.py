import numpy as np

# ---------- BASIC STATS ----------

def mean(x):
    return np.sum(x) / len(x)


def median(x):
    x_sorted = np.sort(x)
    n = len(x)
    if n % 2 == 0:
        return (x_sorted[n//2 - 1] + x_sorted[n//2]) / 2
    return x_sorted[n//2]


def mode(x):
    values, counts = np.unique(x, return_counts=True)
    return values[np.argmax(counts)]


def variance(x):
    m = mean(x)
    return np.sum((x - m) ** 2) / len(x)


def std(x):
    return np.sqrt(variance(x))


def skewness(x):
    m = mean(x)
    s = std(x) + 1e-8
    return np.mean(((x - m) / s) ** 3)


def kurtosis(x):
    m = mean(x)
    s = std(x) + 1e-8
    return np.mean(((x - m) / s) ** 4)


# ---------- CORRELATION ----------

def pearson_correlation(x, y):
    mx, my = mean(x), mean(y)
    numerator = np.sum((x - mx) * (y - my))
    denominator = np.sqrt(np.sum((x - mx)**2) * np.sum((y - my)**2)) + 1e-8
    return numerator / denominator


def correlation_matrix(X):
    n_features = X.shape[1]
    corr = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            corr[i, j] = pearson_correlation(X[:, i], X[:, j])

    return corr


# ---------- OUTLIER DETECTION ----------

def iqr_bounds(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def detect_outliers(X, column_types):
    """
    Detects outliers in each numerical column of X based on IQR.
    Returns a dictionary {col_index: array_of_outliers}
    """
    outliers = {}
    for i, col in enumerate(X.T):
        if column_types[i] == "numerical":
            col = col.astype(float)
            lower, upper = iqr_bounds(col)
            mask = (col < lower) | (col > upper)
            if np.any(mask):
                outliers[i] = col[mask].tolist()  # store the outlier values
    return outliers


