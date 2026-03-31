import numpy as np
from MathFoundations import stats as stat 

class DataInspector:
    """
    Automated EDA class that wraps Phase 1 functions
    """

    def __init__(self):
        self.column_types = {}
        self.statistics = {}
        self.outliers = {}

    # ---------------- Auto-Typing ----------------
    def detect_column_types(self, X):
        """
        Detects column types: numerical, categorical, boolean
        """
        types = {}
        for i in range(X.shape[1]):
            col = X[:, i]
            if all(isinstance(x, (int, float, np.integer, np.floating)) for x in col):
                types[i] = "numerical"
            elif all(isinstance(x, bool) for x in col):
                types[i] = "boolean"
            else:
                types[i] = "categorical"
        self.column_types = types
        return types
    # ---------------- Statistical Profiling ----------------
    def fit(self, X):
        """
        Computes statistics for each column and detects outliers
        """
        stats = {}
        self._detect_types(X)

        for i, col in enumerate(X.T):
            col_type = self.column_types[i]

            if col_type == "numerical":
                col = col.astype(float)
                stats[i] = {
                    "mean": stat.mean(col),
                    "median": stat.median(col),
                    "mode": stat.mode(col),
                    "std": stat.std(col),
                    "skew": stat.skewness(col),
                    "kurtosis": stat.kurtosis(col)
                }

            elif col_type == "categorical":
                stats[i] = {"mode": stat.mode(col)}

            elif col_type == "boolean":
                col = col.astype(int)
                stats[i] = {"mean": stat.mean(col), "mode": stat.mode(col)}

        self.statistics = stats

        # ---------------- Outliers ----------------
        self.outliers = stat.detect_outliers(X, self.column_types)

        return {"statistics": stats, "outliers": self.outliers}

    # ---------------- Pearson Correlation ----------------
    def correlation_matrix(self, X):
        """
        Returns Pearson correlation for numerical columns
        """
        num_cols = [i for i, t in self.column_types.items() if t == "numerical"]
        if len(num_cols) < 2:
            return np.array([])  # No correlation possible
        num_data = X[:, num_cols].astype(float)
        corr = np.corrcoef(num_data.T)
        return corr

    def correlated_pairs(self, X, threshold=0.7):
        """
        Returns list of column index pairs with correlation > threshold
        """
        corr = self.correlation_matrix(X)
        if corr.size == 0:
            return []

        pairs = []
        n = corr.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) >= threshold:
                    pairs.append([i, j])
        return pairs