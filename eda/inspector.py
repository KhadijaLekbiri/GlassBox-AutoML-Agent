import numpy as np
from core.stats import mean, median, mode, std, skewness, kurtosis, detect_outliers
from core.utils import detect_column_types

class DataInspector:
    """
    Automated EDA — wraps core/stats.py functions.
    Usage:
        inspector = DataInspector()
        profile   = inspector.fit(X, feature_names)
    """

    def __init__(self):
        self.column_types = {}
        self.statistics   = {}
        self.outliers     = {}

    def fit(self, X, feature_names=None):
        """
        Parameters
        ----------
        X             : 2-D NumPy float array  (n_samples, n_features)
        feature_names : list of column name strings (optional)

        Returns
        -------
        dict with keys: statistics, outliers, column_types,
                        n_rows, n_cols, numeric_cols, categorical_cols,
                        outliers_flagged, missing_filled
        """
        self.column_types = detect_column_types(X)
        if feature_names is None:
            feature_names = [f"col_{i}" for i in range(X.shape[1])]

        stats = {}
        for i, col in enumerate(X.T):
            col_type = self.column_types[i]
            name     = feature_names[i]

            if col_type == "numerical":
                col = col.astype(float)
                stats[name] = {
                    "type":     "numerical",
                    "mean":     float(mean(col)),
                    "median":   float(median(col)),
                    "mode":     float(mode(col)),
                    "std":      float(std(col)),
                    "skew":     float(skewness(col)),
                    "kurtosis": float(kurtosis(col)),
                    "missing":  int(np.isnan(col).sum()),
                }
            elif col_type == "categorical":
                stats[name] = {
                    "type":    "categorical",
                    "mode":    str(mode(col)),
                    "n_unique": int(len(np.unique(col))),
                    "missing": int(np.sum(col == "")),
                }
            elif col_type == "boolean":
                col_int = col.astype(int)
                stats[name] = {
                    "type": "boolean",
                    "mean": float(mean(col_int)),
                    "mode": float(mode(col_int)),
                }

        self.statistics = stats
        self.outliers   = detect_outliers(X, self.column_types)

        numeric_cols     = [feature_names[i] for i, t in self.column_types.items() if t == "numerical"]
        categorical_cols = [feature_names[i] for i, t in self.column_types.items() if t == "categorical"]
        outliers_flagged = sum(len(v) for v in self.outliers.values())
        missing_filled   = sum(
            s.get("missing", 0) for s in stats.values()
        )

        return {
            "statistics":      stats,
            "outliers":        self.outliers,
            "column_types":    self.column_types,
            "n_rows":          X.shape[0],
            "n_cols":          X.shape[1],
            "numeric_cols":    numeric_cols,
            "categorical_cols": categorical_cols,
            "outliers_flagged": outliers_flagged,
            "missing_filled":  missing_filled,
            "feature_names":   feature_names,
        }

    def correlation_matrix(self, X):
        num_cols = [i for i, t in self.column_types.items() if t == "numerical"]
        if len(num_cols) < 2:
            return np.array([])
        return np.corrcoef(X[:, num_cols].astype(float).T)

    def correlated_pairs(self, X, threshold=0.7):
        corr = self.correlation_matrix(X)
        if corr.size == 0:
            return []
        pairs = []
        n = corr.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) >= threshold:
                    pairs.append((i, j))
        return pairs
