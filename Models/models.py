import numpy as np


# ─────────────────────────────────────────────────────────
# Linear Regression
# ─────────────────────────────────────────────────────────
class LinearRegression:
    def __init__(self, lr=0.01, epochs=200):
        self.lr     = lr
        self.epochs = epochs
        self.w      = None
        self.b      = 0

    def fit(self, X, y):
        X, y = X.astype(float), y.astype(float)
        n, features = X.shape
        self.w = np.zeros(features)
        self.b = 0
        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            error  = y_pred - y
            self.w -= self.lr * (2/n) * (X.T @ error)
            self.b -= self.lr * (2/n) * np.sum(error)

    def predict(self, X):
        return X.astype(float) @ self.w + self.b

    @property
    def feature_importances_(self):
        if self.w is None: return None
        a = np.abs(self.w)
        return a / (a.sum() + 1e-10)

    def score(self, X, y):
        y, y_pred = y.astype(float), self.predict(X)
        return 1 - ((y - y_pred)**2).sum() / ((y - y.mean())**2).sum()


# ─────────────────────────────────────────────────────────
# Logistic Regression
# ─────────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self, lr=0.01, epochs=200):
        self.lr     = lr
        self.epochs = epochs
        self.w      = None
        self.b      = 0

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X, y = X.astype(float), y.astype(float)
        n, features = X.shape
        self.w = np.zeros(features)
        self.b = 0
        for _ in range(self.epochs):
            y_pred = self._sigmoid(X @ self.w + self.b)
            error  = y_pred - y
            self.w -= self.lr * (1/n) * (X.T @ error)
            self.b -= self.lr * (1/n) * np.sum(error)

    def predict(self, X):
        return (self._sigmoid(X.astype(float) @ self.w + self.b) >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        if self.w is None: return None
        a = np.abs(self.w)
        return a / (a.sum() + 1e-10)

    def score(self, X, y):
        return np.mean(self.predict(X) == y.astype(int))


# ─────────────────────────────────────────────────────────
# Gaussian Naive Bayes
# ─────────────────────────────────────────────────────────
class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes = self.prior = self.mean = self.var = None

    def fit(self, X, y):
        X, y = X.astype(float), y.astype(int)
        self.classes = np.unique(y)
        n_classes, n_features = len(self.classes), X.shape[1]
        self.prior = np.zeros(n_classes)
        self.mean  = np.zeros((n_classes, n_features))
        self.var   = np.zeros((n_classes, n_features))
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.prior[idx]  = len(X_c) / len(X)
            self.mean[idx]   = np.mean(X_c, axis=0)
            self.var[idx]    = np.var(X_c, axis=0) + self.var_smoothing

    def _log_gaussian(self, x, mean, var):
        return -0.5 * np.log(2 * np.pi * var) - (x - mean)**2 / (2 * var)

    def predict(self, X):
        X = X.astype(float)
        preds = []
        for x in X:
            log_probs = [
                np.log(self.prior[i]) + np.sum(self._log_gaussian(x, self.mean[i], self.var[i]))
                for i in range(len(self.classes))
            ]
            preds.append(self.classes[np.argmax(log_probs)])
        return np.array(preds)


# ─────────────────────────────────────────────────────────
# Decision Tree  (classification AND regression)
# ─────────────────────────────────────────────────────────
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, task="classification"):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.task              = task
        self.tree              = None
        self._importances      = None

    # ── public API ──────────────────────────────────────
    def fit(self, X, y):
        X = X.astype(float)
        y = y.astype(int) if self.task == "classification" else y.astype(float)
        self._n_features  = X.shape[1]
        self._importances = np.zeros(self._n_features)
        self.tree = self._build(X, y, 0)
        s = self._importances.sum()
        if s > 0:
            self._importances /= s

    def predict(self, X):
        return np.array([self._predict_row(row, self.tree) for row in X.astype(float)])

    @property
    def feature_importances_(self):
        return self._importances

    # ── internals ───────────────────────────────────────
    def _impurity(self, y):
        if self.task == "classification":
            counts = np.bincount(y.astype(int))
            p = counts / len(y)
            return 1 - np.sum(p**2)           # Gini
        else:
            return np.var(y)                   # MSE variance

    def _leaf(self, y):
        if self.task == "classification":
            return int(np.bincount(y.astype(int)).argmax())
        return float(np.mean(y))

    def _best_split(self, X, y):
        best_imp, best_feat, best_val = self._impurity(y), None, None
        for fi in range(X.shape[1]):
            for val in np.unique(X[:, fi]):
                lm = X[:, fi] <= val
                rm = ~lm
                if lm.sum() == 0 or rm.sum() == 0:
                    continue
                imp = (lm.sum() * self._impurity(y[lm]) +
                       rm.sum() * self._impurity(y[rm])) / len(y)
                if imp < best_imp:
                    best_imp, best_feat, best_val = imp, fi, val
        return best_feat, best_val

    def _build(self, X, y, depth):
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                len(np.unique(y)) == 1):
            return self._leaf(y)

        fi, val = self._best_split(X, y)
        if fi is None:
            return self._leaf(y)

        lm, rm = X[:, fi] <= val, X[:, fi] > val
        gain = self._impurity(y) - (
            lm.sum() * self._impurity(y[lm]) +
            rm.sum() * self._impurity(y[rm])
        ) / len(y)
        self._importances[fi] += gain * len(y)

        return {
            "feature":   fi,
            "threshold": val,
            "left":      self._build(X[lm], y[lm], depth + 1),
            "right":     self._build(X[rm], y[rm], depth + 1),
        }

    def _predict_row(self, row, node):
        if not isinstance(node, dict):
            return node
        branch = "left" if row[node["feature"]] <= node["threshold"] else "right"
        return self._predict_row(row, node[branch])


# ─────────────────────────────────────────────────────────
# Random Forest  (classification AND regression)
# ─────────────────────────────────────────────────────────
class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2,
                 max_features=None, task="classification"):
        self.n_trees           = n_trees
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.task              = task
        self.trees             = []

    def fit(self, X, y):
        X = X.astype(float)
        y = y.astype(int) if self.task == "classification" else y.astype(float)
        n_samples, n_features = X.shape
        mf = self.max_features or max(1, int(np.sqrt(n_features)))
        self.trees = []
        for _ in range(self.n_trees):
            idx  = np.random.choice(n_samples, n_samples, replace=True)
            fidx = np.random.choice(n_features, mf, replace=False)
            tree = DecisionTree(self.max_depth, self.min_samples_split, self.task)
            tree.fit(X[np.ix_(idx, fidx)], y[idx])
            self.trees.append((tree, fidx))

    def predict(self, X):
        X = X.astype(float)
        preds = np.array([t.predict(X[:, fi]) for t, fi in self.trees])
        if self.task == "classification":
            return np.array([np.bincount(preds[:, i].astype(int)).argmax()
                             for i in range(X.shape[0])])
        return preds.mean(axis=0)

    @property
    def feature_importances_(self):
        if not self.trees:
            return None
        # average importances across trees (mapped back to full feature space)
        n_feat = max(max(fi) for _, fi in self.trees) + 1
        imp    = np.zeros(n_feat)
        for tree, fi in self.trees:
            if tree.feature_importances_ is not None:
                for local_i, global_i in enumerate(fi):
                    if local_i < len(tree.feature_importances_):
                        imp[global_i] += tree.feature_importances_[local_i]
        s = imp.sum()
        return imp / s if s > 0 else imp


# ─────────────────────────────────────────────────────────
# K-Nearest Neighbors
# ─────────────────────────────────────────────────────────
class KNearestNeighbors:
    def __init__(self, k=3, metric="euclidean", task="classification"):
        self.k      = k
        self.metric = metric
        self.task   = task

    def fit(self, X, y):
        self.X_train = X.astype(float)
        self.y_train = y.astype(int) if self.task == "classification" else y.astype(float)
        return self

    def _dist(self, x):
        if self.metric == "manhattan":
            return np.sum(np.abs(self.X_train - x), axis=1)
        return np.sqrt(np.sum((self.X_train - x)**2, axis=1))

    def _predict_one(self, x):
        k_labels = self.y_train[np.argsort(self._dist(x))[:self.k]]
        if self.task == "classification":
            vals, counts = np.unique(k_labels, return_counts=True)
            return vals[np.argmax(counts)]
        return np.mean(k_labels)

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X.astype(float)])

    def score(self, X, y):
        y_pred = self.predict(X)
        if self.task == "classification":
            return np.mean(y_pred == y.astype(int))
        return 1 - ((y - y_pred)**2).sum() / ((y - y.mean())**2).sum()