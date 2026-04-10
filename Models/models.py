from core.stats import mean
from core.utils import sigmoid
import numpy as np

# ─────────────────────────────────────────────────────────
# Linear Regression
# ─────────────────────────────────────────────────────────

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0

    
    def fit(self,X,y):
        X, y = X.astype(float), y.astype(float)
        n, features = X.shape
        self.w = np.zeros(features)
        self.b = 0

        for epoch in range(self.epochs):
            y_pred = X @ self.w + self.b
            error = y_pred - y

            dw = 2/n* X.T @error
            db = 2/n*np.sum(error)

            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self, X):
        return X.astype(float) @ self.w + self.b

    @property
    def feature_importances_(self):
        if self.w is None: return None
        a = np.abs(self.w)
        return a / (a.sum() + 1e-10)
        
    def score(self, X, y, sample_weight=None):
        y, y_pred = y.astype(float), self.predict(X)
        return 1 - ((y - y_pred)** 2).sum() / ((y - y.mean()) ** 2).sum()

# ─────────────────────────────────────────────────────────
# Logistic Regression
# ─────────────────────────────────────────────────────────

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr # learning rate
        self.epochs = epochs
        self.w = None
        self.b = 0
    
    
    def fit(self,X,y):
        n, features = X.shape
        self.w = np.zeros(features)
        self.b = 0

        for epoch in range(self.epochs):
            y_pred = sigmoid(X@self.w + self.b)
            error = y_pred - y

            dw = 1/n* X.T @error
            db = 1/n*np.sum(error)

            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self, X):
        y_pred = sigmoid(X @ self.w + self.b)
        return (y_pred >= 0.5).astype(int)
        
    @property
    def feature_importances_(self):
        if self.w is None: return None
        a = np.abs(self.w)
        return a / (a.sum() + 1e-10)
        
    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return mean(y_pred == y)

# ─────────────────────────────────────────────────────────
# Gaussian Naive Bayes
# ─────────────────────────────────────────────────────────

class GaussianNB:
    def __init__(self, prior = None,var_smoothing=1e-09):
        self.prior = None
        self.classes = None
        self.mean = None
        self.var = None
        self.var_smoothing = var_smoothing

    def fit(self,X,y):
        X, y = X.astype(float), y.astype(int)
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes =  len(self.classes)

        self.prior = np.zeros(n_classes)
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y==c]
            self.prior[idx] = len(X_c)/ len(X)
            self.mean[idx, :] = mean(X_c, axis=0)
            self.var[idx, :] = np.var(X_c, axis=0) + self.var_smoothing 
        
    def _log_gaussian(self, x, mean, var):
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)

    def predict(self, X):
        predictions = []

        for x in X:
            log_probs = []

            for idx, c in enumerate(self.classes):
                log_prob = np.log(self.prior[idx])
                log_prob += np.sum(self._log_gaussian(x, self.mean[idx], self.var[idx]))
                log_probs.append(log_prob)

            predictions.append(self.classes[np.argmax(log_probs)])

        return np.array(predictions)

# ─────────────────────────────────────────────────────────
# Decision Tree  (classification AND regression)
# ─────────────────────────────────────────────────────────
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.task = None
        self._importances = None

    def fit(self, X, y):
        X = X.astype(float)
        y = y.astype(int) if self.task == "classification" else y.astype(float)
        self._n_features  = X.shape[1]
        self._importances = np.zeros(self._n_features)
        self.tree = self._build_tree(X, y, 0)
        s = self._importances.sum()
        if s > 0:
            self._importances /= s
    
    def predict(self, X):
        return np.array([self._predict_row(row, self.tree) for row in X.astype(float)])

    @property
    def feature_importances_(self):
        return self._importances
        
    # ----------------  METHODS ----------------
    
    def _gini(self, y):
        if self.task == "classification":
            counts = np.bincount(y.astype(int))
            p = counts / len(y)
            return 1 - np.sum(p**2)       
        else:
            return np.var(y)

    def _leaf(self, y):
        if self.task == "classification":
            return int(np.bincount(y.astype(int)).argmax())
        return float(np.mean(y))
        
    def _best_split(self, X, y):
        best_gini = 1
        best_idx, best_val = None, None
        
        for feature_idx in range(X.shape[1]):
            values = np.unique(X[:, feature_idx])
            for val in values:
                left_mask = X[:, feature_idx] <= val
                right_mask = X[:, feature_idx] > val
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                gini_total = (len(y[left_mask]) * gini_left + len(y[right_mask]) * gini_right) / len(y)
                
                if gini_total < best_gini:
                    best_gini = gini_total
                    best_idx = feature_idx
                    best_val = val
        return best_idx, best_val
    
    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return self._leaf(y)
        
        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            return self._leaf(y)
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold

        gain = self._gini(y) - (
            left_mask.sum() * self._gini(y[left_mask]) +
            right_mask.sum() * self._gini(y[right_mask])
        ) / len(y)
        self._importances[feature_idx] += gain * len(y)
        
        return {
            'feature': feature_idx,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }
    
    def _predict_row(self, row, node):
        if isinstance(node, dict):
            if row[node['feature']] <= node['threshold']:
                return self._predict_row(row, node['left'])
            else:
                return self._predict_row(row, node['right'])
        return node

# ─────────────────────────────────────────────────────────
# Random Forest  (classification AND regression)
# ─────────────────────────────────────────────────────────

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2, max_features=None, task="classification"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.task = task
        self.trees = []
    
    def fit(self, X, y):
        X = X.astype(float)
        y = y.astype(int) if self.task == "classification" else y.astype(float)
        n_samples, n_features = X.shape
        self.max_features = self.max_features or int(np.sqrt(n_features))
        self.trees = []
        
        for _ in range(self.n_trees):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Random feature subset
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, feature_indices], y_sample)
            
            self.trees.append((tree, feature_indices))
    
    def predict(self, X):
        X = X.astype(float)
        # Collect all tree predictions
        tree_preds = np.array([tree.predict(X[:, feat_idx]) for tree, feat_idx in self.trees])
        # Majority vote
        y_pred = []
        for i in range(X.shape[0]):
            counts = np.bincount(tree_preds[:, i])
            y_pred.append(np.argmax(counts))
        return np.array(y_pred)

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
        if metric not in ("euclidean", "manhattan"):
            raise ValueError(f"metric must be 'euclidean' or 'manhattan', got '{metric}'")
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")
        self.k = k
        self.metric = metric
        self.task = task
        self.X_train = None
        self.y_train = None

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def _euclidean(self, x):
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def _manhattan(self, x):
        return np.sum(np.abs(self.X_train - x), axis=1)

    def _distances(self, x):
        if self.metric == "euclidean":
            return self._euclidean(x)
        return self._manhattan(x)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _predict_one(self, x):
        dists = self._distances(x)
        k_indices = np.argsort(dists)[: self.k]
        k_labels = self.y_train[k_indices]

        if self.task == "classification":
            labels, counts = np.unique(k_labels, return_counts=True)
            return labels[np.argmax(counts)]
        else:
            return mean(k_labels)
        
    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)

        if self.task == "classification":
            return mean(y_pred == y)                      # accuracy
        else:
            return 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()  # R²
