import numpy as np


class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0

    
    def fit(self,X,y):
        n, features = X.shape
        self.w = np.zeros(features)
        self.b = 0

        for epoch in range(self.epochs):
            y_pred = X@self.w + self.b
            error = y_pred - y

            dw = 2/n* X.T @error
            db = 2/n*np.sum(error)

            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self, X):
        y_pred = X @ self.w + self.b
        return y_pred
    
    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return 1 - ((y - y_pred)** 2).sum() / ((y - y.mean()) ** 2).sum()


class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr # learning rate
        self.epochs = epochs
        self.w = None
        self.b = 0
    
    def sigmoid(self, z):
        return 1/(1+ np.exp(-z))
    
    def fit(self,X,y):
        n, features = X.shape
        self.w = np.zeros(features)
        self.b = 0

        for epoch in range(self.epochs):
            y_pred = self.sigmoid(X@self.w + self.b)
            error = y_pred - y

            dw = 1/n* X.T @error
            db = 1/n*np.sum(error)

            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self, X):
        y_pred = self.sigmoid(X @ self.w + self.b)
        return (y_pred >= 0.5).astype(int)
    
    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class GaussianNB:
    def __init__(self, prior = None,var_smoothing=1e-09):
        self.prior = None
        self.classes = None
        self.mean = None
        self.var = None
        self.var_smoothing = var_smoothing

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes =  len(self.classes)

        self.prior = np.zeros(n_classes)
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y==c]
            self.prior[idx] = len(X_c)/ len(X)
            self.mean[idx, :] = np.mean(X_c, axis=0)
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
            return np.mean(k_labels)
        
    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)

        if self.task == "classification":
            return np.mean(y_pred == y)                      # accuracy
        else:
            return 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()  # R²