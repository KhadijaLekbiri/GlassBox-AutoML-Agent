import random
import numpy as np
from Optimization.helpers import evaluate_model
def sample_params(param_grid):
    return {k: random.choice(v) for k, v in param_grid.items()}
class RandomSearch:
    def __init__(self, model_class, param_grid, kf, metric=None, n_iter=10):
        self.model_class = model_class
        self.param_grid = param_grid
        self.kf = kf
        self.metric = metric
        self.n_iter = n_iter
        self.best_score  = -np.inf
        self.best_params = {}
        self.best_estimator_ = None
    def fit(self, X, y):
        for _ in range(self.n_iter):
            params = sample_params(self.param_grid)
            model  = self.model_class(**params)
            score  = evaluate_model(model, X, y, self.kf, self.metric)
            if score > self.best_score:
                self.best_score      = score
                self.best_params     = params
                self.best_estimator_ = self.model_class(**params)
        if self.best_estimator_ is not None:
            self.best_estimator_.fit(X, y)
        return self
