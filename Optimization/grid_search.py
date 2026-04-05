import itertools
import numpy as np
from Optimization.helpers import evaluate_model

def generate_combinations(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

class GridSearch:
    def __init__(self, model_class, param_grid, kf, metric=None):
        self.model_class = model_class
        self.param_grid = param_grid
        self.kf = kf
        self.metric = metric
        self.best_score  = -np.inf
        self.best_params = {}
        self.best_estimator_ = None
    def fit(self, X, y):
        for params in generate_combinations(self.param_grid):
            model = self.model_class(**params)
            score = evaluate_model(model, X, y, self.kf, self.metric)
            if score > self.best_score:
                self.best_score      = score
                self.best_params     = params
                self.best_estimator_ = self.model_class(**params)
        # refit best on full data
        if self.best_estimator_ is not None:
            self.best_estimator_.fit(X, y)
        return self
