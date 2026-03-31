import random
from helpers import evaluate_model 
def sample_params(param_grid):
    params = {}
    for key, values in param_grid.items():
        params[key] = random.choice(values)
    return params

class RandomSearch:
    def __init__(self, model_class, param_grid, kf, metric, n_iter=10):
        self.model_class = model_class
        self.param_grid = param_grid
        self.kf = kf
        self.metric = metric
        self.n_iter = n_iter
    def fit(self, X, y):
        best_score = -np.inf
        best_params = None
        for _ in range(self.n_iter):
            params = sample_params(self.param_grid)
            model = self.model_class(**params)
            score = evaluate_model(model, X, y, self.kf, self.metric)
            if score > best_score:
                best_score = score
                best_params = params
        self.best_score = best_score
        self.best_params = best_params 