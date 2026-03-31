import itertools
from helpers import evaluate_model 
def generate_combinations(param_grid):
    keys = param_grid.keys()
    values = param_grid.values()
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

class GridSearch:
    def __init__(self, model_class, param_grid, kf, metric):
        self.model_class = model_class
        self.param_grid = param_grid
        self.kf = kf
        self.metric = metric
    def fit(self, X, y):
        best_score = -np.inf
        best_params = None
        for params in generate_combinations(self.param_grid):
            model = self.model_class(**params)
            score = evaluate_model(model, X, y, self.kf, self.metric)
            if score > best_score:
                best_score = score
                best_params = params
        self.best_score = best_score
        self.best_params = best_params
