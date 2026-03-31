from kfold import KFold
from grid_search import GridSearch
from random_search import RandomSearch
kf = KFold(n_splits=5)
grid = GridSearch(
    model_class=DecisionTree,
    param_grid={
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5]
    },
    kf=kf,
    metric=accuracy
)

grid.fit(X, y)

print(grid.best_params)
print(grid.best_score)