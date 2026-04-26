"""
agent/autofit.py — master pipeline connecting all GlassBox modules.
"""
import sys
import time
import json

# All progress goes to stderr — stdout is reserved for MCP JSON
def _log(msg):
    print(msg, file=sys.stderr, flush=True)
import numpy as np

from core.utils     import detect_task
from core.metrics   import accuracy, f1_score, mse, mae, precision, r2_score, recall

from eda.inspector               import DataInspector
from Preprocessing.preprocessor  import Preprocessor
from Models.models               import (LinearRegression, LogisticRegression,
                                          DecisionTree, RandomForest,
                                          KNearestNeighbors, GaussianNB)
from Optimization.kfold          import KFold
from Optimization.grid_search    import GridSearch
from Optimization.random_search  import RandomSearch
from agent.report                import build_report

# ── Hyperparameter grids ──────────────────────────────────────────────────────
GRIDS = {
    "DecisionTree_clf":   {"max_depth": [3, 5], "min_samples_split": [2, 5], "task": ["classification"]},
    "DecisionTree_reg":   {"max_depth": [3, 5], "min_samples_split": [2, 5], "task": ["regression"]},
    "RandomForest_clf":   {"n_trees": [10, 30], "max_depth": [3, 5], "task": ["classification"]},
    "RandomForest_reg":   {"n_trees": [10, 30], "max_depth": [3, 5], "task": ["regression"]},
    "LogisticRegression": {"lr": [0.01, 0.1], "epochs": [200]},
    "LinearRegression":   {"lr": [0.01, 0.1], "epochs": [200]},
    "KNN_clf":            {"k": [3, 5], "metric": ["euclidean"], "task": ["classification"]},
    "KNN_reg":            {"k": [3, 5], "metric": ["euclidean"], "task": ["regression"]},
    "GaussianNB":         {"var_smoothing": [1e-9, 1e-7]},
}

CLF_MODELS = {
    "LogisticRegression": LogisticRegression,
    "DecisionTree_clf":   DecisionTree,
    "RandomForest_clf":   RandomForest,
    "KNN_clf":            KNearestNeighbors,
    "GaussianNB":         GaussianNB,
}

REG_MODELS = {
    "LinearRegression": LinearRegression,
    "DecisionTree_reg": DecisionTree,
    "RandomForest_reg": RandomForest,
    "KNN_reg":          KNearestNeighbors,
}

# ── CSV loader (pure NumPy) ───────────────────────────────────────────────────
def _load_csv(csv_path, target_col):
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    header = lines[0].split(",")
    rows   = [l.split(",") for l in lines[1:]]
    data   = np.array(rows, dtype=object)
    if target_col not in header:
        raise ValueError(f"Column '{target_col}' not found. Available: {header}")
    t_idx         = header.index(target_col)
    feature_idx   = [i for i in range(len(header)) if i != t_idx]
    feature_names = [header[i] for i in feature_idx]
    return data[:, feature_idx], data[:, t_idx], feature_names

def _to_numeric(X):
    """Best-effort conversion to float for EDA (non-numeric → NaN)."""
    X_out = np.full(X.shape, np.nan, dtype=float)
    for i in range(X.shape[1]):
        for j, v in enumerate(X[:, i]):
            try:
                X_out[j, i] = float(v)
            except (ValueError, TypeError):
                pass
    return X_out

# ── Feature importance ────────────────────────────────────────────────────────
def _feature_importance(model, feature_names):
    n = len(feature_names)
    fi = getattr(model, "feature_importances_", None)
    if fi is not None and len(fi) == n:
        scores = np.array(fi, dtype=float)
    elif hasattr(model, "w") and model.w is not None and len(model.w) == n:
        scores = np.abs(model.w)
    else:
        scores = np.ones(n)
    scores = scores / (scores.sum() + 1e-10)
    return sorted(zip(feature_names, scores.tolist()), key=lambda x: -x[1])

# ── Evaluation ────────────────────────────────────────────────────────────────
def _evaluate(model, X, y, kf, task):
    fold_results = []
    for train_idx, test_idx in kf.get_splits(X):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        y_t   = y[test_idx]
        if task == "classification":
            fold_results.append({
                "accuracy": float(accuracy(y_t.astype(int), preds.astype(int))),
                "f1":       float(f1_score(y_t.astype(int), preds.astype(int))),
                "precision": float(precision(y_t.astype(int), preds.astype(int))),
                "recall":    float(recall(y_t.astype(int), preds.astype(int)))
            })
        else:
            fold_results.append({
                "mae": float(mae(y_t.astype(float), preds.astype(float))),
                "mse": float(mse(y_t.astype(float), preds.astype(float))),
                "r2":  float(r2_score(y_t.astype(float), preds.astype(float))),
            })
    keys = fold_results[0].keys()
    return {k: float(np.mean([f[k] for f in fold_results])) for k in keys}

# ── Main pipeline ─────────────────────────────────────────────────────────────
def autofit(csv_path, target_col, task_type="auto",
            time_budget=60, cv_folds=5, use_random_search=False):
    wall_start = time.time()

    _log("[GlassBox] Loading data...")
    X_raw, y_raw, feature_names = _load_csv(csv_path, target_col)

    _log("[GlassBox] Running EDA...")
    inspector   = DataInspector()
    eda_profile = inspector.fit(_to_numeric(X_raw), feature_names)

    _log("[GlassBox] Preprocessing...")
    prep = Preprocessor(scaler="standard", imputer_strategy="mean")
    X, y, feat_names_out = prep.fit_transform(X_raw, y_raw, feature_names)

    task = task_type if task_type != "auto" else detect_task(y)
    _log(f"[GlassBox] Task: {task}")
    y = y.astype(int) if task == "classification" else y.astype(float)

    _log("[GlassBox] Searching best model...")
    models       = CLF_MODELS if task == "classification" else REG_MODELS
    kf           = KFold(n_splits=cv_folds)
    best_model   = None
    best_name    = None
    best_params  = {}
    best_score   = -np.inf
    search_start = time.time()

    for name, model_cls in models.items():
        if time.time() - search_start > time_budget:
            _log("[GlassBox] Time budget reached.")
            break
        grid      = GRIDS.get(name, {})
        SearchCls = RandomSearch if use_random_search else GridSearch
        try:
            searcher = SearchCls(model_class=model_cls, param_grid=grid,
                                 kf=kf, metric=None)
            searcher.fit(X, y)
        except Exception as e:
            _log(f"  [skip] {name}: {e}")
            continue
        if searcher.best_score > best_score:
            best_score  = searcher.best_score
            best_name   = name
            best_params = searcher.best_params
            best_model  = searcher.best_estimator_

    if best_model is None:
        raise RuntimeError("All models failed. Check your data.")

    _log(f"[GlassBox] Evaluating {best_name}...")
    metrics      = _evaluate(best_model, X, y, KFold(n_splits=cv_folds), task)
    top_features = _feature_importance(best_model, feat_names_out)

    elapsed = time.time() - wall_start
    report  = build_report(
        best_model_name = best_name,
        best_params     = {k: v for k, v in best_params.items() if k != "task"},
        metrics         = metrics,
        top_features    = top_features[:10],
        eda_summary     = eda_profile,
        task_type       = task,
        elapsed_seconds = elapsed,
    )
    _log(f"[GlassBox] Done in {elapsed:.1f}s — best: {best_name} | score: {best_score:.4f}")
    return report