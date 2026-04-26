"""
sklearn_benchmark.py
─────────────────────────────────────────────────────────────────────────────
Scikit-learn Benchmark — Titanic Dataset

Trains the same 5 models as GlassBox using scikit-learn, saves results to
sklearn_benchmark_results.json, then compares with GlassBox results.

Usage:
    pip install scikit-learn pandas numpy
    python sklearn_benchmark.py

Output:
    sklearn_benchmark_results.json  — sklearn results
    comparison_report.json          — side-by-side comparison
─────────────────────────────────────────────────────────────────────────────
"""

import json, time, os, sys
import numpy as np

# ── sklearn imports ───────────────────────────────────────────────────────────
try:
    import pandas as pd
    from sklearn.tree          import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble      import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model  import LogisticRegression, LinearRegression
    from sklearn.naive_bayes   import GaussianNB
    from sklearn.neighbors     import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute        import SimpleImputer
    from sklearn.metrics       import f1_score, precision_score, recall_score
except ImportError:
    print("ERROR: Install dependencies first:")
    print("  pip install scikit-learn pandas numpy")
    sys.exit(1)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(PROJECT_DIR, "agent", "titanic_dataset.csv")
OUT_SKLEARN = os.path.join(PROJECT_DIR, "sklearn_benchmark_results.json")
OUT_COMPARE = os.path.join(PROJECT_DIR, "comparison_report.json")
GLASSBOX_REPORT = os.path.join(PROJECT_DIR, "agent", "titanic_dataset_glassbox_report.json")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df, target_col):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Encode ALL columns that are not numeric
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors="raise")
        except (ValueError, TypeError):
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.astype(float)

    # Impute + scale
    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    return X, y.values


def run_classification(X, y, cv_folds=5):
    """Train 5 classifiers, return per-model CV results."""
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    models = {
        "DecisionTree":       DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42),
        "RandomForest":       RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "GaussianNB":         GaussianNB(),
        "KNN":                KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    for name, model in models.items():
        t0  = time.time()
        acc = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
        f1  = cross_val_score(model, X, y, cv=kf, scoring="f1_weighted")
        pre = cross_val_score(model, X, y, cv=kf, scoring="precision_weighted")
        rec = cross_val_score(model, X, y, cv=kf, scoring="recall_weighted")
        elapsed = time.time() - t0

        results[name] = {
            "accuracy":  round(float(acc.mean()), 4),
            "f1":        round(float(f1.mean()),  4),
            "precision": round(float(pre.mean()), 4),
            "recall":    round(float(rec.mean()), 4),
            "time_s":    round(elapsed, 2),
        }
        print(f"  {name:<22} accuracy={results[name]['accuracy']:.4f}  "
              f"f1={results[name]['f1']:.4f}  time={elapsed:.2f}s")

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    return results, best_name


def run_regression(X, y, cv_folds=5):
    """Train 4 regressors, return per-model CV results."""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree":     DecisionTreeRegressor(max_depth=5, min_samples_split=5, random_state=42),
        "RandomForest":     RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "KNN":              KNeighborsRegressor(n_neighbors=5),
    }

    results = {}
    for name, model in models.items():
        t0  = time.time()
        r2  = cross_val_score(model, X, y, cv=kf, scoring="r2")
        mae = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
        elapsed = time.time() - t0

        results[name] = {
            "r2":     round(float(r2.mean()),   4),
            "mae":    round(float(-mae.mean()), 4),
            "time_s": round(elapsed, 2),
        }
        print(f"  {name:<22} r2={results[name]['r2']:.4f}  "
              f"mae={results[name]['mae']:.4f}  time={elapsed:.2f}s")

    best_name = max(results, key=lambda k: results[k]["r2"])
    return results, best_name


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SKLEARN BENCHMARK — Titanic Dataset")
    print("  Trains the same 5 models as GlassBox using scikit-learn")
    print("=" * 65)

    df = pd.read_csv(CSV_PATH)
    print(f"\n  Dataset: {CSV_PATH}")
    print(f"  Shape:   {df.shape[0]} rows x {df.shape[1]} columns\n")

    # ── Classification: predict Survived ─────────────────────────────────────
    print("━" * 65)
    print("TASK 1 — Classification: predict 'Survived'")
    print("━" * 65)
    X_clf, y_clf = preprocess(df, "Survived")
    t_clf = time.time()
    clf_results, clf_best = run_classification(X_clf, y_clf, cv_folds=5)
    clf_total = round(time.time() - t_clf, 2)
    print(f"\n  ✓ Best model: {clf_best} "
          f"(accuracy={clf_results[clf_best]['accuracy']:.4f})")
    print(f"  Total time: {clf_total}s")

    # ── Regression: predict Fare ──────────────────────────────────────────────
    print()
    print("━" * 65)
    print("TASK 2 — Regression: predict 'Fare'")
    print("━" * 65)
    X_reg, y_reg = preprocess(df, "Fare")
    t_reg = time.time()
    reg_results, reg_best = run_regression(X_reg, y_reg.astype(float), cv_folds=5)
    reg_total = round(time.time() - t_reg, 2)
    print(f"\n  ✓ Best model: {reg_best} "
          f"(r2={reg_results[reg_best]['r2']:.4f})")
    print(f"  Total time: {reg_total}s")

    # ── Save sklearn results ──────────────────────────────────────────────────
    sklearn_report = {
        "library":    "scikit-learn",
        "dataset":    "titanic_dataset.csv",
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "classification": {
            "target":      "Survived",
            "cv_folds":    5,
            "models":      clf_results,
            "best_model":  clf_best,
            "best_metrics": clf_results[clf_best],
            "total_time_s": clf_total,
        },
        "regression": {
            "target":      "Fare",
            "cv_folds":    5,
            "models":      reg_results,
            "best_model":  reg_best,
            "best_metrics": reg_results[reg_best],
            "total_time_s": reg_total,
        }
    }

    with open(OUT_SKLEARN, "w") as f:
        json.dump(sklearn_report, f, indent=2)
    print(f"\n  ✓ Sklearn results saved → {OUT_SKLEARN}")

    # ── Load GlassBox results and compare ─────────────────────────────────────
    print()
    print("━" * 65)
    print("COMPARISON — GlassBox (scratch) vs Scikit-learn")
    print("━" * 65)

    # Run GlassBox now
    sys.path.insert(0, PROJECT_DIR)
    os.chdir(PROJECT_DIR)
    from agent.autofit import autofit

    print("\n  Running GlassBox pipeline (classification)...")
    t0 = time.time()
    gb_clf = autofit(CSV_PATH, target_col="Survived",
                     task_type="classification", time_budget=60, cv_folds=5)
    gb_clf_time = round(time.time() - t0, 2)

    print("  Running GlassBox pipeline (regression)...")
    t0 = time.time()
    gb_reg = autofit(CSV_PATH, target_col="Fare",
                     task_type="regression", time_budget=60, cv_folds=5)
    gb_reg_time = round(time.time() - t0, 2)

    # Build comparison
    comparison = {
        "dataset": "titanic_dataset.csv",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "classification": {
            "target": "Survived",
            "glassbox": {
                "best_model":  gb_clf["best_model"],
                "accuracy":    gb_clf["metrics"].get("accuracy", 0),
                "f1":          gb_clf["metrics"].get("f1", 0),
                "precision":   gb_clf["metrics"].get("precision", 0),
                "recall":      gb_clf["metrics"].get("recall", 0),
                "time_s":      gb_clf_time,
            },
            "sklearn": {
                "best_model":  clf_best,
                "accuracy":    clf_results[clf_best]["accuracy"],
                "f1":          clf_results[clf_best]["f1"],
                "precision":   clf_results[clf_best]["precision"],
                "recall":      clf_results[clf_best]["recall"],
                "time_s":      clf_total,
            },
            "accuracy_diff": round(
                gb_clf["metrics"].get("accuracy", 0) - clf_results[clf_best]["accuracy"], 4
            ),
        },
        "regression": {
            "target": "Fare",
            "glassbox": {
                "best_model": gb_reg["best_model"],
                "r2":         gb_reg["metrics"].get("r2", 0),
                "mae":        gb_reg["metrics"].get("mae", 0),
                "time_s":     gb_reg_time,
            },
            "sklearn": {
                "best_model": reg_best,
                "r2":         reg_results[reg_best]["r2"],
                "mae":        reg_results[reg_best]["mae"],
                "time_s":     reg_total,
            },
            "r2_diff": round(
                gb_reg["metrics"].get("r2", 0) - reg_results[reg_best]["r2"], 4
            ),
        }
    }

    with open(OUT_COMPARE, "w") as f:
        json.dump(comparison, f, indent=2)

    # Print comparison table
    c = comparison["classification"]
    r = comparison["regression"]

    print()
    print(f"  {'':30} {'GlassBox':>12} {'Sklearn':>12} {'Diff':>8}")
    print(f"  {'-'*62}")
    print(f"  {'CLASSIFICATION (Survived)':30}")
    print(f"  {'  Best Model':30} {c['glassbox']['best_model']:>12} {c['sklearn']['best_model']:>12}")
    print(f"  {'  Accuracy':30} {c['glassbox']['accuracy']:>12.4f} {c['sklearn']['accuracy']:>12.4f} {c['accuracy_diff']:>+8.4f}")
    print(f"  {'  F1 Score':30} {c['glassbox']['f1']:>12.4f} {c['sklearn']['f1']:>12.4f}")
    print(f"  {'  Time (s)':30} {c['glassbox']['time_s']:>12.1f} {c['sklearn']['time_s']:>12.1f}")
    print(f"  {'-'*62}")
    print(f"  {'REGRESSION (Fare)':30}")
    print(f"  {'  Best Model':30} {r['glassbox']['best_model']:>12} {r['sklearn']['best_model']:>12}")
    print(f"  {'  R2 Score':30} {r['glassbox']['r2']:>12.4f} {r['sklearn']['r2']:>12.4f} {r['r2_diff']:>+8.4f}")
    print(f"  {'  MAE':30} {r['glassbox']['mae']:>12.4f} {r['sklearn']['mae']:>12.4f}")
    print(f"  {'  Time (s)':30} {r['glassbox']['time_s']:>12.1f} {r['sklearn']['time_s']:>12.1f}")
    print(f"  {'-'*62}")

    print(f"\n  ✓ Comparison saved → {OUT_COMPARE}")
    print()
    print("=" * 65)
    print("  BENCHMARK COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
