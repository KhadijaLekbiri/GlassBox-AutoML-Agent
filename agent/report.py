"""
report.py
---------
Builds the structured JSON report that GlassBox returns to the IronClaw agent.
The agent uses this JSON to produce natural-language explanations to the user.
"""

import json
import numpy as np


def _safe(val):
    """Convert numpy scalars to native Python types for JSON serialisation."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return round(float(val), 6)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def build_report(
    best_model_name: str,
    best_params: dict,
    metrics: dict,
    top_features: list,
    eda_summary: dict,
    task_type: str = "classification",
    elapsed_seconds: float = 0.0,
) -> dict:
    """
    Build the final JSON report.

    Parameters
    ----------
    best_model_name  : e.g. "RandomForest"
    best_params      : e.g. {"depth": 5, "n_trees": 100}
    metrics          : dict of metric_name -> value
    top_features     : list of (feature_name, importance_score) tuples
    eda_summary      : output from Inspector (outliers, missing values, dtypes…)
    task_type        : "classification" or "regression"
    elapsed_seconds  : total wall-clock time for the pipeline

    Returns
    -------
    A plain Python dict (JSON-serialisable).
    """
    report = {
        "status": "success",
        "task_type": task_type,
        "best_model": best_model_name,
        "best_params": {k: _safe(v) for k, v in best_params.items()},
        "metrics": {k: _safe(v) for k, v in metrics.items()},
        "top_features": [
            {"feature": name, "importance": round(float(score), 4)}
            for name, score in top_features
        ],
        "eda_summary": {
            "n_rows": _safe(eda_summary.get("n_rows", 0)),
            "n_cols": _safe(eda_summary.get("n_cols", 0)),
            "outliers_flagged": _safe(eda_summary.get("outliers_flagged", 0)),
            "missing_filled": _safe(eda_summary.get("missing_filled", 0)),
            "numeric_cols": eda_summary.get("numeric_cols", []),
            "categorical_cols": eda_summary.get("categorical_cols", []),
        },
        "pipeline_seconds": round(elapsed_seconds, 2),
        "benchmark_pass": elapsed_seconds < 120,
    }
    return report


def report_to_json(report: dict, indent: int = 2) -> str:
    """Serialise the report dict to a JSON string."""
    return json.dumps(report, indent=indent)


def report_to_explanation(report: dict) -> str:
    """
    Convert the JSON report into a plain-English explanation suitable for
    the IronClaw agent to present directly to the user.
    """
    m = report["best_model"]
    p = report["best_params"]
    mt = report["metrics"]
    tf = report["top_features"]
    eda = report["eda_summary"]

    lines = [
        f"I analysed your dataset ({eda['n_rows']} rows, {eda['n_cols']} columns).",
    ]

    if eda["missing_filled"]:
        lines.append(f"I filled {eda['missing_filled']} missing values automatically.")
    if eda["outliers_flagged"]:
        lines.append(f"I detected and capped {eda['outliers_flagged']} outliers.")

    lines.append(
        f"After searching, the best model was **{m}** with params {p}."
    )

    if report["task_type"] == "classification":
        acc = mt.get("accuracy", "N/A")
        f1 = mt.get("f1", "N/A")
        recall= mt.get("recall", "N/A")
        precision = mt.get("precision", "N/A")
        lines.append(f"It achieved {acc:.2%} accuracy and an F1-score of {f1:.4f}, precision of {precision:.4f}, and recall of {recall:.4f}.")
    else:
        mae = mt.get("mae", "N/A")
        r2 = mt.get("r2", "N/A")
        lines.append(f"It achieved MAE={mae:.4f} and R²={r2:.4f}.")

    if tf:
        top = tf[0]["feature"]
        lines.append(f"The most important feature was **'{top}'**.")

    lines.append(
        f"Total pipeline time: {report['pipeline_seconds']}s "
        f"({'✓ within' if report['benchmark_pass'] else '✗ exceeded'} the 120 s budget)."
    )

    return " ".join(lines)
