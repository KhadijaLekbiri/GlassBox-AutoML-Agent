import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

glassbox = load_json("glassbox_report.json")
sklearn  = load_json("sklearn_report.json")

def compare_reports(gb, sk):
    comparison = {}

    # Model names
    comparison["models"] = {
        "glassbox": gb["best_model_name"],
        "sklearn":  sk["best_model_name"]
    }

    # Time comparison
    comparison["time_seconds"] = {
        "glassbox": gb.get("elapsed_seconds"),
        "sklearn":  sk.get("elapsed_seconds")
    }

    # Metrics comparison
    comparison["metrics"] = {}
    for metric in gb["metrics"]:
        if metric in sk["metrics"]:
            comparison["metrics"][metric] = {
                "glassbox": gb["metrics"][metric],
                "sklearn":  sk["metrics"][metric],
                "difference": gb["metrics"][metric] - sk["metrics"][metric]
            }

    # Winner (simple logic)
    primary_metric = list(gb["metrics"].keys())[0]
    if gb["metrics"][primary_metric] > sk["metrics"][primary_metric]:
        comparison["winner"] = "GlassBox"
    else:
        comparison["winner"] = "Scikit-learn"

    return comparison

comparison = compare_reports(glassbox, sklearn)

with open("comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=4)
