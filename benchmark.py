"""
benchmark.py  — run from repo root:  python benchmark.py
Tests the full pipeline on synthetic data.
"""
import numpy as np
import time, json, os, csv

def make_clf_csv(path, n=200, seed=42):
    rng = np.random.default_rng(seed)
    age    = rng.integers(18, 70, n).astype(str)
    income = rng.uniform(20000, 100000, n).round(0).astype(int).astype(str)
    edu    = rng.choice(["low","medium","high"], n)
    target = ((rng.integers(18,70,n)>40) & (rng.uniform(20000,100000,n)>50000)).astype(int).astype(str)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Age","Income","Education","Target"])
        for row in zip(age, income, edu, target):
            w.writerow(row)

def make_reg_csv(path, n=200, seed=42):
    rng   = np.random.default_rng(seed)
    area  = rng.uniform(40, 250, n).round(1).astype(str)
    rooms = rng.integers(1, 6, n).astype(str)
    age   = rng.integers(0, 40, n).astype(str)
    price = (rng.uniform(40,250,n)*1500 + rng.integers(1,6,n)*15000).round(0).astype(int).astype(str)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Area","Rooms","Age","Price"])
        for row in zip(area, rooms, age, price):
            w.writerow(row)

def run():
    from agent.autofit import autofit
    from agent.report  import report_to_explanation

    print("="*55)
    print("  GlassBox-AutoML — Phase 5 Benchmark")
    print("="*55)

    # ── Classification ────────────────────────────────────────
    print("\n── Classification ───────────────────────────────────")
    make_clf_csv("tmp_clf.csv")
    t0 = time.time()
    r  = autofit("tmp_clf.csv", target_col="Target",
                 task_type="classification", time_budget=60, cv_folds=3)
    elapsed = time.time()-t0
    print(f"  Best model : {r['best_model']}")
    print(f"  Accuracy   : {r['metrics'].get('accuracy',0):.4f}")
    print(f"  F1         : {r['metrics'].get('f1',0):.4f}")
    print(f"  Time       : {elapsed:.1f}s")
    assert r["benchmark_pass"], "Pipeline exceeded 120s!"
    print(f"\n  Explanation:\n  {report_to_explanation(r)}")

    # ── Regression ───────────────────────────────────────────
    print("\n── Regression ───────────────────────────────────────")
    make_reg_csv("tmp_reg.csv")
    t0 = time.time()
    r  = autofit("tmp_reg.csv", target_col="Price",
                 task_type="regression", time_budget=60, cv_folds=3)
    elapsed = time.time()-t0
    print(f"  Best model : {r['best_model']}")
    print(f"  MAE        : {r['metrics'].get('mae',0):.4f}")
    print(f"  R2         : {r['metrics'].get('r2',0):.4f}")
    print(f"  Time       : {elapsed:.1f}s")
    assert r["benchmark_pass"], "Pipeline exceeded 120s!"

    # ── JSON check ───────────────────────────────────────────
    s = json.dumps(r)
    assert json.loads(s)["best_model"] == r["best_model"]
    print("\n  JSON serialisation OK")

    # cleanup
    os.remove("tmp_clf.csv")
    os.remove("tmp_reg.csv")

    print("\n" + "="*55)
    print("  ALL BENCHMARKS PASSED ✓")
    print("="*55)

if __name__ == "__main__":
    run()
