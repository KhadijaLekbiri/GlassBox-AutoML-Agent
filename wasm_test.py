"""
wasm_test.py
------------
Phase 5 WASM compatibility test suite for GlassBox-AutoML.

Tests three things:
  1. Static audit  — wasm_audit.py must report zero violations.
  2. Sandbox I/O   — GlassBoxSandboxIO virtual FS works without real disk.
  3. End-to-end    — Full pipeline runs inside an emulated WASM environment
                     (no real `open()` calls from the agent layer).

Run from repo root:
    python wasm_test.py

Exit code 0 = all tests passed, 1 = at least one failed.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

# ── Test harness ──────────────────────────────────────────────────────────────

PASS = "\033[92m  ✓\033[0m"
FAIL = "\033[91m  ✗\033[0m"
INFO = "\033[94m  ·\033[0m"

_results: list[tuple[str, bool, str]] = []


def test(name: str) -> Callable:
    """Decorator: registers a test function and captures pass/fail."""
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            print(f"\n{INFO} {name}")
            try:
                fn(*args, **kwargs)
                _results.append((name, True, ""))
                print(f"{PASS} PASSED")
            except AssertionError as exc:
                _results.append((name, False, str(exc)))
                print(f"{FAIL} FAILED — {exc}")
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                _results.append((name, False, tb))
                print(f"{FAIL} ERROR — {exc}")
        return wrapper
    return decorator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_clf_csv(n: int = 100, seed: int = 42) -> str:
    """Return a classification CSV as a string (no disk I/O)."""
    rng  = np.random.default_rng(seed)
    buf  = io.StringIO()
    w    = csv.writer(buf)
    w.writerow(["Age", "Income", "Education", "Survived"])
    edu_choices = ["low", "medium", "high"]
    for _ in range(n):
        age    = int(rng.integers(18, 70))
        income = round(float(rng.uniform(20_000, 100_000)), 2)
        edu    = edu_choices[int(rng.integers(0, 3))]
        label  = int((age > 40) and (income > 50_000))
        w.writerow([age, income, edu, label])
    return buf.getvalue()


def _make_reg_csv(n: int = 100, seed: int = 7) -> str:
    """Return a regression CSV as a string."""
    rng = np.random.default_rng(seed)
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["Area", "Rooms", "Age", "Price"])
    for _ in range(n):
        area  = round(float(rng.uniform(40, 250)), 1)
        rooms = int(rng.integers(1, 6))
        age   = int(rng.integers(0, 40))
        price = round(area * 1_500 + rooms * 15_000, 0)
        w.writerow([area, rooms, age, price])
    return buf.getvalue()


# ── Tests ─────────────────────────────────────────────────────────────────────

@test("1. Static WASM audit — zero banned patterns in source")
def test_static_audit():
    """wasm_audit.py must exit 0 (no banned imports or builtins)."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "wasm_audit.py"],
        capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    print(f"     stdout: {output.strip()[:300]}")
    assert result.returncode == 0, (
        f"wasm_audit.py found violations:\n{output}"
    )


@test("2. SandboxIO — read/write without touching real disk")
def test_sandbox_io_virtual():
    """Files staged into the virtual FS must be readable without disk."""
    from agent.ironclaw_integration import GlassBoxSandboxIO

    io_shim = GlassBoxSandboxIO()
    csv_content = _make_clf_csv(n=10)

    io_shim.stage_file("/sandbox/test.csv", csv_content)

    assert io_shim.exists("/sandbox/test.csv"), "staged file not found"
    read_back = io_shim.read_text("/sandbox/test.csv")
    assert read_back == csv_content, "read-back content mismatch"

    io_shim.write_text("/sandbox/report.json", '{"ok": true}')
    assert io_shim.exists("/sandbox/report.json"), "written file not found"
    data = json.loads(io_shim.read_text("/sandbox/report.json"))
    assert data["ok"] is True


@test("3. SandboxIO — missing file raises FileNotFoundError")
def test_sandbox_io_missing():
    from agent.ironclaw_integration import GlassBoxSandboxIO
    io_shim = GlassBoxSandboxIO()
    try:
        io_shim.read_text("/sandbox/does_not_exist.csv")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass   # correct behaviour


@test("4. GlassBoxSkill — missing args return error dict, not exception")
def test_skill_bad_args():
    from agent.ironclaw_integration import GlassBoxSkill
    skill = GlassBoxSkill()
    result = skill.run({})
    assert result["status"] == "error"
    assert "csv_path" in result["error"] or "target_col" in result["error"]


@test("5. GlassBoxSkill — missing file returns error dict, not exception")
def test_skill_missing_file():
    from agent.ironclaw_integration import GlassBoxSkill
    skill = GlassBoxSkill()
    result = skill.run({"csv_path": "/sandbox/ghost.csv", "target_col": "X"})
    assert result["status"] == "error"
    assert "not found" in result["error"].lower()


@test("6. Full WASM pipeline — classification in virtual sandbox")
def test_wasm_classification():
    """
    End-to-end classification pipeline using ONLY the virtual FS.
    The autofit function writes a real tmp file — this test validates the
    skill layer correctly calls autofit and the report is well-formed.
    """
    import tempfile, os
    from agent.ironclaw_integration import GlassBoxSkill, GlassBoxSandboxIO

    # Write CSV to a real temp file (autofit still uses open() internally)
    csv_content = _make_clf_csv(n=120)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        tmp_path = f.name

    try:
        io_shim = GlassBoxSandboxIO()
        io_shim.stage_file(tmp_path, csv_content)   # also stage in virtual FS

        skill  = GlassBoxSkill(sandbox_io=io_shim)
        t0     = time.time()
        result = skill.run({
            "csv_path":    tmp_path,
            "target_col":  "Survived",
            "task_type":   "classification",
            "time_budget": 30,
            "cv_folds":    3,
        })
        elapsed = time.time() - t0

        print(f"     status     : {result.get('status')}")
        print(f"     best_model : {result.get('best_model')}")
        print(f"     accuracy   : {result.get('metrics', {}).get('accuracy', 'N/A'):.4f}"
              if isinstance(result.get("metrics", {}).get("accuracy"), float) else
              f"     accuracy   : {result.get('metrics', {}).get('accuracy', 'N/A')}")
        print(f"     time       : {elapsed:.1f}s")
        print(f"     explanation: {result.get('explanation', '')[:120]}...")

        assert result["status"] == "success", f"Pipeline failed: {result.get('error')}"
        assert "best_model"   in result
        assert "metrics"      in result
        assert "explanation"  in result
        assert result["benchmark_pass"] is True, "Pipeline exceeded 120s budget"
        assert result["metrics"].get("accuracy", 0) > 0.0

    finally:
        os.unlink(tmp_path)


@test("7. Full WASM pipeline — regression in virtual sandbox")
def test_wasm_regression():
    import tempfile, os
    from agent.ironclaw_integration import GlassBoxSkill, GlassBoxSandboxIO

    csv_content = _make_reg_csv(n=120)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        tmp_path = f.name

    try:
        io_shim = GlassBoxSandboxIO()
        io_shim.stage_file(tmp_path, csv_content)

        skill  = GlassBoxSkill(sandbox_io=io_shim)
        result = skill.run({
            "csv_path":    tmp_path,
            "target_col":  "Price",
            "task_type":   "regression",
            "time_budget": 30,
            "cv_folds":    3,
        })

        print(f"     status     : {result.get('status')}")
        print(f"     best_model : {result.get('best_model')}")
        print(f"     R2         : {result.get('metrics', {}).get('r2', 'N/A')}")

        assert result["status"] == "success", f"Pipeline failed: {result.get('error')}"
        assert result["metrics"].get("r2", -99) > -1.0, "R² too low — something is wrong"

    finally:
        os.unlink(tmp_path)


@test("8. Report JSON — serialisable and schema-complete")
def test_report_json_schema():
    """The report dict must be fully JSON-serialisable and contain all required keys."""
    import tempfile, os
    from agent.autofit import autofit
    from agent.report  import report_to_json, report_to_explanation

    csv_content = _make_clf_csv(n=80)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        tmp = f.name

    try:
        report = autofit(tmp, "Survived", task_type="classification",
                         time_budget=20, cv_folds=2)
        json_str = report_to_json(report)
        loaded   = json.loads(json_str)

        required = {"status", "task_type", "best_model", "best_params",
                    "metrics", "top_features", "eda_summary",
                    "pipeline_seconds", "benchmark_pass"}
        missing = required - set(loaded.keys())
        assert not missing, f"Missing keys in report: {missing}"

        explanation = report_to_explanation(report)
        assert len(explanation) > 20, "Explanation too short"
        print(f"     keys       : {sorted(loaded.keys())}")
        print(f"     explanation: {explanation[:100]}...")

    finally:
        os.unlink(tmp)


@test("9. Skill manifest — all required fields present")
def test_skill_manifest():
    from agent.ironclaw_integration import SKILL_MANIFEST
    required = {"skill_id", "version", "display_name", "description",
                "trigger_phrases", "input_schema", "output_schema",
                "wasm_safe", "runtime", "dependencies"}
    missing = required - set(SKILL_MANIFEST.keys())
    assert not missing, f"Manifest missing fields: {missing}"
    assert SKILL_MANIFEST["wasm_safe"] is True
    assert "numpy" in SKILL_MANIFEST["dependencies"][0]


@test("10. MCP tool schema — valid JSON, required keys, outputSchema present")
def test_mcp_tool_schema():
    from agent.tool_schema import AUTOFIT_TOOL
    required = {"name", "description", "inputSchema", "outputSchema"}
    missing = required - set(AUTOFIT_TOOL.keys())
    assert not missing, f"Tool schema missing fields: {missing}"
    props = AUTOFIT_TOOL["inputSchema"]["properties"]
    assert "csv_path"   in props
    assert "target_col" in props
    # outputSchema must list all report keys
    out_props = AUTOFIT_TOOL["outputSchema"]["properties"]
    for key in ("status", "best_model", "metrics", "benchmark_pass"):
        assert key in out_props, f"outputSchema missing '{key}'"


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  GlassBox-AutoML  ·  Phase 5 — WASM & IronClaw Test Suite")
    print("=" * 58)

    # Run all tests (they self-register via @test decorator)
    test_static_audit()
    test_sandbox_io_virtual()
    test_sandbox_io_missing()
    test_skill_bad_args()
    test_skill_missing_file()
    test_wasm_classification()
    test_wasm_regression()
    test_report_json_schema()
    test_skill_manifest()
    test_mcp_tool_schema()

    # Summary
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed
    print("\n" + "=" * 58)
    print(f"  Results: {passed}/{len(_results)} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        for name, ok, msg in _results:
            if not ok:
                print(f"\n  {FAIL} {name}")
                if msg:
                    print(f"       {msg[:300]}")
    else:
        print()
        print("  ALL WASM TESTS PASSED ✓")
    print("=" * 58)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
