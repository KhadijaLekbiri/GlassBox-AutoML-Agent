"""
agent/ironclaw_integration.py
------------------------------
IronClaw (NEAR AI) agent integration for GlassBox-AutoML.

This module bridges GlassBox to the IronClaw agentic runtime. It wraps
the AutoFit pipeline as a skill that the agent can discover, invoke, and
explain back to the user in natural language.

IronClaw skill contract
-----------------------
An IronClaw skill is a Python callable that:
  1. Accepts a plain dict of arguments (the agent fills these from user intent).
  2. Returns a JSON-serialisable dict.
  3. Exposes a SKILL_MANIFEST dict for discovery.

The sandbox executes the skill inside a WASM-isolated environment — all
I/O must go through the `GlassBoxSandboxIO` interface below; no raw
`open()` or `os.*` calls are allowed at runtime.
"""

from __future__ import annotations

import json
import time
from typing import Any

# ── Skill manifest (IronClaw discovery format) ────────────────────────────────

SKILL_MANIFEST = {
    "skill_id":    "glassbox.automl.autofit",
    "version":     "1.0.0",
    "display_name": "GlassBox AutoML",
    "description": (
        "Automated end-to-end machine-learning pipeline. "
        "Given a CSV and a target column, GlassBox profiles the data, "
        "cleans it, searches for the best model, and returns a structured "
        "report with metrics and a plain-English explanation."
    ),
    "trigger_phrases": [
        "build a model",
        "train a model",
        "predict",
        "classify",
        "regression",
        "machine learning",
        "automl",
    ],
    "input_schema": {
        "type": "object",
        "required": ["csv_path", "target_col"],
        "properties": {
            "csv_path":    {"type": "string",  "description": "Path to the CSV file inside the sandbox."},
            "target_col":  {"type": "string",  "description": "Column name to predict."},
            "task_type":   {"type": "string",  "enum": ["classification", "regression", "auto"], "default": "auto"},
            "time_budget": {"type": "integer", "default": 60,  "description": "Max seconds for search (< 120)."},
            "cv_folds":    {"type": "integer", "default": 5,   "description": "K-Fold splits."},
        },
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status":           {"type": "string"},
            "best_model":       {"type": "string"},
            "metrics":          {"type": "object"},
            "top_features":     {"type": "array"},
            "explanation":      {"type": "string"},
            "benchmark_pass":   {"type": "boolean"},
            "pipeline_seconds": {"type": "number"},
        },
    },
    "wasm_safe": True,
    "runtime": "python3.11+",
    "dependencies": ["numpy>=1.26.0"],
}


# ── Sandbox I/O shim ──────────────────────────────────────────────────────────

class GlassBoxSandboxIO:
    """
    Thin wrapper around file I/O that is WASM-compatible.

    In the IronClaw WASM sandbox the host runtime injects a virtual
    filesystem.  Files that the agent passes to the skill (e.g. the user's
    uploaded CSV) are pre-staged at a known sandbox path.

    Outside the sandbox (local dev / testing) this class falls back to
    plain filesystem access so you can run and test normally.

    Usage
    -----
    io = GlassBoxSandboxIO()
    content = io.read_text("data.csv")
    io.write_text("report.json", json.dumps(report))
    """

    def __init__(self, sandbox_root: str = "/sandbox"):
        self._root = sandbox_root
        self._virtual_fs: dict[str, str] = {}   # in-memory FS for pure-WASM mode

    # -- write a file into the virtual FS (used by tests / sandbox host)
    def stage_file(self, virtual_path: str, content: str) -> None:
        """Pre-stage a file into the in-memory virtual filesystem."""
        self._virtual_fs[virtual_path] = content

    def read_text(self, path: str) -> str:
        """Read a text file — virtual FS first, then real disk."""
        if path in self._virtual_fs:
            return self._virtual_fs[path]
        # Real disk fallback (local dev only — not available in WASM)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError as exc:
            raise FileNotFoundError(
                f"[GlassBox] File not found in sandbox or on disk: {path}"
            ) from exc

    def write_text(self, path: str, content: str) -> None:
        """Write output back to the virtual FS (and optionally to disk)."""
        self._virtual_fs[path] = content
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError:
            pass  # WASM mode: only virtual FS write succeeds — that's fine

    def exists(self, path: str) -> bool:
        import os
        return path in self._virtual_fs or os.path.isfile(path)


# ── IronClaw skill entry point ────────────────────────────────────────────────

class GlassBoxSkill:
    """
    The IronClaw skill class.

    IronClaw instantiates this once per session and calls `.run()` whenever
    a user intent matches one of the trigger phrases in SKILL_MANIFEST.

    Example
    -------
    skill = GlassBoxSkill()
    result = skill.run({"csv_path": "/sandbox/upload.csv", "target_col": "Survived"})
    print(result["explanation"])
    """

    MANIFEST = SKILL_MANIFEST

    def __init__(self, sandbox_io: GlassBoxSandboxIO | None = None):
        self._io = sandbox_io or GlassBoxSandboxIO()

    # ── main entry point called by IronClaw ───────────────────────────────────
    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the GlassBox pipeline and return a JSON-serialisable result.

        Parameters
        ----------
        arguments : dict
            Must contain ``csv_path`` and ``target_col``.
            Optional: ``task_type``, ``time_budget``, ``cv_folds``.

        Returns
        -------
        dict
            Full GlassBox report enriched with a ``explanation`` string for
            the agent to relay to the user.
        """
        # -- validate inputs
        csv_path   = arguments.get("csv_path")
        target_col = arguments.get("target_col")
        if not csv_path or not target_col:
            return {
                "status": "error",
                "error":  "Missing required arguments: 'csv_path' and 'target_col'.",
            }

        # -- verify file is accessible
        if not self._io.exists(csv_path):
            return {
                "status": "error",
                "error":  f"CSV file not found in sandbox: {csv_path}",
            }

        # -- run pipeline
        try:
            from agent.autofit import autofit
            from agent.report  import report_to_explanation

            report = autofit(
                csv_path   = csv_path,
                target_col = target_col,
                task_type  = arguments.get("task_type",   "auto"),
                time_budget= arguments.get("time_budget", 60),
                cv_folds   = arguments.get("cv_folds",    5),
            )

            # attach the plain-English explanation for the agent to relay
            report["explanation"] = report_to_explanation(report)

            # persist JSON report back into sandbox
            report_path = csv_path.replace(".csv", "_glassbox_report.json")
            self._io.write_text(report_path, json.dumps(report, indent=2))
            report["report_path"] = report_path

            return report

        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error":  str(exc),
            }

    # ── IronClaw discovery hook ───────────────────────────────────────────────
    @classmethod
    def get_manifest(cls) -> dict:
        """Return the skill manifest for IronClaw discovery."""
        return cls.MANIFEST

    # ── convenience: register with a live IronClaw agent object ───────────────
    @classmethod
    def register(cls, agent, sandbox_io: GlassBoxSandboxIO | None = None) -> None:
        """
        Register GlassBox as a skill with a live IronClaw agent instance.

        The agent must expose a ``register_skill(manifest, handler)`` method.

        Usage
        -----
        import ironclaw
        agent = ironclaw.Agent(...)
        GlassBoxSkill.register(agent)
        """
        skill = cls(sandbox_io=sandbox_io)
        agent.register_skill(cls.MANIFEST, skill.run)
        print("[GlassBox] Skill registered with IronClaw agent.")
