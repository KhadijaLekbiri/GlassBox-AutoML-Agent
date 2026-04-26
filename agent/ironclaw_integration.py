"""
agent/ironclaw_integration.py
------------------------------
GlassBox adapter for agentic runtimes (IronClaw / NEAR AI / Claude Desktop).

HOW THE INTEGRATION ACTUALLY WORKS
------------------------------------
GlassBox does NOT use a proprietary "IronClaw SDK" — it speaks the open
Model Context Protocol (MCP) standard, which is the correct, real way to
integrate with agentic runtimes including IronClaw and Claude Desktop.

Architecture:

    User: "Build a model to predict 'Survived'"
               ↓
    Agent matches intent → selects AutoFit tool via MCP
               ↓
    Agent spawns subprocess:  python run_server.py   (stdio transport)
               ↓
    GlassBox pipeline runs → returns JSON over stdout
               ↓
    Agent reads JSON → explains result to user in plain English

The actual MCP server lives in:   agent/tool_schema.py → _run_mcp_server()
The entry point is:                run_server.py
Claude Desktop config is at:       claude_desktop_config.json  (project root)

This file provides:
  1. SKILL_MANIFEST  — metadata / tool description for any agent that needs it
  2. GlassBoxSkill   — a clean Python wrapper around the autofit pipeline
                       (useful for testing or embedding without MCP overhead)
"""

from __future__ import annotations

import json
import time
from typing import Any


# ── Skill manifest ─────────────────────────────────────────────────────────────
# This is the human/agent-readable description of what GlassBox can do.
# MCP-compatible runtimes use the schema in agent/tool_schema.py instead,
# but this manifest is kept here for documentation and non-MCP embeddings.

SKILL_MANIFEST = {
    "skill_id":    "glassbox.automl.autofit",
    "version":     "1.0.0",
    "display_name": "GlassBox AutoML",
    "description": (
        "Automated end-to-end machine-learning pipeline. "
        "Given a CSV and a target column, GlassBox profiles the data, "
        "cleans it, searches for the best model, and returns a structured "
        "JSON report with metrics and a plain-English explanation."
    ),
    "trigger_phrases": [
        "build a model", "train a model", "predict",
        "classify", "regression", "machine learning", "automl",
    ],
    "input_schema": {
        "type": "object",
        "required": ["csv_path", "target_col"],
        "properties": {
            "csv_path":    {"type": "string",  "description": "Path to the CSV file."},
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
    "transport": "mcp-stdio",     # real integration mechanism
    "wasm_safe": True,
    "runtime":   "python3.11+",
    "dependencies": ["numpy>=1.26.0"],
}


# ── Sandbox I/O shim ──────────────────────────────────────────────────────────

class GlassBoxSandboxIO:
    """
    Thin file-I/O wrapper that stays WASM-compatible.

    In a WASM sandbox the host injects a virtual filesystem.
    Files uploaded by the user are pre-staged at a known path.
    Outside the sandbox this class falls back to plain disk I/O
    so you can run and test locally without any changes.
    """

    def __init__(self, sandbox_root: str = "/sandbox"):
        self._root = sandbox_root
        self._virtual_fs: dict[str, str] = {}

    def stage_file(self, virtual_path: str, content: str) -> None:
        """Pre-stage a file into the in-memory virtual filesystem."""
        self._virtual_fs[virtual_path] = content

    def read_text(self, path: str) -> str:
        """Read a text file — virtual FS first, then real disk."""
        if path in self._virtual_fs:
            return self._virtual_fs[path]
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


# ── GlassBox skill wrapper ────────────────────────────────────────────────────

class GlassBoxSkill:
    """
    Python wrapper around the GlassBox autofit pipeline.

    Use this class when you want to call GlassBox directly from Python
    (e.g. unit tests, embedding in another framework, or a WASM host that
    drives Python directly).

    For Claude Desktop / IronClaw / any MCP-compatible agent, use the
    MCP server instead (run_server.py).

    Example
    -------
    skill = GlassBoxSkill()
    result = skill.run({"csv_path": "agent/data.csv", "target_col": "Survived"})
    print(result["explanation"])
    """

    MANIFEST = SKILL_MANIFEST

    def __init__(self, sandbox_io: GlassBoxSandboxIO | None = None):
        self._io = sandbox_io or GlassBoxSandboxIO()

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
        dict  — full GlassBox report with an ``explanation`` string attached.
        """
        csv_path   = arguments.get("csv_path")
        target_col = arguments.get("target_col")
        if not csv_path or not target_col:
            return {
                "status": "error",
                "error":  "Missing required arguments: 'csv_path' and 'target_col'.",
            }

        if not self._io.exists(csv_path):
            return {
                "status": "error",
                "error":  f"CSV file not found: {csv_path}",
            }

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

            report["explanation"] = report_to_explanation(report)

            report_path = csv_path.replace(".csv", "_glassbox_report.json")
            self._io.write_text(report_path, json.dumps(report, indent=2))
            report["report_path"] = report_path

            return report

        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    @classmethod
    def get_manifest(cls) -> dict:
        """Return the skill manifest for agent discovery."""
        return cls.MANIFEST