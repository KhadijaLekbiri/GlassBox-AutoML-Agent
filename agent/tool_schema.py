"""
tool_schema.py
--------------
Defines the MCP-compatible tool specification for GlassBox-AutoML.

IronClaw agents discover tools through this schema.  When a user says
"Build a model to predict 'Survived' from this CSV", the agent matches
the intent and calls `autofit` with the appropriate arguments.

MCP tool spec reference:
  https://spec.modelcontextprotocol.io/specification/server/tools/
"""

# ── Tool definition (MCP-compatible JSON schema) ─────────────────────────────

AUTOFIT_TOOL = {
    "name": "AutoFit",
    "description": (
        "Automated end-to-end machine-learning pipeline. "
        "Given a CSV file path and a target column name, GlassBox will: "
        "(1) profile and clean the data, "
        "(2) search for the best model and hyperparameters, "
        "(3) evaluate on a held-out fold, and "
        "(4) return a structured JSON report with the best model, metrics, "
        "top features, and a plain-English explanation."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "csv_path": {
                "type": "string",
                "description": "Absolute or relative path to the input CSV file.",
            },
            "target_col": {
                "type": "string",
                "description": "Name of the column the model should predict.",
            },
            "task_type": {
                "type": "string",
                "enum": ["classification", "regression", "auto"],
                "default": "auto",
                "description": (
                    "Whether this is a classification or regression problem. "
                    "Use 'auto' to let GlassBox decide from the target column."
                ),
            },
            "time_budget": {
                "type": "integer",
                "default": 60,
                "description": (
                    "Maximum seconds allowed for hyperparameter search. "
                    "Must be less than 120 to satisfy the benchmark constraint."
                ),
            },
            "cv_folds": {
                "type": "integer",
                "default": 5,
                "description": "Number of K-Fold cross-validation folds.",
            },
        },
        "required": ["csv_path", "target_col"],
    },
    "outputSchema": {
        "type": "object",
        "description": "JSON report produced by GlassBox (see report.py for full spec).",
        "properties": {
            "status":           {"type": "string"},
            "task_type":        {"type": "string"},
            "best_model":       {"type": "string"},
            "best_params":      {"type": "object"},
            "metrics":          {"type": "object"},
            "top_features":     {"type": "array"},
            "eda_summary":      {"type": "object"},
            "pipeline_seconds": {"type": "number"},
            "benchmark_pass":   {"type": "boolean"},
        },
    },
}


# ── IronClaw registration helper ─────────────────────────────────────────────

def register_with_ironclaw(agent):
    """
    Register the AutoFit tool with an IronClaw agent instance.

    Usage
    -----
    from agent.tool_schema import register_with_ironclaw
    register_with_ironclaw(my_ironclaw_agent)

    The agent must expose a `.register_tool(schema, handler)` method.
    """
    from .autofit import autofit

    def _handler(csv_path: str, target_col: str,
                 task_type: str = "auto",
                 time_budget: int = 60,
                 cv_folds: int = 5) -> dict:
        return autofit(
            csv_path=csv_path,
            target_col=target_col,
            task_type=task_type,
            time_budget=time_budget,
            cv_folds=cv_folds,
        )

    agent.register_tool(AUTOFIT_TOOL, _handler)
    print(f"[GlassBox] AutoFit tool registered with IronClaw agent.")


# ── MCP server (stdio transport) ─────────────────────────────────────────────
# Run this file directly to expose GlassBox as a standalone MCP server:
#   python -m agent.tool_schema
#
# The MCP client (e.g. Claude Desktop, IronClaw) connects over stdio.

def _run_mcp_server():
    """
    Minimal MCP server using the official `mcp` Python SDK.
    Install: pip install mcp
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
        import asyncio
        import json
    except ImportError:
        raise ImportError(
            "MCP SDK not installed. Run: pip install mcp"
        )

    from .autofit import autofit

    server = Server("glassbox-automl")

    @server.list_tools()
    async def list_tools():
        return [Tool(**AUTOFIT_TOOL)]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name != "AutoFit":
            raise ValueError(f"Unknown tool: {name}")
        result = autofit(**arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def main():
        async with stdio_server() as (r, w):
            await server.run(r, w, server.create_initialization_options())

    asyncio.run(main())


if __name__ == "__main__":
    _run_mcp_server()