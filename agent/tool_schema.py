"""
tool_schema.py — GlassBox MCP server
"""

import os
import sys
import asyncio
import json

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

AUTOFIT_TOOL = {
    "name": "AutoFit",
    "description": (
        "Automated end-to-end machine-learning pipeline. "
        "Pass ONLY the filename (e.g. 'titanic_dataset.csv'), never a full path. "
        "The tool automatically finds the file in the agent/ folder. "
        "Returns a JSON report with best model, metrics, and top features."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "csv_path": {
                "type": "string",
                "description": "Filename only, e.g. 'titanic_dataset.csv'. Do NOT include any folder path.",
            },
            "target_col": {
                "type": "string",
                "description": "Name of the column to predict.",
            },
            "task_type": {
                "type": "string",
                "enum": ["classification", "regression", "auto"],
                "default": "auto",
            },
            "time_budget": {
                "type": "integer",
                "default": 60,
                "description": "Max seconds for search. Must be under 120.",
            },
            "cv_folds": {
                "type": "integer",
                "default": 5,
            },
        },
        "required": ["csv_path", "target_col"],
    }
}


def _resolve_csv(csv_input: str) -> str:
    """
    Robustly resolve a CSV path to an absolute path in AGENT_DIR.

    Handles all Windows escaping variants IronClaw may produce:
      - quadruple backslashes:  C:\\\\GlassBox\\\\agent\\\\file.csv
      - double backslashes:     C:\\GlassBox\\agent\\file.csv
      - single backslashes:     C:\GlassBox\agent\file.csv
      - forward slashes:        C:/GlassBox/agent/file.csv
      - bare filename:          file.csv

    Always returns:  AGENT_DIR\filename.csv
    """
    # Step 1: decode any double-escaped backslashes
    cleaned = csv_input.replace('\\\\', '\\')

    # Step 2: normalise all separators to forward slash
    cleaned = cleaned.replace('\\', '/')

    # Step 3: extract just the filename — ignore any directory component
    filename = cleaned.split('/')[-1].strip()

    if not filename:
        filename = csv_input.strip()  # last resort fallback

    # Step 4: build the final path inside AGENT_DIR
    return os.path.join(AGENT_DIR, filename)


def _run_mcp_server():
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
    except ImportError:
        raise ImportError("Run: pip install mcp")

    from .autofit import autofit

    server = Server("glassbox-automl")

    @server.list_tools()
    async def list_tools():
        return [Tool(**AUTOFIT_TOOL)]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name != "AutoFit":
            raise ValueError(f"Unknown tool: {name}")

        csv_input = arguments.get("csv_path", "")
        csv_path  = _resolve_csv(csv_input)

        print(f"[GlassBox] csv_input  = {repr(csv_input)}", file=sys.stderr, flush=True)
        print(f"[GlassBox] csv_path   = {repr(csv_path)}",  file=sys.stderr, flush=True)

        if not os.path.isfile(csv_path):
            available = [f for f in os.listdir(AGENT_DIR) if f.endswith(".csv")]
            result = {
                "status": "error",
                "error": f"File not found: {csv_path}",
                "available_csv_files": available,
                "hint": f"Pass only the filename, e.g. '{available[0] if available else 'data.csv'}'"
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        kwargs = {k: v for k, v in arguments.items() if k != "csv_path"}
        result  = autofit(csv_path=csv_path, **kwargs)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def main():
        async with stdio_server() as (r, w):
            await server.run(r, w, server.create_initialization_options())

    asyncio.run(main())


if __name__ == "__main__":
    _run_mcp_server()