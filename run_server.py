"""
run_server.py
-------------
Entry point for the GlassBox MCP server.

Claude Desktop (or any MCP-compatible agent) launches this script as a
subprocess and communicates with it over stdin/stdout using the MCP protocol.

Usage (Claude Desktop handles this automatically via claude_desktop_config.json):
    python run_server.py

Manual test:
    python test_mcp_client.py
"""

import sys
import os
import pathlib

# ── Make sure the project root is on the Python path ─────────────────────────
PROJECT_DIR = pathlib.Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ── Set working directory to agent/ so relative CSV paths resolve correctly ───
# Use pathlib throughout to avoid Windows Errno 22 path issues
AGENT_DIR = PROJECT_DIR / "agent"
os.chdir(str(AGENT_DIR))

# ── Start the MCP server ──────────────────────────────────────────────────────
from agent.tool_schema import _run_mcp_server

if __name__ == "__main__":
    _run_mcp_server()