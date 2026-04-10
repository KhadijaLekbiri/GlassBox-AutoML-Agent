import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import agent.autofit as _autofit_module
from agent.autofit import autofit as _original_autofit

def _patched_autofit(csv_path, target_col, **kwargs):
    filename = os.path.basename(csv_path)
    local_path = os.path.join(PROJECT_DIR, filename)
    print(f"[GlassBox] Resolved path: {local_path}", file=sys.stderr)
    return _original_autofit(local_path, target_col, **kwargs)

_autofit_module.autofit = _patched_autofit

from agent.tool_schema import _run_mcp_server
_run_mcp_server()