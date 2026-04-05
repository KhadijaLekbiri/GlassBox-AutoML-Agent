# GlassBox-AutoML

A transparent, scratch-built AutoML library. NumPy only.

## Install
```bash
pip install -r requirements.txt
```

## Run benchmark (Phase 5 test)
```bash
python benchmark.py
```

## Use directly
```python
from agent.autofit import autofit
from agent.report  import report_to_explanation

report = autofit("your_data.csv", target_col="Target")
print(report_to_explanation(report))
```

## Expose as MCP tool (Claude Desktop / IronClaw)
```bash
python -m agent.tool_schema
```

Add to `~/.claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "glassbox": {
      "command": "python",
      "args": ["-m", "agent.tool_schema"],
      "cwd": "/path/to/GlassBox-AutoML-Agent"
    }
  }
}
```

## Project structure
```
GlassBox-AutoML-Agent/
├── core/           ← math: stats, metrics, distances, matrix, utils
├── eda/            ← DataInspector (EDA profiling)
├── Models/         ← LinearRegression, LogisticRegression, DecisionTree,
│                      RandomForest, KNearestNeighbors, GaussianNB
├── Optimization/   ← KFold, GridSearch, RandomSearch, helpers
├── Preprocessing/  ← Preprocessor, SimpleImputer, Scalers, Encoders
├── agent/          ← autofit.py (pipeline), report.py, tool_schema.py
├── benchmark.py    ← Phase 5 end-to-end test
├── wasm_audit.py   ← WASM compatibility scanner
└── requirements.txt
```
