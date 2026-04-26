# Team : 
Hiba Ouhmad
Ghita Bellamine
Khadija Lekbiri
Rhita Rhallami 

# 🧠 GlassBox AutoML Agent

> **Make AutoML understandable, not just powerful.**

GlassBox is a lightweight, explainable AutoML framework that automatically builds, trains, and evaluates machine learning models from a CSV dataset — while keeping every step fully transparent.

Unlike traditional *black-box* AutoML systems, GlassBox follows a **glass-box approach**, allowing users and agents to understand:

- ✅ What model was selected and why
- ✅ Which features drive predictions
- ✅ Every transformation applied to the data

---

## ⚙️ Features

- 📊 **Automated Exploratory Data Analysis (EDA)** — column profiling, outlier detection, correlation
- 🧹 **Data cleaning & preprocessing** — imputation, scaling, encoding
- 🤖 **Automatic model selection** — classification & regression, multiple algorithms
- 🔍 **Hyperparameter optimization** — Grid Search and Random Search with time budget
- 📈 **Honest evaluation** — K-Fold Cross-Validation on every configuration
- 📦 **Structured JSON output** — ready for APIs, agents, and dashboards
- 🧩 **Agent-ready architecture** — IronClaw (NEAR AI) compatible via MCP

---

## 🏗️ Project Structure

```
GlassBox-AutoML-Agent/
├── run_server.py
├── run_glassbox.bat
├── ironclaw_mcp_config.json
├── IRONCLAW_DEPLOY.md
├── SKILL.md
├── README.md
├── requirements.txt
├── benchmark.py
├── test.py
├── test_mcp_client.py
├── wasm_audit.py
├── wasm_test.py
├── titanic_dataset.csv
├── agent/
│   ├── __init__.py
│   ├── autofit.py
│   ├── ironclaw_integration.py
│   ├── report.py
│   ├── tool_schema.py
│   └── titanic_dataset.csv
├── core/
├── eda/
├── Models/
├── Optimization/
└── Preprocessing/

```

---

## 🚀 How It Works

GlassBox exposes its full pipeline as a single MCP tool called **AutoFit**. Any MCP-compatible agent (IronClaw, Claude Desktop) can call it by name. The pipeline runs automatically:

```
User natural language request
         ↓
IronClaw matches intent → activates GlassBox skill (SKILL.md)
         ↓
Calls AutoFit MCP tool via stdio transport
         ↓
run_server.py spawned as subprocess
         ↓
┌─────────────────────────────────┐
│  EDA       → column profiling   │
│  Imputer   → fill missing data  │
│  Scaler    → normalize features │
│  Encoder   → handle categories  │
│  Search    → Grid / Random CV   │
│  Evaluate  → metrics on holdout │
└─────────────────────────────────┘
         ↓
JSON report returned over stdout
         ↓
Agent explains results in plain English
```

---

## 🔧 Setup Guide

### 1️⃣ Prerequisites

- Python 3.11+
- IronClaw v0.26.0+
- pip

### 2️⃣ Install Python Dependencies

```bash
cd GlassBox-AutoML-Agent
pip install -r requirements.txt
```

### 3️⃣ Test the MCP Server Standalone

Before connecting any agent, verify the pipeline works end-to-end:

```bash
python test_mcp_client.py
```

Expected output:
```
📦 Available tools:
  - AutoFit: Automated end-to-end machine-learning pipeline...

🚀 Calling AutoFit...

✅ RESULT:
{
  "status": "success",
  "best_model": "LogisticRegression",
  "metrics": { "accuracy": 0.727, "f1": 0.697 },
  "benchmark_pass": true,
  "pipeline_seconds": 15.3
}
```

If this works, the ML pipeline is fully functional. The next steps connect it to IronClaw.

---

## 🤖 IronClaw Deployment

### 4️⃣ Install IronClaw

**Windows:**
Download the MSI from:
```
https://github.com/nearai/ironclaw/releases/download/ironclaw-v0.26.0/ironclaw-x86_64-pc-windows-msvc.msi
```
Run the installer, restart PowerShell, then verify:
```powershell
ironclaw --version
```

**macOS / Linux:**
```bash
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/nearai/ironclaw/releases/latest/download/ironclaw-installer.sh | sh
```

### 5️⃣ Configure the LLM Backend

Edit `~/.ironclaw/.env` and add your preferred LLM provider.

**Option A — Anthropic Claude (recommended, best tool-calling reliability):**
```
LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```
Get an API key at https://console.anthropic.com

**Option B — OpenRouter (free tier available):**
```
LLM_BACKEND=openai_compatible
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL=openrouter/free
```
Get a free key at https://openrouter.ai

### 6️⃣ Register the MCP Server

Edit `~/.ironclaw/mcp-servers.json`:

```json
{
  "servers": [
    {
      "name": "glassbox",
      "url": "",
      "transport": {
        "transport": "stdio",
        "command": "python",
        "args": [
          "ABSOLUTE_PATH_TO_PROJECT/run_server.py"
        ],
        "env": {
          "GLASSBOX_PROJECT_PATH": "ABSOLUTE_PATH_TO_PROJECT",
          "PYTHONUNBUFFERED": "1"
        }
      },
      "enabled": true
    }
  ],
  "schema_version": 0
}
```

> ⚠️ **Windows users:** use double backslashes in paths: `C:\\GlassBox-AutoML-Agent\\run_server.py`
> ⚠️ Use `python` not `python3` — `python3` does not exist on Windows.

Verify the connection:
```powershell
ironclaw mcp test glassbox
```
Expected: `✓ glassbox connected (1 tool: AutoFit)`

### 7️⃣ Install the GlassBox Skill

```powershell
# Windows
mkdir "$env:USERPROFILE\.ironclaw\skills\glassbox-automl"
copy "skills\glassbox-automl\SKILL.md" "$env:USERPROFILE\.ironclaw\skills\glassbox-automl\SKILL.md"
```

```bash
# macOS / Linux
mkdir -p ~/.ironclaw/skills/glassbox-automl
cp skills/glassbox-automl/SKILL.md ~/.ironclaw/skills/glassbox-automl/SKILL.md
```

Verify:
```bash
ironclaw skills list
# Expected: glassbox-automl  v1.0.0  Trusted
```

### 8️⃣ Disable Irrelevant Tools (Recommended)

Prevents the LLM from calling wrong tools instead of AutoFit:
```bash
ironclaw tools disable google_drive
ironclaw tools disable gmail
ironclaw tools disable google_calendar
ironclaw tools disable github
```

### 9️⃣ Launch IronClaw and Run AutoFit

```bash
cd GlassBox-AutoML-Agent
ironclaw
```

Then type in the IronClaw chat:
```
Call glassbox_AutoFit with csv_path="titanic_dataset.csv", target_col="Survived", time_budget=10, cv_folds=2
```

Or naturally:
```
Build a model to predict whether a Titanic passenger survived.
```

---

## 📤 Output Format

GlassBox returns a structured JSON report:

```json
{
  "status": "success",
  "task_type": "classification",
  "best_model": "LogisticRegression",
  "best_params": {
    "lr": 0.01,
    "epochs": 200
  },
  "metrics": {
    "accuracy": 0.727,
    "f1": 0.697,
    "precision": 0.604,
    "recall": 0.843
  },
  "top_features": [
    { "feature": "Age",    "importance": 0.0063 },
    { "feature": "Pclass", "importance": 0.0034 }
  ],
  "eda_summary": {
    "n_rows": 891,
    "n_cols": 11,
    "outliers_flagged": 375,
    "missing_filled": 3971
  },
  "pipeline_seconds": 15.3,
  "benchmark_pass": true
}
```

---

## ✅ Project Success Metrics

| Metric | Target | Result |
|--------|--------|--------|
| Zero-dependency core | NumPy only | ✅ Achieved |
| Benchmark accuracy | ≥ 90% of sklearn | ✅ 72.7% vs ~80% sklearn = 90.9% |
| Pipeline time | < 120 seconds | ✅ ~15 seconds |
| Agent integration | IronClaw MCP | ✅ Confirmed working |

---

## ⚠️ Known Limitations

- **IronClaw 30s timeout:** IronClaw v0.26.0 hard-cuts tool calls at 30 seconds. Keep `time_budget ≤ 10` and `cv_folds = 2` to stay within this limit. This is an IronClaw limitation, not a GlassBox bug.
- **LLM quality matters:** Small free models (llama3.2, llama3.1-8b) may not reliably call the AutoFit tool. Claude Haiku or a strong OpenRouter model is recommended for best results.
- **CSV files must be in `agent/`:** Place your dataset in the `agent/` subfolder before calling AutoFit.

---

## 🔮 Future Improvements

- WASM compilation for sandboxed execution inside IronClaw's secure TEE runtime
- Bayesian hyperparameter search
- Advanced visualization dashboards
- Streaming progress updates over MCP
- Support for multi-class classification and time-series data



## 🧠 Key Idea

GlassBox is built with one goal:

> **Make AutoML understandable, not just powerful.**

- 📌 No hidden decisions
- 📌 Clear model reasoning
- 📌 Interpretable feature importance
- 📌 Full pipeline transparency — every step is inspectable
