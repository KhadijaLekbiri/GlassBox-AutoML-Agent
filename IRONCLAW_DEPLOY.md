# GlassBox — IronClaw Deployment & Testing Guide

This guide covers everything you need to deploy GlassBox as a live skill
inside IronClaw and verify it works end-to-end.

---

## How GlassBox Plugs Into IronClaw

IronClaw has two complementary extension systems, and GlassBox uses **both**:

```
┌──────────────────────────────────────────────────────────┐
│                        IronClaw                          │
│                                                          │
│  ┌─────────────────┐        ┌──────────────────────┐    │
│  │   SKILL.md      │        │   MCP Server         │    │
│  │  (Prompt layer) │        │  (Tool layer)        │    │
│  │                 │        │                      │    │
│  │ Tells the agent │        │ Exposes AutoFit as   │    │
│  │ WHEN and HOW to │───────▶│ a callable tool the  │    │
│  │ call the tool,  │        │ agent can actually   │    │
│  │ and how to      │        │ execute              │    │
│  │ explain results │        │                      │    │
│  └─────────────────┘        └──────────┬───────────┘    │
│                                        │                 │
└────────────────────────────────────────┼─────────────────┘
                                         │ stdio / JSON-RPC
                                         ▼
                              ┌──────────────────────┐
                              │   run_server.py       │
                              │   (your machine)      │
                              │                       │
                              │   GlassBox pipeline:  │
                              │   EDA → Preprocess    │
                              │   → Search → Report   │
                              └──────────────────────┘
```

- **SKILL.md** lives in `skills/glassbox-automl/SKILL.md`. It is a prompt
  extension — it tells the agent *when* to activate (keyword/pattern matching)
  and *how* to use the tool and explain results. No code runs here.

- **MCP server** (`run_server.py`) is a real subprocess IronClaw spawns and
  communicates with over stdin/stdout. This is where the actual ML pipeline
  runs.

---

## Part 1 — Install IronClaw

### Option A: NEAR AI Cloud (recommended, hosted TEE)

1. Go to https://agent.near.ai/ and sign in.
2. Create a new IronClaw agent instance.
3. Connect via SSH using the address shown in the dashboard:
   ```bash
   ssh -p <PORT> <username>@agent2.near.ai
   ```
4. IronClaw is already installed on the instance. Skip to Part 2.

### Option B: Local Installation

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/nearai/ironclaw/releases/latest/download/ironclaw-installer.sh | sh

# Windows — download and run the installer from:
# https://github.com/nearai/ironclaw/releases/latest
```

Start IronClaw for the first time (this runs a setup wizard):
```bash
ironclaw
```

The wizard will ask you to choose an LLM provider. You can use:
- **NEAR AI** — default, most private (runs in TEE)
- **Anthropic** — set `LLM_BACKEND=anthropic` and `ANTHROPIC_API_KEY=sk-...`
- **OpenAI**, Mistral, Ollama, etc.

---

## Part 2 — Deploy the GlassBox Project

### 2.1 Clone / Copy Your Project

```bash
# On your local machine or the IronClaw SSH instance:
git clone https://github.com/YOUR_USERNAME/GlassBox-AutoML-Agent-main.git
cd GlassBox-AutoML-Agent-main
```

### 2.2 Install Python Dependencies

```bash
pip install -r requirements.txt
# requirements.txt includes: numpy>=1.26.0  mcp>=1.0.0  scikit-learn>=1.4.0
```

### 2.3 Set the Environment Variable

IronClaw will inject this into the subprocess so `run_server.py` knows where
the project lives:

```bash
# Add to your shell profile (~/.bashrc or ~/.zshrc):
export GLASSBOX_PROJECT_PATH="/absolute/path/to/GlassBox-AutoML-Agent-main"

# Or set it inline for testing:
GLASSBOX_PROJECT_PATH=$(pwd) ironclaw mcp test glassbox
```

---

## Part 3 — Register the MCP Server

This registers `run_server.py` as a tool server IronClaw can call.

### Option A: CLI (quickest)

```bash
ironclaw mcp add glassbox --transport stdio \
  --command python3 \
  --arg "$GLASSBOX_PROJECT_PATH/run_server.py" \
  --env GLASSBOX_PROJECT_PATH="$GLASSBOX_PROJECT_PATH" \
  --env PYTHONUNBUFFERED=1
```

On Windows (PowerShell):
```powershell
ironclaw mcp add glassbox --transport stdio `
  --command python `
  --arg "$env:GLASSBOX_PROJECT_PATH\run_server.py" `
  --env GLASSBOX_PROJECT_PATH="$env:GLASSBOX_PROJECT_PATH" `
  --env PYTHONUNBUFFERED=1
```

### Option B: Edit the config file directly

Copy the contents of `ironclaw_mcp_config.json` into `~/.ironclaw/mcp-servers.json`,
replacing the two placeholder values:

- `PYTHON_EXECUTABLE` → output of `which python3` (macOS/Linux) or `where python` (Windows)
- `ABSOLUTE_PATH_TO_PROJECT` → the full path to your project folder

---

## Part 4 — Install the SKILL.md

The skill file tells IronClaw *when* and *how* to use GlassBox.

```bash
# Create the skills directory and copy the skill:
mkdir -p ~/.ironclaw/skills/glassbox-automl
cp skills/glassbox-automl/SKILL.md ~/.ironclaw/skills/glassbox-automl/SKILL.md
```

This installs it as a **Trusted** skill (full tool access). If you want to
install it as a read-only **Installed** skill instead, use:

```bash
mkdir -p ~/.ironclaw/installed_skills/glassbox-automl
cp skills/glassbox-automl/SKILL.md ~/.ironclaw/installed_skills/glassbox-automl/SKILL.md
```

> **Tip:** For development, always use `~/.ironclaw/skills/` (Trusted).
> Trusted skills have full access to the MCP tools they need.

Alternatively, if you're working **inside your project directory**, IronClaw
auto-discovers skills from the workspace `skills/` folder — meaning you don't
need to copy anything. Just run IronClaw from the project root:

```bash
cd GlassBox-AutoML-Agent-main
ironclaw
```

---

## Part 5 — Test the Deployment

### 5.1 Verify MCP server connectivity

```bash
ironclaw mcp test glassbox
```

Expected output:
```
✓  glassbox  connected  (1 tool: AutoFit)
```

If it fails, check:
```bash
# Debug logging
RUST_LOG=ironclaw::tools::mcp=debug ironclaw mcp test glassbox
```

### 5.2 Verify the skill is loaded

```bash
ironclaw skills list
```

Expected output:
```
glassbox-automl  v1.0.0  Trusted  [ml, data-science, automl, prediction]
```

### 5.3 Verify the MCP server runs standalone

Before involving IronClaw at all, confirm the Python server itself works:

```bash
# Test the MCP server directly (runs the full pipeline):
python test_mcp_client.py
```

Expected output:
```
📦 Available tools:
[Tool(name='AutoFit', ...)]

🚀 Calling AutoFit...

✅ RESULT:
{"status": "success", "best_model": "...", "metrics": {...}, ...}
```

### 5.4 End-to-end test inside IronClaw

Start IronClaw from the project root:
```bash
cd GlassBox-AutoML-Agent-main
ironclaw
```

Then type:
```
Use GlassBox AutoFit with csv_path="titanic_dataset.csv" and target_col="Survived"
```

Or more naturally:
```
Build a model to predict whether a Titanic passenger survived. Use the file titanic_dataset.csv.
```

The agent will:
1. Match `glassbox-automl` skill via keywords ("predict", "model", "csv")
2. Call the `AutoFit` MCP tool
3. Receive the JSON report
4. Explain the result to you in plain English

Expected agent response (approximately):
```
Model selected: RandomForest_clf
Task type:      classification
Key metric:     accuracy 81.3%

Top features:
  1. Sex    — 31.2% importance
  2. Pclass — 22.4% importance
  3. Fare   — 14.8% importance

Pipeline completed in 38.2s  ✓  (benchmark_pass: true)

The best model was a Random Forest classifier, which won the search
because it handles mixed feature types and non-linear relationships well
without overfitting. The most predictive factor was passenger sex, followed
by ticket class — consistent with historical records of the disaster.
```

---

## Part 6 — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ironclaw mcp test glassbox` fails | Wrong Python path or project path | Run `which python3` and use that full path |
| Skill doesn't activate | Missing `GLASSBOX_PROJECT_PATH` env var | Export the variable and restart IronClaw |
| `File not found` error | CSV not in `agent/` folder | Move your CSV into `GlassBox-AutoML-Agent-main/agent/` |
| `Column not found` error | Wrong `target_col` name | Check column names with: `head -1 agent/your_file.csv` |
| Skill listed but never triggers | Empty activation block / no keyword match | Say "automl" or "glassbox autofit" explicitly |
| `mcp` Python package missing | Not installed | `pip install mcp>=1.0.0` |

---

## Part 7 — Updating

```bash
# Update IronClaw itself:
ironclaw-update

# Update GlassBox skill after editing SKILL.md:
cp skills/glassbox-automl/SKILL.md ~/.ironclaw/skills/glassbox-automl/SKILL.md
# Then restart IronClaw — skills are re-indexed on startup.

# Update the MCP server registration after changing run_server.py:
ironclaw mcp remove glassbox
# Then re-run the ironclaw mcp add command from Part 3.
```
