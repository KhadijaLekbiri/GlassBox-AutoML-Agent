# 🧠 GlassBox AutoML Agent (AutoFit Engine)

## 📌 Overview

**GlassBox** is a lightweight, explainable AutoML framework that automatically builds, trains, and evaluates machine learning models from a dataset — while keeping every step transparent.

Unlike traditional *black-box* AutoML systems, GlassBox follows a **glass-box approach**, allowing users to understand:

* ✅ What model was selected
* ✅ Why it was selected
* ✅ Which features influence predictions

It is also designed for seamless integration into AI agent workflows (e.g., **IronClaw Agent**).

---

## ⚙️ Features

* 📊 **Automated Exploratory Data Analysis (EDA)**
* 🧹 **Data cleaning & preprocessing**
* 🤖 **Automatic model selection** (classification & regression)
* 📈 **Model evaluation metrics**
* 🔍 **Feature importance explanations**
* 📦 **Structured JSON output** for APIs & agents
* 🧩 **Agent-ready architecture** (IronClaw compatible)

---

## 🏗️ Project Structure

```
AI-project/
├── agent.py             # IronClaw agent integration
├── test_run.py          # Script to test AutoFit locally
├── data/                # Example datasets
└── README.md
```

---

## 🚀 How It Works

GlassBox follows a fully automated pipeline:

```
Input CSV
   ↓
EDA (Data Profiling)
   ↓
Data Cleaning (missing values, encoding, scaling)
   ↓
Model Selection
   ↓
Training & Evaluation
   ↓
Feature Importance Extraction
   ↓
JSON Output
```

---

## 🔧 Setup Guide

### 1️⃣ Install Dependencies

Make sure you have **Python 3.10+** installed.

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Configure Claude Desktop Integration

Run the following commands in your terminal:

```powershell
New-Item -ItemType Directory -Path "$env:APPDATA\Claude"
notepad "$env:APPDATA\Claude\claude_desktop_config.json"
```

Paste the following configuration into the file:

```json
{
  "mcpServers": {
    "glassbox": {
      "command": "PATH_TO_PYTHON_EXECUTABLE",
      "args": [
        "PATH_TO_YOUR_PROJECT"
      ]
    }
  },
  "preferences": {
    "coworkScheduledTasksEnabled": false,
    "ccdScheduledTasksEnabled": false,
    "coworkWebSearchEnabled": true,
    "sidebarMode": "chat"
  }
}
```

🔁 Replace:

* `PATH_TO_PYTHON_EXECUTABLE` → your Python installation path
* `PATH_TO_YOUR_PROJECT` → your project directory

Save the file.

---

### 3️⃣ Enable GlassBox Connector

1. Open **Claude Desktop**
2. Go to: **File → Settings → Connectors**
3. Find **GlassBox**
4. Click **Configure**
5. Enable **"Always allow AutoFit"**

---

### 4️⃣ Run AutoFit via Prompt

Use the following prompt inside Claude:

```
Use the GlassBox AutoFit tool with csv_path="your_dataset.csv" and target_col="your_target"
```

Example: (make sure your data file is in agent folder)

```
Use the GlassBox AutoFit tool with csv_path="titanic_dataset.csv" and target_col="Survived"
```

---

## 📤 Output Format (JSON)

GlassBox returns a structured JSON response:

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
    "accuracy": 0.79,
    "f1": 0.30,
    "precision": 0.53,
    "recall": 0.21
  },
  "top_features": [
    {
      "feature": "PAY_0",
      "importance": 0.1899
    },
    {
      "feature": "PAY_3",
      "importance": 0.0962
    }
  ]
}
```

---

## 🧠 Key Idea

GlassBox ensures **full transparency in machine learning**:

* 📌 No hidden decisions
* 📌 Clear model reasoning
* 📌 Interpretable feature importance

This makes it ideal for:

* Educational use 🎓
* Explainable AI applications 🔍
* Agent-based automation 🤖

---

## 🔮 Future Improvements

* 🔄 Hyperparameter optimization (Grid / Bayesian search)
* 📊 Advanced visualization dashboards
* ⚡ WASM execution for faster pipelines
* 🧠 Deeper IronClaw Agent integration

---

## 📄 License

MIT License (or specify your license here)

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

* Open issues
* Suggest improvements
* Submit pull requests

---

## ⭐ Final Note

GlassBox is built with one goal:

> **Make AutoML understandable, not just powerful.**
