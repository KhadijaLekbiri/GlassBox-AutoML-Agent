"""
wasm_audit.py
-------------
Static audit tool — scans all GlassBox source files for patterns that are
incompatible with the IronClaw WASM sandbox.

Run from your repo root:
    python -m agent.wasm_audit

Exit code 0 = clean, 1 = issues found.
"""

import ast
import sys
from pathlib import Path

# ── Banned patterns ───────────────────────────────────────────────────────────

# (module, attr_or_None)  — None means the whole import is banned
BANNED_IMPORTS = {
    "os":           None,
    "sys":          None,
    "subprocess":   None,
    "socket":       None,
    "urllib":       None,
    "requests":     None,
    "http":         None,
    "ftplib":       None,
    "ssl":          None,
    "threading":    None,
    "multiprocessing": None,
    "ctypes":       None,
    "cffi":         None,
    "sklearn":      None,   # no scikit-learn in the core
    "scipy":        None,   # no scipy in the core
    "pandas":       None,   # no pandas — NumPy only
}

BANNED_BUILTINS = {"open", "exec", "eval", "compile", "__import__"}

ALLOWED_DIRS = {"eda", "Preprocessing", "Models", "Optimization", "core", "agent"}


# ── AST visitor ───────────────────────────────────────────────────────────────

class WASMAuditVisitor(ast.NodeVisitor):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.issues   = []

    def _flag(self, node, msg: str):
        self.issues.append(f"  Line {node.lineno}: {msg}")

    # import X
    def visit_Import(self, node):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in BANNED_IMPORTS:
                self._flag(node, f"Banned import: `import {alias.name}`")
        self.generic_visit(node)

    # from X import Y
    def visit_ImportFrom(self, node):
        root = (node.module or "").split(".")[0]
        if root in BANNED_IMPORTS:
            self._flag(node, f"Banned import: `from {node.module} import ...`")
        self.generic_visit(node)

    # open(), exec(), eval() ...
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in BANNED_BUILTINS:
                self._flag(node, f"Banned builtin call: `{node.func.id}(...)`")
        # os.path.join(), subprocess.run() ...
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in BANNED_IMPORTS:
                    self._flag(node,
                        f"Banned call: `{node.func.value.id}.{node.func.attr}(...)`")
        self.generic_visit(node)


# ── File scanner ──────────────────────────────────────────────────────────────

def audit_file(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
        tree   = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return [f"  SyntaxError: {e}"]

    visitor = WASMAuditVisitor(str(path))
    visitor.visit(tree)
    return visitor.issues


def audit_directory(root: Path) -> dict[str, list[str]]:
    results = {}
    for py_file in sorted(root.rglob("*.py")):
        # Skip test files and this script itself
        if any(part.startswith("test") for part in py_file.parts):
            continue
        if py_file.name == "wasm_audit.py":
            continue

        issues = audit_file(py_file)
        if issues:
            results[str(py_file)] = issues
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).resolve().parent.parent
    print("=" * 55)
    print("  GlassBox-AutoML  ·  WASM Compatibility Audit")
    print(f"  Scanning: {repo_root}")
    print("=" * 55)

    all_issues = {}
    for dir_name in ALLOWED_DIRS:
        d = repo_root / dir_name
        if d.exists():
            found = audit_directory(d)
            all_issues.update(found)

    if not all_issues:
        print("\n  ✓  No WASM-incompatible patterns found.\n")
        sys.exit(0)
    else:
        total = sum(len(v) for v in all_issues.values())
        print(f"\n  ✗  Found {total} issue(s) in {len(all_issues)} file(s):\n")
        for filepath, issues in all_issues.items():
            rel = Path(filepath).relative_to(repo_root)
            print(f"\n  [{rel}]")
            for issue in issues:
                print(issue)
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
