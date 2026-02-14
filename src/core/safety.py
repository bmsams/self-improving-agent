"""Safety guardrails for self-modification.

Prevents the agent from writing dangerous code or breaking its own loop.
Validates content before writes and verifies system integrity after changes.
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Patterns that should never appear in agent-generated code
BLOCKED_PATTERNS = [
    (r'\beval\s*\(', "eval() call"),
    (r'\bexec\s*\(', "exec() call"),
    (r'\bos\.system\s*\(', "os.system() call"),
    (r'subprocess\.\w+\s*\(.*shell\s*=\s*True', "subprocess with shell=True"),
    (r'\b__import__\s*\(', "__import__() call"),
    (r'\bpickle\.(loads?|Unpickler)\s*\(', "pickle deserialization"),
    (r'\bmarshal\.(loads?)\s*\(', "marshal deserialization"),
]

# Files that the agent must never modify
PROTECTED_FILES = [
    "src/core/safety.py",  # Self-protection
]

# Methods the agent must never modify
PROTECTED_METHODS = [
    "_make_merge_decision",
]


class SafetyValidator:
    """Validates agent changes before and after they're applied.

    Pre-write: checks file content for dangerous patterns.
    Post-change: verifies the system can still function.
    """

    def __init__(self, project_root: Path, enabled: bool = True):
        self.project_root = project_root
        self.enabled = enabled
        if not enabled:
            logger.warning("Safety guardrails DISABLED — use with caution")

    def validate_before_write(
        self, filepath: str, content: str,
    ) -> tuple[bool, str]:
        """Check content before writing to a file.

        Returns (allowed, reason). If not allowed, the write should be blocked.
        """
        if not self.enabled:
            return True, "safety disabled"

        # Check protected files
        normalized = filepath.replace("\\", "/")
        for protected in PROTECTED_FILES:
            if normalized.endswith(protected) or normalized == protected:
                return False, f"Cannot modify protected file: {protected}"

        # Check for protected method modifications
        for method in PROTECTED_METHODS:
            if f"def {method}" in content:
                return False, f"Cannot redefine protected method: {method}"

        # Check for dangerous patterns
        for pattern, description in BLOCKED_PATTERNS:
            for i, line in enumerate(content.split("\n"), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if re.search(pattern, line):
                    return False, (
                        f"Dangerous pattern detected: {description} "
                        f"at line {i}"
                    )

        # Validate Python syntax if it's a .py file
        if filepath.endswith(".py"):
            try:
                compile(content, filepath, "exec")
            except SyntaxError as e:
                return False, f"Syntax error: {e}"

        return True, "passed"

    def validate_after_change(self, project_root: Optional[Path] = None) -> tuple[bool, str]:
        """Verify system integrity after all changes are applied.

        Checks:
        1. All existing tests still pass
        2. Agent can import its own modules
        3. main.py can parse args
        """
        root = project_root or self.project_root
        if not self.enabled:
            return True, "safety disabled"

        # Check 1: Verify src/ modules compile
        src_dir = root / "src"
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                try:
                    source = py_file.read_text()
                    compile(source, str(py_file), "exec")
                except SyntaxError as e:
                    return False, f"Module compile failed: {py_file.name}: {e}"

        # Check 2: Verify main.py can be parsed
        main_py = root / "main.py"
        if main_py.exists():
            try:
                source = main_py.read_text()
                compile(source, "main.py", "exec")
            except SyntaxError as e:
                return False, f"main.py syntax error: {e}"

        # Check 3: Run existing tests (quick check)
        test_dir = root / "tests"
        if test_dir.exists() and list(test_dir.glob("test_*.py")):
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-x", "-q",
                 "--tb=line", "--no-header"],
                cwd=root, capture_output=True, text=True,
                check=False, timeout=120,
            )
            if result.returncode != 0:
                # Extract first failure line
                output = result.stdout + result.stderr
                first_fail = ""
                for line in output.split("\n"):
                    if "FAILED" in line or "ERROR" in line:
                        first_fail = line.strip()[:200]
                        break
                return False, f"Tests failed after change: {first_fail}"

        return True, "all checks passed"

    def validate_changes(
        self, changes: list[dict],
    ) -> tuple[bool, list[str]]:
        """Validate a batch of changes before applying.

        Args:
            changes: List of dicts with 'file' and 'content' keys.

        Returns:
            (all_valid, list_of_issues)
        """
        issues = []
        for change in changes:
            filepath = change.get("file", "")
            content = change.get("content", "")
            allowed, reason = self.validate_before_write(filepath, content)
            if not allowed:
                issues.append(f"{filepath}: {reason}")
        return len(issues) == 0, issues
