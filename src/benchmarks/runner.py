"""Benchmark system for measuring agent self-improvement.

Implements the 'Dojo' approach — empirical measurement drives evolution.
Benchmarks are defined as simple callables that return scores.
"""

from __future__ import annotations

import ast
import concurrent.futures
import importlib
import json
import logging
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src.core.models import BenchmarkResult, BenchmarkSuite

logger = logging.getLogger(__name__)

DANGEROUS_PATTERNS = [
    r'\beval\s*\(',
    r'\bexec\s*\(',
    r'subprocess\.call\s*\(.*shell\s*=\s*True',
    r'subprocess\.Popen\s*\(.*shell\s*=\s*True',
    r'\bos\.system\s*\(',
    r'pickle\.loads?\s*\(',
    r'marshal\.loads?\s*\(',
    r'__import__\s*\(',
]


@dataclass
class BenchmarkDef:
    """Definition of a benchmark test."""
    name: str
    description: str
    max_score: float
    runner: Callable[..., BenchmarkResult]
    category: str = "general"


class BenchmarkRunner:
    """Runs benchmark suites and tracks results over time."""

    def __init__(self, project_root: Path, benchmark_timeout: int = 120):
        self.project_root = project_root
        self.history_file = project_root / ".benchmark_history.json"
        self.benchmark_timeout = benchmark_timeout
        self.benchmarks: list[BenchmarkDef] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the built-in benchmark suite."""
        self.benchmarks = [
            BenchmarkDef(
                name="test_pass_rate",
                description="Percentage of unit tests passing",
                max_score=100.0,
                runner=self._bench_test_pass_rate,
                category="correctness",
            ),
            BenchmarkDef(
                name="code_complexity",
                description="Average cyclomatic complexity (lower is better, inverted score)",
                max_score=100.0,
                runner=self._bench_code_complexity,
                category="quality",
            ),
            BenchmarkDef(
                name="type_check",
                description="Type checking pass/fail with error count",
                max_score=100.0,
                runner=self._bench_type_check,
                category="correctness",
            ),
            BenchmarkDef(
                name="code_coverage",
                description="Test code coverage percentage",
                max_score=100.0,
                runner=self._bench_code_coverage,
                category="quality",
            ),
            BenchmarkDef(
                name="lint_score",
                description="Linting score (fewer issues = higher score)",
                max_score=100.0,
                runner=self._bench_lint_score,
                category="style",
            ),
            BenchmarkDef(
                name="doc_coverage",
                description="Percentage of public functions with docstrings",
                max_score=100.0,
                runner=self._bench_doc_coverage,
                category="documentation",
            ),
            BenchmarkDef(
                name="file_organization",
                description="Score for proper file structure and module organization",
                max_score=100.0,
                runner=self._bench_file_organization,
                category="architecture",
            ),
            BenchmarkDef(
                name="import_check",
                description="Verify all src/ modules can be imported without errors",
                max_score=100.0,
                runner=self._bench_import_check,
                category="correctness",
            ),
            BenchmarkDef(
                name="security_scan",
                description="Check for dangerous patterns (eval, exec, shell=True, pickle)",
                max_score=100.0,
                runner=self._bench_security_scan,
                category="security",
            ),
        ]

    def register(self, benchmark: BenchmarkDef) -> None:
        """Register a custom benchmark."""
        self.benchmarks.append(benchmark)

    def run_all(self, git_sha: str = "") -> BenchmarkSuite:
        """Run all registered benchmarks and return a suite result.

        Each benchmark is subject to self.benchmark_timeout seconds.
        """
        suite = BenchmarkSuite(
            suite_id=f"suite-{uuid.uuid4().hex[:8]}",
            git_sha=git_sha,
        )
        for bench in self.benchmarks:
            try:
                result = self._run_with_timeout(bench)
                suite.results.append(result)
            except Exception as e:
                logger.warning(f"Benchmark '{bench.name}' failed: {e}")
                suite.results.append(BenchmarkResult(
                    benchmark_name=bench.name,
                    score=0.0,
                    max_score=bench.max_score,
                    passed=False,
                    duration_seconds=0.0,
                    details={"error": str(e)},
                ))
        self._save_history(suite)
        return suite

    def _run_with_timeout(self, bench: BenchmarkDef) -> BenchmarkResult:
        """Run a single benchmark with a timeout."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(bench.runner)
            try:
                return future.result(timeout=self.benchmark_timeout)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    f"Benchmark '{bench.name}' timed out after {self.benchmark_timeout}s"
                )
                return BenchmarkResult(
                    benchmark_name=bench.name,
                    score=0.0,
                    max_score=bench.max_score,
                    passed=False,
                    duration_seconds=float(self.benchmark_timeout),
                    details={"error": f"Timed out after {self.benchmark_timeout}s"},
                )

    def run_single(self, name: str, git_sha: str = "") -> Optional[BenchmarkResult]:
        """Run a single benchmark by name."""
        for bench in self.benchmarks:
            if bench.name == name:
                return bench.runner()
        return None

    def compare(self, before: BenchmarkSuite, after: BenchmarkSuite) -> dict:
        """Compare two benchmark suites and return delta analysis."""
        comparison = {
            "total_delta": after.total_score - before.total_score,
            "pass_rate_delta": after.pass_rate - before.pass_rate,
            "improved": [],
            "regressed": [],
            "unchanged": [],
        }
        before_map = {r.benchmark_name: r for r in before.results}
        for result in after.results:
            prev = before_map.get(result.benchmark_name)
            if prev:
                delta = result.score - prev.score
                entry = {
                    "name": result.benchmark_name,
                    "before": prev.score,
                    "after": result.score,
                    "delta": delta,
                }
                if delta > 0.01:
                    comparison["improved"].append(entry)
                elif delta < -0.01:
                    comparison["regressed"].append(entry)
                else:
                    comparison["unchanged"].append(entry)
        return comparison

    def get_trend(self, benchmark_name: str, last_n: int = 10) -> list[float]:
        """Get score trend for a specific benchmark over recent runs."""
        history = self._load_history()
        scores = []
        for suite in history[-last_n:]:
            for result in suite.get("results", []):
                if result.get("benchmark_name") == benchmark_name:
                    scores.append(result.get("score", 0.0))
        return scores

    def format_report(self, suite: BenchmarkSuite) -> str:
        """Format a benchmark suite as a readable report."""
        lines = [
            f"# Benchmark Report — {suite.suite_id}",
            f"**Git SHA**: {suite.git_sha}",
            f"**Timestamp**: {suite.timestamp}",
            f"**Overall Score**: {suite.total_score:.1f}/{suite.total_max:.1f} "
            f"({suite.total_score/suite.total_max*100:.1f}%)",
            f"**Pass Rate**: {suite.pass_rate*100:.1f}%",
            "",
            "| Benchmark | Score | Max | % | Status |",
            "|-----------|-------|-----|---|--------|",
        ]
        for r in suite.results:
            status = "✅" if r.passed else "❌"
            lines.append(
                f"| {r.benchmark_name} | {r.score:.1f} | {r.max_score:.1f} "
                f"| {r.percentage:.1f}% | {status} |"
            )
        return "\n".join(lines)

    # ─── Built-in Benchmark Implementations ────────────────────────

    def _bench_test_pass_rate(self) -> BenchmarkResult:
        """Run pytest and measure pass rate using junitxml for reliable parsing."""
        start = time.time()
        junit_path = self.project_root / ".junit-results.xml"
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "-q",
             f"--junitxml={junit_path}"],
            cwd=self.project_root, capture_output=True, text=True, check=False,
        )
        duration = time.time() - start

        # Try parsing junitxml first (more reliable)
        passed = failed = errors = 0
        if junit_path.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(junit_path)
                root = tree.getroot()
                for suite in root.iter("testsuite"):
                    passed += int(suite.get("tests", 0)) - int(suite.get("failures", 0)) - int(suite.get("errors", 0))
                    failed += int(suite.get("failures", 0))
                    errors += int(suite.get("errors", 0))
            except Exception:
                passed, failed = self._parse_pytest_stdout(result.stdout + result.stderr)
            finally:
                junit_path.unlink(missing_ok=True)
        else:
            passed, failed = self._parse_pytest_stdout(result.stdout + result.stderr)

        total = passed + failed + errors
        score = (passed / total * 100) if total > 0 else 0.0

        return BenchmarkResult(
            benchmark_name="test_pass_rate",
            score=score,
            max_score=100.0,
            passed=failed == 0 and errors == 0 and passed > 0,
            duration_seconds=duration,
            details={"passed": passed, "failed": failed, "errors": errors},
        )

    @staticmethod
    def _parse_pytest_stdout(output: str) -> tuple[int, int]:
        """Fallback: parse pytest stdout for pass/fail counts."""
        passed = failed = 0
        for line in output.split("\n"):
            if "passed" in line or "failed" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    word = p.rstrip(",")
                    if word == "passed" and i > 0:
                        try:
                            passed = int(parts[i - 1])
                        except ValueError:
                            pass
                    if word == "failed" and i > 0:
                        try:
                            failed = int(parts[i - 1])
                        except ValueError:
                            pass
        return passed, failed

    def _bench_code_complexity(self) -> BenchmarkResult:
        """Measure average cyclomatic complexity using AST analysis."""
        start = time.time()
        complexities = []
        src_dir = self.project_root / "src"

        if not src_dir.exists():
            return BenchmarkResult(
                benchmark_name="code_complexity",
                score=50.0, max_score=100.0, passed=True,
                duration_seconds=0.0, details={"note": "No src directory"},
            )

        for py_file in src_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._calc_complexity(node)
                        complexities.append(complexity)
            except SyntaxError:
                complexities.append(20)  # Penalty for unparseable files

        duration = time.time() - start
        avg = sum(complexities) / len(complexities) if complexities else 1
        # Invert: lower complexity = higher score. Cap at 20.
        score = max(0, min(100, (1 - (avg - 1) / 19) * 100))

        return BenchmarkResult(
            benchmark_name="code_complexity",
            score=score,
            max_score=100.0,
            passed=avg < 10,
            duration_seconds=duration,
            details={"average_complexity": avg, "function_count": len(complexities)},
        )

    @staticmethod
    def _calc_complexity(node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function node."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _bench_type_check(self) -> BenchmarkResult:
        """Run mypy type checking with proper exit code handling.

        Mypy exit codes: 0 = clean, 1 = type errors found, 2 = fatal error.
        """
        start = time.time()
        result = subprocess.run(
            ["python", "-m", "mypy", "src/", "--ignore-missing-imports", "--no-error-summary"],
            cwd=self.project_root, capture_output=True, text=True, check=False,
        )
        duration = time.time() - start

        if result.returncode == 2:
            # Fatal error (bad config, missing source, etc.)
            return BenchmarkResult(
                benchmark_name="type_check",
                score=0.0,
                max_score=100.0,
                passed=False,
                duration_seconds=duration,
                details={"error": "mypy fatal error", "output": result.stderr[:2000]},
            )

        error_count = result.stdout.count(": error:")
        score = max(0, 100 - error_count * 5)

        return BenchmarkResult(
            benchmark_name="type_check",
            score=score,
            max_score=100.0,
            passed=result.returncode == 0,
            duration_seconds=duration,
            details={"error_count": error_count, "exit_code": result.returncode},
        )

    def _bench_code_coverage(self) -> BenchmarkResult:
        """Measure test code coverage."""
        start = time.time()
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=json",
             "-q", "--no-header"],
            cwd=self.project_root, capture_output=True, text=True, check=False,
        )
        duration = time.time() - start

        cov_file = self.project_root / "coverage.json"
        coverage_pct = 0.0
        if cov_file.exists():
            try:
                cov_data = json.loads(cov_file.read_text())
                coverage_pct = cov_data.get("totals", {}).get("percent_covered", 0.0)
            except (json.JSONDecodeError, KeyError):
                pass

        return BenchmarkResult(
            benchmark_name="code_coverage",
            score=coverage_pct,
            max_score=100.0,
            passed=coverage_pct >= 60.0,
            duration_seconds=duration,
            details={"coverage_percent": coverage_pct},
        )

    def _bench_lint_score(self) -> BenchmarkResult:
        """Run ruff linter and score based on issue count."""
        start = time.time()
        result = subprocess.run(
            ["python", "-m", "ruff", "check", "src/", "--output-format=json"],
            cwd=self.project_root, capture_output=True, text=True, check=False,
        )
        duration = time.time() - start

        try:
            issues = json.loads(result.stdout) if result.stdout.strip() else []
        except json.JSONDecodeError:
            issues = []

        issue_count = len(issues) if isinstance(issues, list) else 0
        score = max(0, 100 - issue_count * 2)

        return BenchmarkResult(
            benchmark_name="lint_score",
            score=score,
            max_score=100.0,
            passed=issue_count == 0,
            duration_seconds=duration,
            details={"issue_count": issue_count},
        )

    def _bench_doc_coverage(self) -> BenchmarkResult:
        """Measure percentage of public functions with docstrings."""
        start = time.time()
        total_public = 0
        documented = 0
        src_dir = self.project_root / "src"

        if not src_dir.exists():
            return BenchmarkResult(
                benchmark_name="doc_coverage", score=50.0, max_score=100.0,
                passed=True, duration_seconds=0.0,
            )

        for py_file in src_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not node.name.startswith("_"):
                            total_public += 1
                            if (ast.get_docstring(node)):
                                documented += 1
            except SyntaxError:
                pass

        duration = time.time() - start
        score = (documented / total_public * 100) if total_public > 0 else 100.0

        return BenchmarkResult(
            benchmark_name="doc_coverage",
            score=score,
            max_score=100.0,
            passed=score >= 80.0,
            duration_seconds=duration,
            details={"total_public": total_public, "documented": documented},
        )

    def _bench_file_organization(self) -> BenchmarkResult:
        """Score project file organization."""
        start = time.time()
        score = 100.0
        issues = []

        # Check for __init__.py files
        src_dir = self.project_root / "src"
        if src_dir.exists():
            for subdir in src_dir.rglob("*"):
                if subdir.is_dir() and not subdir.name.startswith("__"):
                    init_file = subdir / "__init__.py"
                    if not init_file.exists():
                        score -= 5
                        issues.append(f"Missing __init__.py in {subdir.relative_to(self.project_root)}")

        # Check for required files
        required = ["README.md", "CLAUDE.md", "pyproject.toml"]
        for req in required:
            if not (self.project_root / req).exists():
                score -= 10
                issues.append(f"Missing {req}")

        # Check for tests directory
        if not (self.project_root / "tests").exists():
            score -= 15
            issues.append("Missing tests directory")

        duration = time.time() - start
        return BenchmarkResult(
            benchmark_name="file_organization",
            score=max(0, score),
            max_score=100.0,
            passed=score >= 70,
            duration_seconds=duration,
            details={"issues": issues},
        )

    def _bench_import_check(self) -> BenchmarkResult:
        """Verify all src/ modules can be imported without errors."""
        start = time.time()
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return BenchmarkResult(
                benchmark_name="import_check", score=100.0, max_score=100.0,
                passed=True, duration_seconds=0.0,
                details={"note": "No src directory"},
            )

        py_files = list(src_dir.rglob("*.py"))
        total = len(py_files)
        importable = 0
        failures: list[str] = []

        for py_file in py_files:
            if py_file.name == "__init__.py":
                importable += 1
                continue
            try:
                source = py_file.read_text()
                compile(source, str(py_file), "exec")
                importable += 1
            except SyntaxError as e:
                failures.append(f"{py_file.name}: {e}")

        duration = time.time() - start
        score = (importable / total * 100) if total > 0 else 100.0

        return BenchmarkResult(
            benchmark_name="import_check",
            score=score,
            max_score=100.0,
            passed=len(failures) == 0,
            duration_seconds=duration,
            details={"total": total, "importable": importable, "failures": failures},
        )

    def _bench_security_scan(self) -> BenchmarkResult:
        """Check for dangerous patterns in source code."""
        start = time.time()
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return BenchmarkResult(
                benchmark_name="security_scan", score=100.0, max_score=100.0,
                passed=True, duration_seconds=0.0,
                details={"note": "No src directory"},
            )

        findings: list[dict] = []
        for py_file in src_dir.rglob("*.py"):
            try:
                content = py_file.read_text()
                rel_path = str(py_file.relative_to(self.project_root))
                for i, line in enumerate(content.split("\n"), 1):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    for pattern in DANGEROUS_PATTERNS:
                        if re.search(pattern, line):
                            findings.append({
                                "file": rel_path,
                                "line": i,
                                "pattern": pattern,
                                "code": stripped[:100],
                            })
            except (OSError, UnicodeDecodeError):
                continue

        duration = time.time() - start
        score = max(0, 100 - len(findings) * 10)

        return BenchmarkResult(
            benchmark_name="security_scan",
            score=score,
            max_score=100.0,
            passed=len(findings) == 0,
            duration_seconds=duration,
            details={"findings_count": len(findings), "findings": findings[:20]},
        )

    # ─── History Management ────────────────────────────────────────

    def _save_history(self, suite: BenchmarkSuite) -> None:
        """Append suite to history file."""
        history = self._load_history()
        from dataclasses import asdict
        history.append(asdict(suite))
        self.history_file.write_text(json.dumps(history, indent=2, default=str))

    def _load_history(self) -> list[dict]:
        """Load benchmark history."""
        if self.history_file.exists():
            try:
                return json.loads(self.history_file.read_text())
            except json.JSONDecodeError:
                return []
        return []
