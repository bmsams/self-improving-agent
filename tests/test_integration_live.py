"""Live integration tests for the self-improving agent.

These tests require a real ANTHROPIC_API_KEY and make actual LLM calls.
They are skipped unless the ANTHROPIC_API_KEY environment variable is set.

Run with: make test-live
Or: python -m pytest tests/test_integration_live.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.core.models import AgentConfig, AgentState, EvolutionEntry
from src.core.providers import AnthropicProvider
from src.core.loop import ImprovementLoop
from src.benchmarks.runner import BenchmarkRunner
from src.git_ops.git_manager import GitOps

# Skip all tests in this module if no API key is set
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live integration tests",
)


def _create_target_project(root: Path) -> None:
    """Create a small target project with intentional issues.

    Issues:
    - A Python file missing type hints
    - A function with no docstring
    - A missing __init__.py in a subpackage
    - A test file with one failing test
    """
    # Project structure
    (root / "README.md").write_text("# Target Project\nA test project for self-improvement.\n")
    (root / "CLAUDE.md").write_text("# Steering\nThis is a test project.\n")
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'target-project'\nversion = '0.1.0'\n"
        "requires-python = '>=3.11'\n\n"
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n"
    )
    (root / ".gitignore").write_text(
        ".benchmark_history.json\n.agent_state.json\n"
        ".coverage\ncoverage.json\n.pytest_cache/\n__pycache__/\n"
        ".junit-results.xml\n"
    )

    # Source code with intentional issues
    src = root / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")

    core = src / "core"
    core.mkdir()
    # Missing __init__.py in core — intentional issue

    # File with missing type hints and no docstring
    (src / "utils.py").write_text(
        "def add_numbers(a, b):\n"
        "    return a + b\n\n"
        "def multiply(x, y):\n"
        "    result = x * y\n"
        "    return result\n\n"
        "def greet(name):\n"
        "    return f'Hello, {name}!'\n"
    )

    # File in core subpackage (will fail import without __init__.py)
    (core / "processor.py").write_text(
        '"""Data processor module."""\n\n'
        "def process(data: list) -> list:\n"
        '    """Process a list of items."""\n'
        "    return [item.strip() for item in data if isinstance(item, str)]\n"
    )

    # Tests directory
    tests = root / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")

    # One passing test
    (tests / "test_utils.py").write_text(
        "from src.utils import add_numbers, multiply, greet\n\n"
        "def test_add_numbers():\n"
        "    assert add_numbers(2, 3) == 5\n\n"
        "def test_multiply():\n"
        "    assert multiply(4, 5) == 20\n\n"
        "def test_greet():\n"
        "    assert greet('World') == 'Hello, World!'\n"
    )

    # One failing test — intentional issue
    (tests / "test_broken.py").write_text(
        "def test_intentionally_broken():\n"
        "    \"\"\"This test is intentionally broken for the agent to fix.\"\"\"\n"
        "    assert 1 + 1 == 3, 'This should be fixed by the agent'\n"
    )


class TestLiveGeneration:
    """Live integration tests that run a real LLM generation."""

    @pytest.mark.asyncio
    async def test_single_generation_completes(self):
        """Run one generation with real LLM and verify it completes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_target_project(root)

            # Initialize git repo
            git = GitOps(root)

            # Verify target project has issues
            runner = BenchmarkRunner(root)
            pre_suite = runner.run_all(git_sha=git.current_sha)
            assert pre_suite.total_score > 0, "Benchmarks should produce non-zero scores"

            # Configure and run
            config = AgentConfig(
                project_root=root,
                auto_merge=True,
                benchmark_threshold=-50.0,  # Lenient for first generation
            )
            llm = AnthropicProvider()
            loop = ImprovementLoop(config, llm, root)

            # Run one generation
            entry = await loop.run_generation()

            # Verify EvolutionEntry
            assert isinstance(entry, EvolutionEntry)
            assert entry.generation == 1
            assert entry.action is not None and len(entry.action) > 0
            assert entry.rationale is not None
            assert entry.timestamp is not None

            # Verify state file
            state_file = root / ".agent_state.json"
            assert state_file.exists(), "State file should be saved"
            state = AgentState.load(state_file)
            assert state.generation >= 1
            assert len(state.evolution_log) >= 1

            # Verify git has commits
            log_result = subprocess.run(
                ["git", "log", "--oneline"],
                cwd=root, capture_output=True, text=True, check=True,
            )
            commits = log_result.stdout.strip().split("\n")
            assert len(commits) >= 1, "Should have at least the initial commit"

            # Verify decision is valid
            assert isinstance(entry.accepted, bool)
            if entry.accepted:
                assert entry.benchmark_delta is not None

            # Verify LLM was actually called
            assert llm.usage.total_calls >= 2, (
                f"Expected at least 2 LLM calls, got {llm.usage.total_calls}"
            )
            assert llm.usage.input_tokens > 0
            assert llm.usage.output_tokens > 0

    @pytest.mark.asyncio
    # Note: live tests may take several minutes due to LLM calls
    async def test_generation_records_benchmark_scores(self):
        """Verify benchmark scores are recorded before and after."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_target_project(root)
            git = GitOps(root)

            config = AgentConfig(
                project_root=root,
                auto_merge=True,
                benchmark_threshold=-50.0,
            )
            llm = AnthropicProvider()
            loop = ImprovementLoop(config, llm, root)

            entry = await loop.run_generation()

            # Benchmark history should exist
            history_file = root / ".benchmark_history.json"
            assert history_file.exists(), "Benchmark history should be saved"
            history = json.loads(history_file.read_text())
            assert len(history) >= 1, "Should have at least one benchmark suite recorded"

    @pytest.mark.asyncio
    # Note: live tests may take several minutes due to LLM calls
    async def test_generation_handles_cost_tracking(self):
        """Verify token usage and cost are tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_target_project(root)
            GitOps(root)

            config = AgentConfig(
                project_root=root,
                auto_merge=True,
                benchmark_threshold=-50.0,
            )
            llm = AnthropicProvider()
            loop = ImprovementLoop(config, llm, root)

            await loop.run_generation()

            # Verify cost tracking
            summary = llm.usage.summary()
            assert summary["total_calls"] >= 2
            assert summary["total_cost_usd"] > 0
            assert summary["avg_latency_s"] > 0


class TestLiveBenchmarks:
    """Test that benchmarks produce meaningful results on the target project."""

    def test_benchmarks_run_on_target(self):
        """All benchmarks should run without crashing on the target project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_target_project(root)
            GitOps(root)

            runner = BenchmarkRunner(root)
            suite = runner.run_all()

            assert len(suite.results) >= 7, (
                f"Expected at least 7 benchmark results, got {len(suite.results)}"
            )

            # Every benchmark should have a name and score
            for result in suite.results:
                assert result.benchmark_name is not None
                assert 0 <= result.score <= result.max_score

            # At least some benchmarks should detect issues
            failing = [r for r in suite.results if not r.passed]
            assert len(failing) >= 1, (
                "Target project has intentional issues — at least one benchmark should fail"
            )
