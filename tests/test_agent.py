"""Comprehensive tests for the self-improving agent."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.models import (
    AgentConfig, AgentRole, AgentState, BenchmarkResult,
    BenchmarkSuite, EvolutionEntry, PRStatus, PullRequest,
    ReviewFinding, ReviewResult, Severity,
)
from src.core.providers import MockProvider
from src.agents.personas import get_persona, PERSONAS, AgentRole
from src.benchmarks.runner import BenchmarkRunner
from src.git_ops.git_manager import GitOps


# ─── Model Tests ──────────────────────────────────────────────────

class TestModels:
    """Tests for core data models."""

    def test_benchmark_result_percentage(self):
        """Verify percentage calculation."""
        result = BenchmarkResult(
            benchmark_name="test", score=75.0, max_score=100.0,
            passed=True, duration_seconds=1.0,
        )
        assert result.percentage == 75.0

    def test_benchmark_result_zero_max(self):
        """Handle zero max score gracefully."""
        result = BenchmarkResult(
            benchmark_name="test", score=0, max_score=0,
            passed=False, duration_seconds=0,
        )
        assert result.percentage == 0.0

    def test_benchmark_suite_aggregation(self):
        """Verify suite-level score aggregation."""
        suite = BenchmarkSuite(
            suite_id="test-suite",
            results=[
                BenchmarkResult("a", 80.0, 100.0, True, 1.0),
                BenchmarkResult("b", 60.0, 100.0, True, 1.0),
                BenchmarkResult("c", 40.0, 100.0, False, 1.0),
            ],
        )
        assert suite.total_score == 180.0
        assert suite.total_max == 300.0
        assert abs(suite.pass_rate - 2 / 3) < 0.01

    def test_review_result_blocking(self):
        """Verify blocking detection on critical findings."""
        review = ReviewResult(
            pr_id="PR-001", reviewer_role=AgentRole.REVIEWER,
            approved=False,
            findings=[
                ReviewFinding(
                    severity=Severity.CRITICAL, category="security",
                    file_path="test.py", line_start=1, line_end=5,
                    description="SQL injection", suggestion="Use parameterized queries",
                ),
            ],
        )
        assert review.blocking is True
        assert review.critical_count == 1

    def test_pull_request_benchmark_delta(self):
        """Verify PR benchmark delta calculation."""
        pr = PullRequest(
            pr_id="PR-001", branch_name="improve/test",
            title="Test", description="Test PR",
        )
        pr.benchmark_before = BenchmarkSuite(
            suite_id="before",
            results=[BenchmarkResult("a", 70.0, 100.0, True, 1.0)],
        )
        pr.benchmark_after = BenchmarkSuite(
            suite_id="after",
            results=[BenchmarkResult("a", 85.0, 100.0, True, 1.0)],
        )
        assert pr.benchmark_delta == 15.0

    def test_agent_state_save_load(self):
        """Test state serialization round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            state = AgentState(
                generation=5, total_prs_created=10,
                total_prs_merged=7, total_prs_rejected=3,
                current_best_score=450.0,
            )
            state.save(path)
            loaded = AgentState.load(path)
            assert loaded.generation == 5
            assert loaded.total_prs_merged == 7
            assert loaded.current_best_score == 450.0

    def test_agent_state_save_load_with_evolution_log(self):
        """Test that evolution_log entries survive save/load as EvolutionEntry objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            entry = EvolutionEntry(
                generation=3, action="add type hints",
                target_file="src/utils.py", rationale="Improve correctness",
                benchmark_delta=5.2, accepted=True,
            )
            state = AgentState(
                generation=3, total_prs_created=3,
                total_prs_merged=2, total_prs_rejected=1,
                evolution_log=[entry],
                current_best_score=350.0,
            )
            state.save(path)
            loaded = AgentState.load(path)
            assert len(loaded.evolution_log) == 1
            e = loaded.evolution_log[0]
            assert isinstance(e, EvolutionEntry), f"Expected EvolutionEntry, got {type(e)}"
            assert e.generation == 3
            assert e.action == "add type hints"
            assert e.accepted is True
            assert e.benchmark_delta == 5.2

    def test_evolution_entry_creation(self):
        """Verify evolution entry defaults."""
        entry = EvolutionEntry(
            generation=1, action="add tests",
            target_file="tests/test_new.py",
            rationale="Improve coverage",
        )
        assert entry.accepted is False
        assert entry.benchmark_delta is None
        assert entry.timestamp is not None


# ─── Persona Tests ────────────────────────────────────────────────

class TestPersonas:
    """Tests for agent personas."""

    def test_all_roles_have_personas(self):
        """Every AgentRole should have a corresponding persona."""
        for role in AgentRole:
            persona = get_persona(role)
            assert persona is not None
            assert persona.role == role

    def test_reviewer_has_criteria(self):
        """Reviewer persona must have review criteria."""
        persona = get_persona(AgentRole.REVIEWER)
        assert persona.review_criteria is not None
        assert "MUST" in persona.review_criteria

    def test_builder_high_temperature(self):
        """Builder should have higher temperature for creativity."""
        builder = get_persona(AgentRole.BUILDER)
        reviewer = get_persona(AgentRole.REVIEWER)
        assert builder.temperature > reviewer.temperature

    def test_persona_full_prompt(self):
        """Full prompt should include system prompt and context."""
        persona = get_persona(AgentRole.BUILDER)
        prompt = persona.get_full_prompt(context="Some context here")
        assert "Builder" in prompt
        assert "Some context here" in prompt


# ─── Mock Provider Tests ──────────────────────────────────────────

class TestMockProvider:
    """Tests for the mock LLM provider."""

    @pytest.mark.asyncio
    async def test_mock_detects_architect(self):
        """Mock should return architect-style response."""
        mock = MockProvider()
        result = await mock.complete(
            system_prompt="You are the Architect",
            user_message="Plan an improvement",
        )
        data = json.loads(result)
        assert "requirements" in data
        assert "design" in data

    @pytest.mark.asyncio
    async def test_mock_detects_reviewer(self):
        """Mock should return reviewer-style response."""
        mock = MockProvider()
        result = await mock.complete(
            system_prompt="You are the Reviewer",
            user_message="Review this PR",
        )
        data = json.loads(result)
        assert "approved" in data
        assert "findings" in data

    @pytest.mark.asyncio
    async def test_mock_logs_calls(self):
        """Mock should log all calls for debugging."""
        mock = MockProvider()
        await mock.complete("sys", "user")
        await mock.complete("sys2", "user2")
        assert len(mock.call_log) == 2


# ─── Git Operations Tests ─────────────────────────────────────────

class TestGitOps:
    """Tests for git operations (uses temp directories)."""

    def test_init_creates_repo(self):
        """GitOps should initialize a git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            git = GitOps(root)
            assert (root / ".git").exists()
            assert git.current_branch == "main" or git.current_branch == "master"

    def test_create_branch(self):
        """Should create and switch to improvement branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            git = GitOps(root)
            branch = git.create_improvement_branch("test feature")
            assert branch.startswith("improve/test-feature")
            assert git.current_branch == branch

    def test_commit_changes(self):
        """Should commit file changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            git = GitOps(root)
            (root / "new_file.py").write_text("print('hello')\n")
            sha = git.commit_changes("[agent] feat: add new file")
            assert len(sha) > 0

    def test_create_and_merge_pr(self):
        """Full PR lifecycle: branch → commit → PR → merge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            git = GitOps(root)

            # Create branch and make changes
            branch = git.create_improvement_branch("test merge")
            (root / "feature.py").write_text("# new feature\n")
            git.commit_changes("[agent] feat: add feature")

            # Create and merge PR
            pr = git.create_pr("Test PR", "Testing merge flow")
            assert pr.status == PRStatus.OPEN

            git.merge_pr(pr)
            assert pr.status == PRStatus.MERGED
            assert git.current_branch in ("main", "master")


# ─── Benchmark Tests ──────────────────────────────────────────────

class TestBenchmarks:
    """Tests for the benchmark runner."""

    def test_doc_coverage_benchmark(self):
        """Doc coverage benchmark should score files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src"
            src.mkdir()
            (src / "example.py").write_text(
                'def public_func():\n'
                '    """Has a docstring."""\n'
                '    pass\n\n'
                'def another_func():\n'
                '    pass  # No docstring\n'
            )
            runner = BenchmarkRunner(root)
            result = runner._bench_doc_coverage()
            assert result.benchmark_name == "doc_coverage"
            assert 0 <= result.score <= 100

    def test_file_organization_benchmark(self):
        """File org benchmark should check structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("# Test\n")
            (root / "tests").mkdir()
            runner = BenchmarkRunner(root)
            result = runner._bench_file_organization()
            assert result.benchmark_name == "file_organization"
            # Missing CLAUDE.md and pyproject.toml should reduce score
            assert result.score < 100

    def test_complexity_benchmark(self):
        """Complexity benchmark should analyze AST."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src"
            src.mkdir()
            (src / "simple.py").write_text(
                "def simple():\n    return 1\n"
            )
            runner = BenchmarkRunner(root)
            result = runner._bench_code_complexity()
            assert result.score > 80  # Simple function = low complexity = high score

    def test_suite_comparison(self):
        """Compare two benchmark suites and detect deltas."""
        runner = BenchmarkRunner(Path("."))
        before = BenchmarkSuite(
            suite_id="before",
            results=[
                BenchmarkResult("a", 70.0, 100.0, True, 1.0),
                BenchmarkResult("b", 80.0, 100.0, True, 1.0),
            ],
        )
        after = BenchmarkSuite(
            suite_id="after",
            results=[
                BenchmarkResult("a", 85.0, 100.0, True, 1.0),
                BenchmarkResult("b", 75.0, 100.0, True, 1.0),
            ],
        )
        comp = runner.compare(before, after)
        assert comp["total_delta"] == 10.0
        assert len(comp["improved"]) == 1
        assert len(comp["regressed"]) == 1

    def test_format_report(self):
        """Benchmark report should be human-readable."""
        runner = BenchmarkRunner(Path("."))
        suite = BenchmarkSuite(
            suite_id="test",
            results=[BenchmarkResult("test_a", 90.0, 100.0, True, 0.5)],
        )
        report = runner.format_report(suite)
        assert "test_a" in report
        assert "90.0" in report


# ─── Integration Tests ────────────────────────────────────────────

class TestIntegration:
    """Integration tests for the full improvement loop."""

    @pytest.mark.asyncio
    async def test_full_generation_with_mock(self):
        """Run a complete generation with mock LLM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Set up minimal project structure
            (root / "README.md").write_text("# Test Project\n")
            (root / "CLAUDE.md").write_text("# Steering\n")
            (root / "pyproject.toml").write_text("[project]\nname='test'\n")
            (root / ".gitignore").write_text(
                ".benchmark_history.json\n.agent_state.json\n"
                ".coverage\ncoverage.json\n.pytest_cache/\n__pycache__/\n"
            )
            src = root / "src"
            src.mkdir()
            (src / "__init__.py").write_text("")
            (src / "core").mkdir()
            (src / "core" / "__init__.py").write_text("")
            tests = root / "tests"
            tests.mkdir()
            (tests / "__init__.py").write_text("")
            (tests / "test_basic.py").write_text(
                "def test_true():\n    assert True\n"
            )

            # Initialize git
            git = GitOps(root)

            config = AgentConfig(
                project_root=root,
                auto_merge=True,
                benchmark_threshold=-50.0,  # Very lenient for mock
            )
            mock = MockProvider()

            from src.core.loop import ImprovementLoop
            loop = ImprovementLoop(config, mock, root)
            entry = await loop.run_generation()

            assert entry.generation == 1
            assert entry.action is not None
            assert len(mock.call_log) >= 3  # Plan + Build + Review at minimum
