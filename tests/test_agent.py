"""Comprehensive tests for the self-improving agent."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.core.models import (
    AgentConfig, AgentRole, AgentState, BenchmarkResult,
    BenchmarkSuite, EvolutionEntry, PRStatus, PullRequest,
    ReviewFinding, ReviewResult, Severity,
)
from src.core.providers import MockProvider, TokenUsage, AnthropicProvider, BedrockProvider
from src.core.loop import ImprovementLoop
from src.core.reporting import (
    GenerationReport, PhaseTimestamp, LLMCallRecord,
    load_report, load_all_reports, format_report, format_dashboard,
)
from src.core.config import (
    AgentSettings, load_config, validate_config, format_config,
)
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

    def test_import_check_benchmark(self):
        """Import check benchmark should compile all .py files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src"
            src.mkdir()
            (src / "__init__.py").write_text("")
            (src / "valid.py").write_text("x = 1\ndef foo(): return x\n")
            runner = BenchmarkRunner(root)
            result = runner._bench_import_check()
            assert result.benchmark_name == "import_check"
            assert result.score == 100.0
            assert result.passed is True
            assert result.details["importable"] == 2

    def test_import_check_with_syntax_error(self):
        """Import check should catch syntax errors and reduce score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src"
            src.mkdir()
            (src / "__init__.py").write_text("")
            (src / "good.py").write_text("x = 1\n")
            (src / "bad.py").write_text("def broken(\n")  # syntax error
            runner = BenchmarkRunner(root)
            result = runner._bench_import_check()
            assert result.benchmark_name == "import_check"
            assert result.score < 100.0
            assert result.passed is False
            assert len(result.details["failures"]) == 1

    def test_import_check_no_src_dir(self):
        """Import check should pass gracefully when no src/ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runner = BenchmarkRunner(root)
            result = runner._bench_import_check()
            assert result.score == 100.0
            assert result.passed is True

    def test_security_scan_clean(self):
        """Security scan should pass for clean code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src"
            src.mkdir()
            (src / "safe.py").write_text("def add(a, b):\n    return a + b\n")
            runner = BenchmarkRunner(root)
            result = runner._bench_security_scan()
            assert result.benchmark_name == "security_scan"
            assert result.score == 100.0
            assert result.passed is True

    def test_security_scan_finds_eval(self):
        """Security scan should detect eval() usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src"
            src.mkdir()
            (src / "dangerous.py").write_text(
                "def run_code(code):\n    return eval(code)\n"
            )
            runner = BenchmarkRunner(root)
            result = runner._bench_security_scan()
            assert result.score < 100.0
            assert result.passed is False
            assert result.details["findings_count"] >= 1

    def test_security_scan_ignores_comments(self):
        """Security scan should not flag eval() in comments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src"
            src.mkdir()
            (src / "commented.py").write_text(
                "# Don't use eval() in production\n"
                "x = 1\n"
            )
            runner = BenchmarkRunner(root)
            result = runner._bench_security_scan()
            assert result.score == 100.0
            assert result.passed is True

    def test_security_scan_no_src_dir(self):
        """Security scan should pass when no src/ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runner = BenchmarkRunner(root)
            result = runner._bench_security_scan()
            assert result.score == 100.0
            assert result.passed is True

    def test_parse_pytest_stdout(self):
        """Test fallback pytest stdout parser."""
        output = "====== 5 passed, 2 failed in 1.23s ======\n"
        passed, failed = BenchmarkRunner._parse_pytest_stdout(output)
        assert passed == 5
        assert failed == 2

    def test_parse_pytest_stdout_only_passed(self):
        """Test fallback parser with only passed tests."""
        output = "====== 10 passed in 0.5s ======\n"
        passed, failed = BenchmarkRunner._parse_pytest_stdout(output)
        assert passed == 10
        assert failed == 0

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


# ─── JSON Parsing Tests ──────────────────────────────────────────

class TestJsonParsing:
    """Tests for robust JSON response parsing (Task 1)."""

    def test_parse_clean_json(self):
        """Parse a clean JSON string with no wrapping."""
        result = ImprovementLoop._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_fenced_with_lang(self):
        """Parse JSON wrapped in ```json ... ``` fences."""
        text = 'Here is the plan:\n```json\n{"title": "test"}\n```\nDone!'
        result = ImprovementLoop._parse_json_response(text)
        assert result == {"title": "test"}

    def test_parse_json_fenced_without_lang(self):
        """Parse JSON wrapped in ``` ... ``` fences (no language tag)."""
        text = 'Result:\n```\n{"title": "test", "score": 1.0}\n```'
        result = ImprovementLoop._parse_json_response(text)
        assert result == {"title": "test", "score": 1.0}

    def test_parse_json_with_trailing_text(self):
        """Parse JSON followed by explanation text."""
        text = '{"approved": true, "score": 0.9}\n\nThis looks good because...'
        result = ImprovementLoop._parse_json_response(text)
        assert result["approved"] is True
        assert result["score"] == 0.9

    def test_parse_json_with_leading_text(self):
        """Parse JSON preceded by explanation text."""
        text = 'Here is my analysis:\n\n{"findings": [], "score": 0.8}'
        result = ImprovementLoop._parse_json_response(text)
        assert result["findings"] == []
        assert result["score"] == 0.8

    def test_parse_multiple_json_blocks(self):
        """When multiple JSON blocks exist, take the first valid one."""
        text = (
            'First block:\n```json\n{"title": "first"}\n```\n\n'
            'Second block:\n```json\n{"title": "second"}\n```'
        )
        result = ImprovementLoop._parse_json_response(text)
        assert result["title"] == "first"

    def test_parse_completely_missing_json(self):
        """Return structured error when no JSON found at all."""
        text = "I don't have any JSON for you, sorry."
        result = ImprovementLoop._parse_json_response(text)
        assert "error" in result
        assert "raw" in result

    def test_parse_nested_json(self):
        """Parse JSON with nested objects."""
        data = {"changes": [{"file": "a.py", "content": "x = {'nested': True}"}]}
        text = f"```json\n{json.dumps(data)}\n```"
        result = ImprovementLoop._parse_json_response(text)
        assert result["changes"][0]["file"] == "a.py"

    def test_parse_json_with_newlines_in_strings(self):
        """Parse JSON containing escaped newlines in string values."""
        data = {"content": "line1\\nline2\\nline3"}
        text = json.dumps(data)
        result = ImprovementLoop._parse_json_response(text)
        assert "line1" in result["content"]

    def test_parse_json_embedded_in_markdown(self):
        """Parse JSON embedded in a longer markdown response."""
        text = (
            "## Analysis\n\n"
            "Based on the code review, here are my findings:\n\n"
            '{"approved": false, "score": 0.3, "findings": ['
            '{"severity": "high", "description": "Missing tests"}]}\n\n'
            "### Recommendations\n"
            "You should add more tests."
        )
        result = ImprovementLoop._parse_json_response(text)
        assert result["approved"] is False
        assert len(result["findings"]) == 1

    def test_parse_empty_response(self):
        """Handle empty response gracefully."""
        result = ImprovementLoop._parse_json_response("")
        assert "error" in result

    def test_parse_json_array_not_object(self):
        """Handle response that is a JSON array — wraps it or returns error."""
        text = '[{"item": 1}, {"item": 2}]'
        result = ImprovementLoop._parse_json_response(text)
        # Should return error since we expect dict, or extract first object
        # The brace matcher finds the first { } which is {"item": 1}
        assert "item" in result or "error" in result

    def test_extract_first_json_object(self):
        """Direct test of _extract_first_json_object helper."""
        text = 'prefix {"a": 1, "b": {"c": 2}} suffix'
        result = ImprovementLoop._extract_first_json_object(text)
        assert result == {"a": 1, "b": {"c": 2}}

    def test_extract_first_json_object_none(self):
        """Returns None when no valid JSON object found."""
        result = ImprovementLoop._extract_first_json_object("no json here")
        assert result is None


# ─── Schema Validation Tests ─────────────────────────────────────

class TestSchemaValidation:
    """Tests for _validate_response_schema (Task 1)."""

    def test_validate_architect_valid(self):
        """Valid architect response passes validation."""
        data = {"title": "add-tests", "files_to_modify": ["src/a.py"]}
        result = ImprovementLoop._validate_response_schema(data, "architect")
        assert result["title"] == "add-tests"
        assert "_schema_error" not in result

    def test_validate_architect_missing_title(self):
        """Architect response without title gets safe default."""
        data = {"files_to_modify": ["src/a.py"]}
        result = ImprovementLoop._validate_response_schema(data, "architect")
        assert result["title"] == "invalid-plan"
        assert result["_schema_error"] is True

    def test_validate_builder_valid(self):
        """Valid builder response passes validation."""
        data = {"changes": [{"file": "a.py", "content": "x = 1"}]}
        result = ImprovementLoop._validate_response_schema(data, "builder")
        assert len(result["changes"]) == 1
        assert "_schema_error" not in result

    def test_validate_builder_missing_changes(self):
        """Builder response without changes list gets safe default."""
        data = {"description": "did stuff"}
        result = ImprovementLoop._validate_response_schema(data, "builder")
        assert result["changes"] == []
        assert result["_schema_error"] is True

    def test_validate_builder_bad_change_entry(self):
        """Builder response with bad change entries gets safe default."""
        data = {"changes": [{"file": "a.py"}]}  # missing 'content'
        result = ImprovementLoop._validate_response_schema(data, "builder")
        assert result["changes"] == []
        assert result["_schema_error"] is True

    def test_validate_reviewer_valid(self):
        """Valid reviewer response passes validation."""
        data = {"approved": True, "score": 0.85, "findings": []}
        result = ImprovementLoop._validate_response_schema(data, "reviewer")
        assert result["approved"] is True
        assert "_schema_error" not in result

    def test_validate_reviewer_missing_approved(self):
        """Reviewer response without approved bool gets safe default."""
        data = {"score": 0.5, "findings": []}
        result = ImprovementLoop._validate_response_schema(data, "reviewer")
        assert result["approved"] is False
        assert result["_schema_error"] is True

    def test_validate_reviewer_bad_score_type(self):
        """Reviewer with string score gets safe default."""
        data = {"approved": True, "score": "high", "findings": []}
        result = ImprovementLoop._validate_response_schema(data, "reviewer")
        assert result["approved"] is False
        assert result["_schema_error"] is True

    def test_validate_unknown_persona_passes_through(self):
        """Unknown persona names pass data through unchanged."""
        data = {"anything": "goes"}
        result = ImprovementLoop._validate_response_schema(data, "unknown")
        assert result == data

    def test_validate_error_response_passes_through(self):
        """Error responses from failed parsing pass through."""
        data = {"error": "Failed to parse", "raw": "garbage"}
        result = ImprovementLoop._validate_response_schema(data, "architect")
        assert result == data


# ─── Error Recovery Tests (Task 2) ──────────────────────────────

def _make_test_project(root: Path) -> None:
    """Create a minimal project structure for testing."""
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
    (tests / "test_basic.py").write_text("def test_true():\n    assert True\n")


class TestErrorRecovery:
    """Tests for robust error recovery in the generation loop (Task 2)."""

    @pytest.mark.asyncio
    async def test_planning_failure_records_entry(self):
        """If planning fails, record a failed EvolutionEntry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_test_project(root)
            GitOps(root)

            class FailingMock:
                call_count = 0
                call_log: list = []

                async def complete(self, system_prompt, user_message,
                                   temperature=0.7, max_tokens=4096):
                    self.call_count += 1
                    if "Architect" in system_prompt:
                        raise ConnectionError("LLM timeout during planning")
                    return json.dumps({"response": "ok"})

            config = AgentConfig(project_root=root, auto_merge=True,
                                 benchmark_threshold=-50.0)
            loop = ImprovementLoop(config, FailingMock(), root)
            entry = await loop.run_generation()

            assert entry.generation == 1
            assert entry.accepted is False
            assert "FAILED:plan" in entry.action
            assert "LLM timeout" in entry.rationale

    @pytest.mark.asyncio
    async def test_implement_failure_cleans_up_branch(self):
        """If implementation fails, clean up the orphan branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_test_project(root)
            git = GitOps(root)

            class FailOnBuildMock:
                call_log: list = []

                async def complete(self, system_prompt, user_message,
                                   temperature=0.7, max_tokens=4096):
                    if "Builder" in system_prompt:
                        raise RuntimeError("Builder crashed")
                    return json.dumps(MockProvider._mock_architect_response())

            config = AgentConfig(project_root=root, auto_merge=True,
                                 benchmark_threshold=-50.0)
            loop = ImprovementLoop(config, FailOnBuildMock(), root)
            entry = await loop.run_generation()

            assert entry.accepted is False
            assert "FAILED:implement" in entry.action
            # Should be back on main branch
            assert git.current_branch in ("main", "master")

    @pytest.mark.asyncio
    async def test_review_failure_records_entry(self):
        """If review fails, record a failed entry and clean up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_test_project(root)
            GitOps(root)

            class FailOnReviewMock:
                call_log: list = []

                async def complete(self, system_prompt, user_message,
                                   temperature=0.7, max_tokens=4096):
                    if "Reviewer" in system_prompt:
                        raise TimeoutError("Review timed out")
                    if "Architect" in system_prompt:
                        return json.dumps(MockProvider._mock_architect_response())
                    if "Builder" in system_prompt:
                        return json.dumps(MockProvider._mock_builder_response())
                    return json.dumps({"response": "ok"})

            config = AgentConfig(project_root=root, auto_merge=True,
                                 benchmark_threshold=-50.0)
            loop = ImprovementLoop(config, FailOnReviewMock(), root)
            entry = await loop.run_generation()

            assert entry.accepted is False
            assert "FAILED:review" in entry.action

    @pytest.mark.asyncio
    async def test_dry_run_no_file_writes(self):
        """Dry run mode should skip file writes and git commits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_test_project(root)
            GitOps(root)

            config = AgentConfig(project_root=root, auto_merge=True,
                                 benchmark_threshold=-50.0)
            mock = MockProvider()
            loop = ImprovementLoop(config, mock, root, dry_run=True)
            entry = await loop.run_generation()

            assert entry.generation == 1
            # Files from mock builder should NOT have been written
            assert not (root / "src" / "__init__.py").read_text().startswith('"""Self-improving')

    @pytest.mark.asyncio
    async def test_cleanup_on_failure_returns_to_main(self):
        """_cleanup_on_failure should return to main branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_test_project(root)
            git = GitOps(root)
            config = AgentConfig(project_root=root)
            mock = MockProvider()
            loop = ImprovementLoop(config, mock, root)

            branch = git.create_improvement_branch("test-cleanup")
            assert git.current_branch == branch

            loop._cleanup_on_failure(branch)
            assert git.current_branch in ("main", "master")

    def test_benchmark_timeout(self):
        """Benchmarks that exceed timeout return a failing result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src"
            src.mkdir()
            (src / "simple.py").write_text("x = 1\n")

            runner = BenchmarkRunner(root, benchmark_timeout=1)

            def slow_benchmark():
                time.sleep(10)
                return BenchmarkResult("slow", 100, 100, True, 10.0)

            from src.benchmarks.runner import BenchmarkDef
            runner.benchmarks = [
                BenchmarkDef("slow", "test", 100.0, slow_benchmark)
            ]
            suite = runner.run_all()
            assert len(suite.results) == 1
            assert suite.results[0].passed is False
            assert "Timed out" in suite.results[0].details.get("error", "")


# ─── Provider Tests (Task 3) ─────────────────────────────────────

class TestTokenUsage:
    """Tests for token tracking and cost calculation."""

    def test_cost_calculation(self):
        """Verify cost calculation with known values."""
        usage = TokenUsage()
        usage.record(1_000_000, 100_000, 5.0)
        # 1M input * $3/MTok = $3.00, 100K output * $15/MTok = $1.50
        assert abs(usage.total_cost - 4.50) < 0.01

    def test_multiple_records(self):
        """Track multiple calls."""
        usage = TokenUsage()
        usage.record(500, 100, 1.0)
        usage.record(500, 100, 2.0)
        assert usage.total_calls == 2
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 200
        assert usage.total_latency_seconds == 3.0

    def test_summary(self):
        """Summary dict has expected keys."""
        usage = TokenUsage()
        usage.record(100, 50, 1.0)
        summary = usage.summary()
        assert "total_calls" in summary
        assert "total_cost_usd" in summary
        assert summary["total_calls"] == 1

    def test_mock_provider_has_usage(self):
        """MockProvider should have a usage tracker."""
        mock = MockProvider()
        assert hasattr(mock, "usage")
        assert isinstance(mock.usage, TokenUsage)

    @pytest.mark.asyncio
    async def test_anthropic_retry_on_rate_limit(self):
        """AnthropicProvider retries on rate limit errors."""
        import anthropic

        provider = AnthropicProvider(api_key="test-key")
        provider.RETRY_DELAYS = [0, 0, 0]  # No actual delays in test

        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise anthropic.RateLimitError(
                    message="rate limited",
                    response=MagicMock(status_code=429),
                    body={"error": {"message": "rate limited"}},
                )

            mock_resp = MagicMock()
            mock_resp.content = [MagicMock(text='{"result": "ok"}')]
            mock_resp.usage = MagicMock(input_tokens=100, output_tokens=50)
            return mock_resp

        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        provider._client = mock_client

        result = await provider.complete("system", "user")
        assert call_count == 3
        assert "ok" in result

    @pytest.mark.asyncio
    async def test_anthropic_no_retry_on_auth_error(self):
        """AnthropicProvider fails fast on auth errors."""
        import anthropic

        provider = AnthropicProvider(api_key="bad-key")
        provider.RETRY_DELAYS = [0, 0, 0]

        async def mock_create(**kwargs):
            raise anthropic.AuthenticationError(
                message="invalid key",
                response=MagicMock(status_code=401),
                body={"error": {"message": "invalid key"}},
            )

        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        provider._client = mock_client

        with pytest.raises(anthropic.AuthenticationError):
            await provider.complete("system", "user")


# ─── Reporting Tests (Task 6) ────────────────────────────────────

class TestGenerationReport:
    """Tests for generation reporting and dashboard."""

    def test_report_creation(self):
        """Create a report and verify fields."""
        report = GenerationReport(generation=1)
        assert report.generation == 1
        assert report.started_at != ""
        assert report.finished_at == ""

    def test_phase_tracking(self):
        """Track phases with start/end timestamps."""
        report = GenerationReport(generation=1)
        phase = report.start_phase("benchmark_baseline")
        assert phase.phase == "benchmark_baseline"
        assert phase.start != ""

        report.end_phase(phase)
        assert phase.end != ""
        assert phase.duration_seconds >= 0
        assert len(report.phases) == 1

    def test_phase_error_tracking(self):
        """Track phase errors."""
        report = GenerationReport(generation=1)
        phase = report.start_phase("plan")
        report.end_phase(phase, error="LLM timeout")
        assert phase.error == "LLM timeout"
        assert len(report.errors) == 1
        assert "plan: LLM timeout" in report.errors[0]

    def test_llm_call_recording(self):
        """Record LLM calls with token counts and cost."""
        report = GenerationReport(generation=1)
        report.record_llm_call("architect", 1000, 500, 2.5, 0.01)
        report.record_llm_call("builder", 2000, 1000, 3.0, 0.02)
        assert len(report.llm_calls) == 2
        assert report.total_input_tokens == 3000
        assert report.total_output_tokens == 1500
        assert abs(report.total_cost_usd - 0.03) < 0.001

    def test_report_finalize(self):
        """Finalize sets end time and duration."""
        report = GenerationReport(generation=1)
        report.finalize()
        assert report.finished_at != ""
        assert report.duration_seconds >= 0

    def test_report_save_load_roundtrip(self):
        """Save and load a report, verify data survives."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir) / "reports"

            report = GenerationReport(generation=5)
            report.action = "add-type-hints"
            report.accepted = True
            report.total_benchmark_delta = 12.5
            report.record_llm_call("architect", 100, 50, 1.0, 0.001)
            phase = report.start_phase("plan")
            report.end_phase(phase)
            report.finalize()

            path = report.save(reports_dir)
            assert path.exists()

            loaded = load_report(path)
            assert loaded.generation == 5
            assert loaded.action == "add-type-hints"
            assert loaded.accepted is True
            assert loaded.total_benchmark_delta == 12.5
            assert len(loaded.llm_calls) == 1
            assert len(loaded.phases) == 1

    def test_load_all_reports(self):
        """Load multiple reports from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir) / "reports"
            reports_dir.mkdir()

            for i in range(3):
                r = GenerationReport(generation=i + 1)
                r.accepted = i % 2 == 0
                r.finalize()
                r.save(reports_dir)

            reports = load_all_reports(reports_dir)
            assert len(reports) == 3
            assert reports[0].generation == 1
            assert reports[2].generation == 3

    def test_format_report_output(self):
        """format_report produces readable output."""
        report = GenerationReport(generation=3)
        report.action = "fix-imports"
        report.accepted = True
        report.total_benchmark_delta = 5.0
        report.benchmark_deltas = {"test_pass_rate": 3.0, "lint_score": 2.0}
        report.review_approved = True
        report.review_score = 0.85
        report.files_changed = ["src/core/loop.py"]
        report.finalize()

        output = format_report(report)
        assert "Generation 3" in output
        assert "fix-imports" in output
        assert "ACCEPTED" in output
        assert "test_pass_rate" in output
        assert "src/core/loop.py" in output

    def test_format_dashboard_output(self):
        """format_dashboard produces readable output."""
        reports = []
        for i in range(5):
            r = GenerationReport(generation=i + 1)
            r.accepted = i % 2 == 0
            r.total_benchmark_delta = float(i)
            r.total_cost_usd = 0.01 * (i + 1)
            r.duration_seconds = 10.0
            r.decision_reason = "delta positive" if r.accepted else "regression"
            reports.append(r)

        output = format_dashboard(reports)
        assert "Dashboard" in output
        assert "Generations:" in output
        assert "Accepted:" in output
        assert "Total cost:" in output

    def test_format_dashboard_empty(self):
        """Dashboard handles empty report list gracefully."""
        output = format_dashboard([])
        assert "No generation reports" in output


# ─── Configuration Tests (Task 7) ────────────────────────────────

class TestConfiguration:
    """Tests for configuration loading, merging, and validation."""

    def test_default_settings(self):
        """Default settings have reasonable values."""
        settings = AgentSettings()
        assert settings.model == "claude-sonnet-4-5-20250929"
        assert settings.provider == "anthropic"
        assert settings.max_generations == 100
        assert settings.auto_merge is False
        assert len(settings.benchmarks_enabled) >= 7

    def test_load_config_from_toml(self):
        """Load config from a TOML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "agent.toml").write_text(
                '[agent]\n'
                'model = "claude-opus-4-6"\n'
                'max_generations = 50\n'
                'auto_merge = true\n'
                '\n'
                '[benchmarks]\n'
                'timeout_seconds = 90\n'
            )
            settings = load_config(root)
            assert settings.model == "claude-opus-4-6"
            assert settings.max_generations == 50
            assert settings.auto_merge is True
            assert settings.benchmarks_timeout == 90

    def test_load_config_no_file(self):
        """Load defaults when no config file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = load_config(Path(tmpdir))
            assert settings.model == "claude-sonnet-4-5-20250929"

    def test_cli_overrides(self):
        """CLI args override file config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "agent.toml").write_text(
                '[agent]\nmodel = "claude-opus-4-6"\nprovider = "anthropic"\n'
            )
            settings = load_config(root, cli_overrides={
                "model": "claude-haiku-3-5",
                "provider": "mock",
            })
            assert settings.model == "claude-haiku-3-5"
            assert settings.provider == "mock"

    def test_env_overrides(self):
        """Environment variables override everything."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original = os.environ.get("AGENT_MODEL")
            try:
                os.environ["AGENT_MODEL"] = "env-model"
                settings = load_config(root)
                assert settings.model == "env-model"
            finally:
                if original is None:
                    os.environ.pop("AGENT_MODEL", None)
                else:
                    os.environ["AGENT_MODEL"] = original

    def test_validate_valid_config(self):
        """Valid config produces no issues."""
        settings = AgentSettings()
        issues = validate_config(settings)
        assert issues == []

    def test_validate_invalid_provider(self):
        """Invalid provider is flagged."""
        settings = AgentSettings(provider="invalid")
        issues = validate_config(settings)
        assert any("provider" in i for i in issues)

    def test_validate_invalid_max_gen(self):
        """max_generations < 1 is flagged."""
        settings = AgentSettings(max_generations=0)
        issues = validate_config(settings)
        assert any("max_generations" in i for i in issues)

    def test_format_config_output(self):
        """format_config produces readable output."""
        settings = AgentSettings()
        output = format_config(settings)
        assert "[agent]" in output
        assert "anthropic" in output
        assert "[benchmarks]" in output
        assert "[safety]" in output
