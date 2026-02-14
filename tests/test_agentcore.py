"""Tests for AWS Bedrock AgentCore integration.

Tests all 6 services (Memory, Code Interpreter, Observability,
Evaluations, Gateway, Policy) with mock/fallback behavior.
"""

from __future__ import annotations

import json
import pytest
import asyncio

from src.core.agentcore import (
    AgentCoreConfig,
    AgentCoreMemory,
    AgentCoreCodeInterpreter,
    AgentCoreObservability,
    AgentCoreEvaluations,
    AgentCoreGateway,
    AgentCorePolicy,
    AgentCoreServices,
    _NoOpSpan,
)


# ─── Config Tests ──────────────────────────────────────────

class TestAgentCoreConfig:
    def test_defaults(self):
        config = AgentCoreConfig()
        assert config.enabled is True
        assert config.region == "us-east-1"
        assert config.memory_namespace == "self-improving-agent"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("AGENTCORE_ENABLED", "false")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        monkeypatch.setenv("AGENTCORE_MEMORY_NS", "test-agent")
        config = AgentCoreConfig.from_env()
        assert config.enabled is False
        assert config.region == "eu-west-1"
        assert config.memory_namespace == "test-agent"

    def test_disabled_by_default_env(self, monkeypatch):
        monkeypatch.delenv("AGENTCORE_ENABLED", raising=False)
        config = AgentCoreConfig.from_env()
        assert config.enabled is True  # True by default


# ─── Policy Tests ──────────────────────────────────────────

class TestAgentCorePolicy:
    def setup_method(self):
        self.policy = AgentCorePolicy(AgentCoreConfig(enabled=False))

    def test_allows_normal_file_write(self):
        allowed, reason = self.policy.check_action(
            "file_write", "src/utils/helpers.py", "builder"
        )
        assert allowed is True
        assert reason == "Permitted"

    def test_blocks_benchmark_modification(self):
        allowed, reason = self.policy.check_action(
            "file_write", "benchmarks/runner.py", "builder"
        )
        assert allowed is False
        assert "benchmark runner" in reason.lower()

    def test_blocks_test_deletion(self):
        allowed, reason = self.policy.check_action(
            "file_delete", "tests/test_models.py", "builder"
        )
        assert allowed is False
        assert "test files" in reason.lower()

    def test_blocks_steering_by_non_retrospective(self):
        allowed, reason = self.policy.check_action(
            "file_write", "steering/project-rules.md", "builder"
        )
        assert allowed is False
        assert "Retrospective" in reason

    def test_allows_steering_by_retrospective(self):
        allowed, reason = self.policy.check_action(
            "file_write", "steering/project-rules.md", "retrospective"
        )
        assert allowed is True

    def test_blocks_claude_md_by_non_retrospective(self):
        allowed, reason = self.policy.check_action(
            "file_write", "CLAUDE.md", "builder"
        )
        assert allowed is False

    def test_allows_claude_md_by_retrospective(self):
        allowed, reason = self.policy.check_action(
            "file_write", "CLAUDE.md", "retrospective"
        )
        assert allowed is True

    def test_blocks_excessive_changes(self):
        allowed, reason = self.policy.check_action(
            "file_write", "src/new_file.py", "builder",
            context={"files_changed_count": 15},
        )
        assert allowed is False
        assert "10 file changes" in reason

    def test_allows_within_change_limit(self):
        allowed, reason = self.policy.check_action(
            "file_write", "src/new_file.py", "builder",
            context={"files_changed_count": 5},
        )
        assert allowed is True

    def test_blocks_protected_file_deletion(self):
        for path in ["main.py", "CLAUDE.md", "pyproject.toml", "src/core/models.py"]:
            allowed, reason = self.policy.check_action(
                "file_delete", path, "builder"
            )
            assert allowed is False, f"Should block deletion of {path}"

    def test_get_policies_summary(self):
        policies = self.policy.get_policies_summary()
        assert len(policies) == 5
        names = [p["name"] for p in policies]
        assert "no_benchmark_modification" in names
        assert "no_test_deletion" in names
        assert "no_safety_bypass" in names
        assert "change_limit_per_generation" in names
        assert "no_steering_self_modification" in names


# ─── Evaluations Tests ─────────────────────────────────────

class TestAgentCoreEvaluations:
    def setup_method(self):
        self.evals = AgentCoreEvaluations(AgentCoreConfig(enabled=False))

    @pytest.mark.asyncio
    async def test_local_evaluate_good_generation(self):
        scores = await self.evals.evaluate_generation(
            generation=1,
            plan={"title": "Add type hints"},
            implementation={"changes": [{"file": "a.py"}], "test_changes": [{"file": "test_a.py"}]},
            review={"score": 0.9, "approved": True},
            benchmark_comparison={"total_delta": 15, "regressed": []},
        )
        assert scores["correctness"] == 0.9
        assert scores["helpfulness"] == 0.9
        assert scores["safety"] == 1.0
        assert scores["spec_compliance"] == 1.0
        assert scores["overall"] > 0.8

    @pytest.mark.asyncio
    async def test_local_evaluate_bad_generation(self):
        scores = await self.evals.evaluate_generation(
            generation=1,
            plan={"title": "Risky refactor"},
            implementation={"changes": [{"file": "a.py"}], "test_changes": []},
            review={"score": 0.3, "approved": False},
            benchmark_comparison={"total_delta": -20, "regressed": ["test_pass_rate"]},
        )
        assert scores["correctness"] < 0.3  # Low: bad review + regressions
        assert scores["helpfulness"] == 0.1  # Very negative delta
        assert scores["spec_compliance"] == 0.0  # No tests, not approved
        assert scores["overall"] < 0.4

    @pytest.mark.asyncio
    async def test_local_evaluate_mediocre(self):
        scores = await self.evals.evaluate_generation(
            generation=1,
            plan={"title": "Minor cleanup"},
            implementation={"changes": [{"file": "a.py"}], "test_changes": [{"file": "t.py"}]},
            review={"score": 0.6, "approved": False},
            benchmark_comparison={"total_delta": 2, "regressed": []},
        )
        assert 0.3 < scores["overall"] < 0.7


# ─── Memory Tests ──────────────────────────────────────────

class TestAgentCoreMemory:
    def setup_method(self):
        self.memory = AgentCoreMemory(AgentCoreConfig(enabled=False))

    @pytest.mark.asyncio
    async def test_store_generation_outcome_graceful(self):
        """Should not raise even without AWS connection."""
        await self.memory.store_generation_outcome(
            generation=1,
            action="Add docs",
            accepted=True,
            benchmark_delta=5.0,
            review_summary="Good change",
            files_changed=["README.md"],
        )
        # No exception = success

    @pytest.mark.asyncio
    async def test_recall_returns_empty_without_connection(self):
        results = await self.memory.recall_similar_attempts("add type hints")
        assert results == []

    @pytest.mark.asyncio
    async def test_success_patterns_returns_empty(self):
        patterns = await self.memory.get_success_patterns()
        assert patterns == []


# ─── Code Interpreter Tests ────────────────────────────────

class TestAgentCoreCodeInterpreter:
    def setup_method(self):
        self.ci = AgentCoreCodeInterpreter(AgentCoreConfig(enabled=False))

    @pytest.mark.asyncio
    async def test_sandbox_fallback(self):
        result = await self.ci.run_benchmarks_sandboxed("/tmp", "print('hi')")
        assert result.get("fallback") is True

    @pytest.mark.asyncio
    async def test_validate_code_fallback(self):
        result = await self.ci.validate_generated_code("x = 1 + 2")
        assert result.get("fallback") is True
        assert result.get("syntax_valid") is True


# ─── Observability Tests ───────────────────────────────────

class TestAgentCoreObservability:
    def setup_method(self):
        self.obs = AgentCoreObservability(AgentCoreConfig(enabled=False))

    def test_noop_span(self):
        span = _NoOpSpan()
        span.set_attribute("key", "value")  # Should not raise
        span.end()  # Should not raise

    def test_start_generation_span_returns_span(self):
        """start_generation_span returns a span (NoOp or real depending on env)."""
        span = self.obs.start_generation_span(1)
        # If OpenTelemetry is installed, we get a real span; otherwise _NoOpSpan
        assert hasattr(span, "end"), "Span must have an end() method"
        span.end()

    def test_record_methods_on_noop(self):
        span = _NoOpSpan()
        self.obs.record_llm_call(span, "claude", "builder", 100, 200, 500.0)
        self.obs.record_benchmark(span, "test_pass_rate", 90, 100)
        self.obs.record_decision(span, "merged", "all good")
        # All no-ops, should not raise


# ─── Gateway Tests ─────────────────────────────────────────

class TestAgentCoreGateway:
    def setup_method(self):
        self.gw = AgentCoreGateway(AgentCoreConfig(enabled=False))

    def test_register_github_tools(self):
        self.gw.register_github_tools("fake-token")
        tools = self.gw.list_tools()
        assert len(tools) == 2
        names = [t["name"] for t in tools]
        assert "github_create_pr" in names
        assert "github_get_reviews" in names

    @pytest.mark.asyncio
    async def test_invoke_unknown_tool(self):
        result = await self.gw.invoke_tool("nonexistent", {})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invoke_without_gateway_id(self):
        self.gw.register_github_tools("fake-token")
        result = await self.gw.invoke_tool("github_create_pr", {"repo": "test"})
        assert "error" in result  # No gateway_id configured


# ─── Services Integration Tests ────────────────────────────

class TestAgentCoreServices:
    def test_initialization(self):
        config = AgentCoreConfig(enabled=False)
        services = AgentCoreServices(config)
        assert services.memory is not None
        assert services.code_interpreter is not None
        assert services.observability is not None
        assert services.evaluations is not None
        assert services.gateway is not None
        assert services.policy is not None

    def test_health_check(self):
        config = AgentCoreConfig(enabled=False)
        services = AgentCoreServices(config)
        health = services.health_check()
        assert "memory" in health
        assert "code_interpreter" in health
        assert "observability" in health
        assert "evaluations" in health
        assert "gateway" in health
        assert "policy" in health
        # Policy always works locally
        assert health["policy"] is True
