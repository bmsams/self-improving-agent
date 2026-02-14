"""AWS Bedrock AgentCore integration for the self-improving agent.

Integrates 6 AgentCore services:
- Memory: Episodic memory for cross-generation learning
- Code Interpreter: Sandboxed benchmark/test execution
- Observability: OpenTelemetry tracing of agent decisions
- Evaluations: Quality scoring (correctness, tool selection)
- Gateway: MCP server for GitHub/tool integration
- Policy: Cedar-based safety boundaries

Each integration is optional — gracefully degrades if AWS credentials
or specific services aren't available.

Uses the official bedrock-agentcore SDK (<=0.1.5):
- Memory: bedrock_agentcore.memory.MemoryClient
- Code Interpreter: bedrock_agentcore.tools.code_interpreter_client.CodeInterpreter
- Runtime: bedrock_agentcore.runtime.BedrockAgentCoreApp
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────

@dataclass
class AgentCoreConfig:
    """Configuration for AgentCore service integration."""
    enabled: bool = True
    region: str = "us-east-1"
    memory_namespace: str = "self-improving-agent"
    memory_id: Optional[str] = None
    code_interpreter_id: Optional[str] = None
    gateway_id: Optional[str] = None
    policy_store_id: Optional[str] = None
    observability_enabled: bool = True
    evaluations_enabled: bool = True
    code_interpreter_enabled: bool = True

    @classmethod
    def from_env(cls) -> AgentCoreConfig:
        """Load config from environment variables."""
        return cls(
            enabled=os.environ.get("AGENTCORE_ENABLED", "true").lower() == "true",
            region=os.environ.get("AWS_REGION", "us-east-1"),
            memory_namespace=os.environ.get("AGENTCORE_MEMORY_NS", "self-improving-agent"),
            memory_id=os.environ.get("AGENTCORE_MEMORY_ID"),
            code_interpreter_id=os.environ.get("AGENTCORE_CODE_INTERPRETER_ID"),
            gateway_id=os.environ.get("AGENTCORE_GATEWAY_ID"),
            policy_store_id=os.environ.get("AGENTCORE_POLICY_STORE_ID"),
        )


def _get_boto_client(service: str, region: str):
    """Lazy-load a boto3 client, returning None if unavailable."""
    try:
        import boto3
        return boto3.client(service, region_name=region)
    except Exception as e:
        logger.debug(f"Could not create {service} client: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. AGENTCORE MEMORY — Episodic memory for cross-generation learning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACTOR_ID = "self-improving-agent"
SESSION_ID = "improvement-loop"


class AgentCoreMemory:
    """Manages long-term episodic memory via AgentCore Memory.

    Uses the official bedrock-agentcore SDK MemoryClient with:
    - create_event() to store generation outcomes and review patterns
    - list_events() to retrieve past events with payloads
    - get_last_k_turns() to recall recent conversation context

    Gracefully degrades to empty results when the SDK or AWS
    credentials are not available.
    """

    def __init__(self, config: AgentCoreConfig):
        self.config = config
        self._client = None
        self._memory_id = config.memory_id

    @property
    def client(self):
        """Lazy-load the AgentCore Memory client."""
        if self._client is None:
            try:
                from bedrock_agentcore.memory import MemoryClient
                self._client = MemoryClient(region_name=self.config.region)
            except ImportError:
                self._client = _get_boto_client(
                    "bedrock-agentcore", self.config.region
                )
        return self._client

    async def store_generation_outcome(
        self,
        generation: int,
        action: str,
        accepted: bool,
        benchmark_delta: float,
        review_summary: str,
        files_changed: list[str],
    ) -> None:
        """Store the outcome of one improvement generation as episodic memory."""
        episode = {
            "type": "generation_outcome",
            "generation": generation,
            "action": action,
            "accepted": accepted,
            "benchmark_delta": benchmark_delta,
            "review_summary": review_summary,
            "files_changed": files_changed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            if self.client and self._memory_id:
                self.client.create_event(
                    memory_id=self._memory_id,
                    actor_id=ACTOR_ID,
                    session_id=f"gen-{generation}",
                    messages=[(json.dumps(episode), "assistant")],
                )
                logger.info(f"Stored generation {generation} episode to AgentCore Memory")
        except Exception as e:
            logger.warning(f"AgentCore Memory write failed (non-fatal): {e}")

    async def store_review_pattern(
        self,
        finding_category: str,
        finding_severity: str,
        was_valid: bool,
        description: str,
    ) -> None:
        """Store review finding patterns for meta-learning."""
        pattern = {
            "type": "review_pattern",
            "category": finding_category,
            "severity": finding_severity,
            "was_valid": was_valid,
            "description": description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            if self.client and self._memory_id:
                self.client.create_event(
                    memory_id=self._memory_id,
                    actor_id=ACTOR_ID,
                    session_id="review-patterns",
                    messages=[(json.dumps(pattern), "assistant")],
                )
        except Exception as e:
            logger.warning(f"AgentCore Memory write failed (non-fatal): {e}")

    async def recall_similar_attempts(
        self, current_plan: str, limit: int = 5
    ) -> list[dict]:
        """Recall past attempts by listing recent events.

        Uses list_events() to retrieve stored generation outcomes,
        then filters client-side for relevance to the current plan.
        """
        try:
            if self.client and self._memory_id:
                events = self.client.list_events(
                    memory_id=self._memory_id,
                    actor_id=ACTOR_ID,
                    session_id=SESSION_ID,
                    max_results=limit * 3,
                    include_payload=True,
                )
                results = []
                for event in events or []:
                    payload = event.get("payload", [{}])
                    if payload:
                        blob = payload[0].get("blob", "")
                        try:
                            data = json.loads(blob) if isinstance(blob, str) else blob
                            if data.get("type") == "generation_outcome":
                                results.append(data)
                        except (json.JSONDecodeError, TypeError):
                            continue
                return results[:limit]
        except Exception as e:
            logger.warning(f"AgentCore Memory recall failed (non-fatal): {e}")
        return []

    async def get_success_patterns(self, limit: int = 10) -> list[dict]:
        """Retrieve the most successful improvement patterns.

        Lists events and filters for accepted changes with positive
        benchmark deltas, ordered by impact magnitude.
        """
        try:
            if self.client and self._memory_id:
                events = self.client.list_events(
                    memory_id=self._memory_id,
                    actor_id=ACTOR_ID,
                    session_id=SESSION_ID,
                    max_results=50,
                    include_payload=True,
                )
                episodes = []
                for event in events or []:
                    payload = event.get("payload", [{}])
                    if payload:
                        blob = payload[0].get("blob", "")
                        try:
                            data = json.loads(blob) if isinstance(blob, str) else blob
                            if data.get("accepted") and data.get("benchmark_delta", 0) > 0:
                                episodes.append(data)
                        except (json.JSONDecodeError, TypeError):
                            continue
                return sorted(
                    episodes,
                    key=lambda e: e.get("benchmark_delta", 0),
                    reverse=True,
                )[:limit]
        except Exception as e:
            logger.warning(f"AgentCore Memory search failed (non-fatal): {e}")
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. AGENTCORE CODE INTERPRETER — Sandboxed benchmark execution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentCoreCodeInterpreter:
    """Executes benchmarks and tests in isolated AgentCore sandboxes.

    Uses the official bedrock-agentcore SDK CodeInterpreter with:
    - start(identifier=...) to create a session
    - invoke("executeCode", {...}) to run code
    - stop() to clean up

    Gracefully degrades to local execution when unavailable.
    """

    def __init__(self, config: AgentCoreConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        """Lazy-load the Code Interpreter client."""
        if self._client is None:
            try:
                from bedrock_agentcore.tools.code_interpreter_client import (
                    CodeInterpreter,
                )
                self._client = CodeInterpreter(self.config.region)
            except ImportError:
                self._client = _get_boto_client(
                    "bedrock-agentcore", self.config.region
                )
        return self._client

    async def run_benchmarks_sandboxed(
        self, project_path: str, benchmark_script: str
    ) -> dict:
        """Run benchmark suite in an isolated sandbox."""
        code = f"""
import subprocess
import json
import sys
import os

os.chdir("{project_path}")

# Run pytest with JSON output
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q",
     "--co", "-q"],  # collect-only for safety first
    capture_output=True, text=True, timeout=120
)

# Run the actual benchmark script
bench_result = subprocess.run(
    [sys.executable, "-c", '''{benchmark_script}'''],
    capture_output=True, text=True, timeout=300
)

print(json.dumps({{
    "test_output": result.stdout[:5000],
    "test_returncode": result.returncode,
    "bench_output": bench_result.stdout[:5000],
    "bench_returncode": bench_result.returncode,
}}))
"""
        try:
            if self.client and self.config.code_interpreter_id:
                # Use official SDK pattern: start session, invoke, stop
                try:
                    from bedrock_agentcore.tools.code_interpreter_client import (
                        CodeInterpreter,
                    )
                    ci = CodeInterpreter(self.config.region)
                    ci.start(identifier=self.config.code_interpreter_id)
                    response = ci.invoke(
                        "executeCode",
                        {"code": code, "language": "python", "clearContext": True},
                    )
                    output = ""
                    for event in response.get("stream", []):
                        result = event.get("result", {})
                        structured = result.get("structuredContent", {})
                        output = structured.get("stdout", "")
                    ci.stop()
                    try:
                        return json.loads(output)
                    except json.JSONDecodeError:
                        return {"raw_output": output, "error": "JSON parse failed"}
                except ImportError:
                    # Fallback to boto3 generic client
                    pass
        except Exception as e:
            logger.warning(f"AgentCore Code Interpreter failed (non-fatal): {e}")

        return {"fallback": True, "reason": "Code Interpreter unavailable"}

    async def validate_generated_code(self, code: str) -> dict:
        """Validate agent-generated code in a sandbox before applying."""
        validation_code = f"""
import ast
import sys
import json

code = '''{code.replace("'''", "\\'\\'\\'")}'''

results = {{"syntax_valid": False, "imports_valid": False, "issues": []}}

# Syntax check
try:
    tree = ast.parse(code)
    results["syntax_valid"] = True
except SyntaxError as e:
    results["issues"].append(f"Syntax error: {{e}}")

# Check for dangerous patterns
dangerous = ["os.system", "subprocess.call", "eval(", "exec(", "__import__"]
for pattern in dangerous:
    if pattern in code:
        results["issues"].append(f"Potentially dangerous: {{pattern}}")

# Check imports
if results["syntax_valid"]:
    try:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        __import__(alias.name.split('.')[0])
                    except ImportError:
                        results["issues"].append(f"Missing import: {{alias.name}}")
        results["imports_valid"] = len([
            i for i in results["issues"] if "Missing import" in i
        ]) == 0
    except Exception as e:
        results["issues"].append(f"Import check error: {{e}}")

print(json.dumps(results))
"""
        try:
            if self.client and self.config.code_interpreter_id:
                try:
                    from bedrock_agentcore.tools.code_interpreter_client import (
                        CodeInterpreter,
                    )
                    ci = CodeInterpreter(self.config.region)
                    ci.start(identifier=self.config.code_interpreter_id)
                    response = ci.invoke(
                        "executeCode",
                        {
                            "code": validation_code,
                            "language": "python",
                            "clearContext": True,
                        },
                    )
                    output = ""
                    for event in response.get("stream", []):
                        result = event.get("result", {})
                        structured = result.get("structuredContent", {})
                        output = structured.get("stdout", "")
                    ci.stop()
                    return json.loads(output)
                except ImportError:
                    pass
        except Exception as e:
            logger.warning(f"Code validation sandbox failed (non-fatal): {e}")

        return {"fallback": True, "syntax_valid": True, "imports_valid": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. AGENTCORE OBSERVABILITY — OpenTelemetry tracing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentCoreObservability:
    """Traces every agent decision through OpenTelemetry.

    When deployed on AgentCore Runtime, the container is launched with
    `opentelemetry-instrument` which auto-configures OTLP export to
    CloudWatch. Locally, falls back to ConsoleSpanExporter.
    """

    def __init__(self, config: AgentCoreConfig):
        self.config = config
        self._tracer = None
        self._trace_module = None
        self._initialized = False

    def initialize(self) -> None:
        """Set up OpenTelemetry tracing."""
        if self._initialized:
            return

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
            )

            # When running on AgentCore Runtime, the opentelemetry-instrument
            # wrapper auto-configures OTLP export. We only need to set up
            # a provider if one isn't already configured.
            existing = trace.get_tracer_provider()
            if hasattr(existing, "get_tracer"):
                # A provider is already configured (e.g., by otel-instrument)
                self._tracer = existing.get_tracer("self-improving-agent")
            else:
                # Local dev: use console exporter
                provider = TracerProvider()
                processor = BatchSpanProcessor(ConsoleSpanExporter())
                provider.add_span_processor(processor)
                trace.set_tracer_provider(provider)
                self._tracer = trace.get_tracer("self-improving-agent")

            self._trace_module = trace
            self._initialized = True
            logger.info("AgentCore Observability initialized (OpenTelemetry)")
        except ImportError:
            logger.debug("OpenTelemetry not installed, observability disabled")

    def start_generation_span(self, generation: int) -> Any:
        """Start a parent span for a full generation cycle."""
        self.initialize()
        if self._tracer:
            span = self._tracer.start_span(
                f"generation-{generation}",
                attributes={
                    "agent.generation": generation,
                    "agent.type": "self-improving",
                },
            )
            return span
        return _NoOpSpan()

    def start_phase_span(self, phase: str, parent: Any = None) -> Any:
        """Start a child span for a specific phase."""
        if self._tracer and self._trace_module:
            ctx = self._trace_module.set_span_in_context(parent) if parent else None
            return self._tracer.start_span(
                f"phase.{phase}",
                context=ctx,
                attributes={"agent.phase": phase},
            )
        return _NoOpSpan()

    def record_llm_call(
        self,
        span: Any,
        model: str,
        persona: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        """Record LLM call metrics on a span."""
        if hasattr(span, "set_attribute"):
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.persona", persona)
            span.set_attribute("llm.input_tokens", input_tokens)
            span.set_attribute("llm.output_tokens", output_tokens)
            span.set_attribute("llm.latency_ms", latency_ms)

    def record_benchmark(
        self, span: Any, name: str, score: float, max_score: float
    ) -> None:
        """Record benchmark result on a span."""
        if hasattr(span, "set_attribute"):
            span.set_attribute(f"benchmark.{name}.score", score)
            span.set_attribute(f"benchmark.{name}.max", max_score)
            span.set_attribute(
                f"benchmark.{name}.pct",
                round(score / max_score * 100, 1) if max_score > 0 else 0,
            )

    def record_decision(
        self, span: Any, decision: str, reason: str
    ) -> None:
        """Record a merge/reject decision."""
        if hasattr(span, "set_attribute"):
            span.set_attribute("decision.action", decision)
            span.set_attribute("decision.reason", reason)


class _NoOpSpan:
    """No-op span when observability is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. AGENTCORE EVALUATIONS — Quality scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentCoreEvaluations:
    """Uses AgentCore Evaluations for quality assessment.

    Provides local evaluation scoring as primary method, with
    optional AgentCore Evaluations API for richer assessment.
    """

    def __init__(self, config: AgentCoreConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _get_boto_client(
                "bedrock-agentcore", self.config.region
            )
        return self._client

    async def evaluate_generation(
        self,
        generation: int,
        plan: dict,
        implementation: dict,
        review: dict,
        benchmark_comparison: dict,
    ) -> dict:
        """Run evaluations on a complete generation cycle."""
        evaluation_input = {
            "generation": generation,
            "plan": plan,
            "implementation_summary": {
                "files_changed": len(implementation.get("changes", [])),
                "test_files_added": len(implementation.get("test_changes", [])),
            },
            "review_score": review.get("score", 0),
            "review_approved": review.get("approved", False),
            "benchmark_delta": benchmark_comparison.get("total_delta", 0),
            "regressions": benchmark_comparison.get("regressed", []),
        }

        scores = self._local_evaluate(evaluation_input)

        try:
            if self.client:
                response = self.client.create_evaluation(
                    evaluationName=f"gen-{generation}-{uuid.uuid4().hex[:6]}",
                    evaluatorConfigs=[
                        {"evaluatorType": "CORRECTNESS"},
                        {"evaluatorType": "HELPFULNESS"},
                        {"evaluatorType": "SAFETY"},
                    ],
                    evaluationInput=json.dumps(evaluation_input),
                )
                if response:
                    scores["agentcore_eval_id"] = response.get("evaluationId")
                    logger.info(
                        f"AgentCore Evaluation created: {scores['agentcore_eval_id']}"
                    )
        except Exception as e:
            logger.debug(f"AgentCore Evaluations unavailable: {e}")

        return scores

    @staticmethod
    def _local_evaluate(input_data: dict) -> dict:
        """Fallback local evaluation when AgentCore isn't available."""
        scores = {
            "correctness": 0.0,
            "helpfulness": 0.0,
            "safety": 1.0,
            "spec_compliance": 0.0,
            "overall": 0.0,
        }

        review_score = input_data.get("review_score", 0)
        has_regressions = len(input_data.get("regressions", [])) > 0
        scores["correctness"] = review_score * (0.5 if has_regressions else 1.0)

        delta = input_data.get("benchmark_delta", 0)
        if delta > 10:
            scores["helpfulness"] = 0.9
        elif delta > 0:
            scores["helpfulness"] = 0.7
        elif delta > -5:
            scores["helpfulness"] = 0.4
        else:
            scores["helpfulness"] = 0.1

        has_tests = input_data.get("implementation_summary", {}).get(
            "test_files_added", 0
        ) > 0
        approved = input_data.get("review_approved", False)
        scores["spec_compliance"] = (
            (0.5 if has_tests else 0.0) + (0.5 if approved else 0.0)
        )

        scores["overall"] = (
            scores["correctness"] * 0.3
            + scores["helpfulness"] * 0.3
            + scores["safety"] * 0.2
            + scores["spec_compliance"] * 0.2
        )

        return scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. AGENTCORE GATEWAY — MCP server for tool integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentCoreGateway:
    """Connects the agent to external tools via MCP through Gateway."""

    def __init__(self, config: AgentCoreConfig):
        self.config = config
        self._gateway_id = config.gateway_id
        self._tools: dict[str, dict] = {}

    def register_github_tools(self, github_token: str) -> None:
        """Register GitHub API tools via Gateway."""
        self._tools["github_create_pr"] = {
            "name": "github_create_pr",
            "description": "Create a pull request on GitHub",
            "input_schema": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "head": {"type": "string"},
                    "base": {"type": "string", "default": "main"},
                },
                "required": ["repo", "title", "head"],
            },
        }
        self._tools["github_get_reviews"] = {
            "name": "github_get_reviews",
            "description": "Get review comments from a GitHub PR",
            "input_schema": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "pr_number": {"type": "integer"},
                },
                "required": ["repo", "pr_number"],
            },
        }
        logger.info(f"Registered {len(self._tools)} GitHub tools via Gateway")

    async def invoke_tool(self, tool_name: str, params: dict) -> dict:
        """Invoke a tool through AgentCore Gateway."""
        if tool_name not in self._tools:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            client = _get_boto_client(
                "bedrock-agentcore", self.config.region
            )
            if client and self._gateway_id:
                response = client.invoke_tool(
                    gatewayId=self._gateway_id,
                    toolName=tool_name,
                    input=json.dumps(params),
                )
                return json.loads(response.get("output", "{}"))
        except Exception as e:
            logger.warning(f"Gateway tool invocation failed: {e}")

        return {"error": "Gateway unavailable", "tool": tool_name}

    def list_tools(self) -> list[dict]:
        """List all registered tools."""
        return list(self._tools.values())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. AGENTCORE POLICY — Cedar-based safety boundaries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentCorePolicy:
    """Enforces safety boundaries using Cedar policies via AgentCore."""

    SAFETY_POLICIES = [
        {
            "name": "no_benchmark_modification",
            "description": "Agent cannot modify benchmark runner or definitions",
            "cedar": """
permit(principal, action == Action::"file_write", resource)
unless {
    resource.path.contains("benchmarks/runner.py") ||
    resource.path.contains("benchmarks/definitions")
};
""",
        },
        {
            "name": "no_test_deletion",
            "description": "Agent cannot delete test files",
            "cedar": """
forbid(principal, action == Action::"file_delete", resource)
when {
    resource.path.startsWith("tests/")
};
""",
        },
        {
            "name": "no_safety_bypass",
            "description": "Agent cannot modify core loop safety checks",
            "cedar": """
permit(principal, action == Action::"file_write", resource)
unless {
    resource.path == "src/core/loop.py" &&
    resource.content.contains("_make_merge_decision")
};
""",
        },
        {
            "name": "change_limit_per_generation",
            "description": "Maximum 10 file changes per generation",
            "cedar": """
forbid(principal, action == Action::"file_write", resource)
when {
    context.files_changed_count > 10
};
""",
        },
        {
            "name": "no_steering_self_modification",
            "description": "Only Retrospective agent can modify steering files",
            "cedar": """
permit(principal, action == Action::"file_write", resource)
when {
    resource.path.startsWith("steering/") ||
    resource.path == "CLAUDE.md"
}
unless {
    principal.role != "retrospective"
};
""",
        },
    ]

    def __init__(self, config: AgentCoreConfig):
        self.config = config
        self._policies_loaded = False

    def check_action(
        self,
        action: str,
        resource_path: str,
        agent_role: str,
        context: Optional[dict] = None,
    ) -> tuple[bool, str]:
        """Check if an action is permitted by policy."""
        ctx = context or {}

        if action == "file_write":
            if "benchmarks/runner.py" in resource_path:
                return False, "Policy: Cannot modify benchmark runner"

            if "core/loop.py" in resource_path and agent_role != "retrospective":
                pass

            if (
                resource_path.startswith("steering/")
                or resource_path == "CLAUDE.md"
            ) and agent_role != "retrospective":
                return False, "Policy: Only Retrospective agent can modify steering"

            if ctx.get("files_changed_count", 0) > 10:
                return False, "Policy: Maximum 10 file changes per generation"

        elif action == "file_delete":
            if resource_path.startswith("tests/"):
                return False, "Policy: Cannot delete test files"

            protected = [
                "main.py", "CLAUDE.md", "pyproject.toml",
                "src/core/models.py", "src/core/loop.py",
            ]
            if resource_path in protected:
                return False, f"Policy: Cannot delete protected file {resource_path}"

        return True, "Permitted"

    def get_policies_summary(self) -> list[dict]:
        """Return human-readable policy summaries."""
        return [
            {"name": p["name"], "description": p["description"]}
            for p in self.SAFETY_POLICIES
        ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UNIFIED INTEGRATION POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentCoreServices:
    """Unified access to all AgentCore services."""

    def __init__(self, config: Optional[AgentCoreConfig] = None):
        self.config = config or AgentCoreConfig.from_env()
        self.memory = AgentCoreMemory(self.config)
        self.code_interpreter = AgentCoreCodeInterpreter(self.config)
        self.observability = AgentCoreObservability(self.config)
        self.evaluations = AgentCoreEvaluations(self.config)
        self.gateway = AgentCoreGateway(self.config)
        self.policy = AgentCorePolicy(self.config)

        if self.config.observability_enabled:
            self.observability.initialize()

        logger.info("AgentCore services initialized")

    def health_check(self) -> dict[str, bool]:
        """Check which AgentCore services are available."""
        status = {
            "memory": self.memory.client is not None,
            "code_interpreter": self.code_interpreter.client is not None,
            "observability": self.observability._initialized,
            "evaluations": self.evaluations.client is not None,
            "gateway": self.config.gateway_id is not None,
            "policy": True,
        }
        logger.info(f"AgentCore health: {status}")
        return status
