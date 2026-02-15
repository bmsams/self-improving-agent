"""LLM provider implementations for the self-improving agent.

Supports Anthropic (Claude), AWS Bedrock, and a local mock for testing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Tracks token usage and cost for LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_calls: int = 0
    total_latency_seconds: float = 0.0

    # Pricing per million tokens (Sonnet defaults)
    input_price_per_mtok: float = 3.0
    output_price_per_mtok: float = 15.0

    @property
    def total_cost(self) -> float:
        """Calculate total cost in USD."""
        input_cost = self.input_tokens / 1_000_000 * self.input_price_per_mtok
        output_cost = self.output_tokens / 1_000_000 * self.output_price_per_mtok
        return input_cost + output_cost

    def record(self, input_tok: int, output_tok: int, latency: float) -> None:
        """Record a single API call's usage."""
        self.input_tokens += input_tok
        self.output_tokens += output_tok
        self.total_calls += 1
        self.total_latency_seconds += latency

    def summary(self) -> dict:
        """Return a summary dict for logging/reporting."""
        return {
            "total_calls": self.total_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "avg_latency_s": round(
                self.total_latency_seconds / max(1, self.total_calls), 2
            ),
        }


class AnthropicProvider:
    """LLM provider using the Anthropic API (Claude).

    Features:
    - Exponential backoff retry on rate limit and connection errors
    - Token counting and cost tracking
    - Fail-fast on authentication errors
    """

    RETRY_DELAYS = [1, 4, 16]  # seconds between retries

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20240620",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self._client = None
        self.usage = TokenUsage()

    def _get_client(self):
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "anthropic package not installed. "
                    "Run: pip install anthropic"
                )
        return self._client

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Send a completion request to Claude with retry logic."""
        import anthropic

        client = self._get_client()
        last_error = None

        for attempt in range(len(self.RETRY_DELAYS) + 1):
            try:
                start = time.monotonic()
                response = await client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                latency = time.monotonic() - start

                # Track token usage
                input_tok = getattr(response.usage, "input_tokens", 0)
                output_tok = getattr(response.usage, "output_tokens", 0)
                self.usage.record(input_tok, output_tok, latency)
                logger.debug(
                    f"LLM call: {input_tok} in / {output_tok} out, "
                    f"{latency:.1f}s, cost=${self.usage.total_cost:.4f}"
                )

                return response.content[0].text

            except anthropic.AuthenticationError:
                raise  # Don't retry auth errors

            except anthropic.RateLimitError as e:
                last_error = e
                if attempt < len(self.RETRY_DELAYS):
                    delay = self.RETRY_DELAYS[attempt]
                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}), "
                        f"retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)

            except anthropic.APIConnectionError as e:
                last_error = e
                if attempt < len(self.RETRY_DELAYS):
                    delay = self.RETRY_DELAYS[attempt]
                    logger.warning(
                        f"Connection error (attempt {attempt + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)

        raise last_error  # type: ignore[misc]


class BedrockProvider:
    """LLM provider using AWS Bedrock (Claude models).

    Same interface as AnthropicProvider, uses boto3 bedrock-runtime.
    """

    RETRY_DELAYS = [1, 4, 16]

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-opus-4-20250514-v1:0",
        region: str = "us-east-1",
    ):
        self.model_id = model_id
        self.region = region
        self._client = None
        self.usage = TokenUsage()

    def _get_client(self):
        """Lazy-initialize the Bedrock runtime client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    "bedrock-runtime", region_name=self.region
                )
            except ImportError:
                raise RuntimeError(
                    "boto3 package not installed. "
                    "Run: pip install boto3"
                )
        return self._client

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Send a completion request via Bedrock with retry logic."""
        client = self._get_client()
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        })

        last_error: Optional[Exception] = None

        for attempt in range(len(self.RETRY_DELAYS) + 1):
            try:
                start = time.monotonic()
                # Run synchronous boto3 call in executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.invoke_model(
                        modelId=self.model_id,
                        body=body,
                        contentType="application/json",
                        accept="application/json",
                    ),
                )
                latency = time.monotonic() - start

                result = json.loads(response["body"].read())
                text = result["content"][0]["text"]

                input_tok = result.get("usage", {}).get("input_tokens", 0)
                output_tok = result.get("usage", {}).get("output_tokens", 0)
                self.usage.record(input_tok, output_tok, latency)

                return text

            except Exception as e:
                error_str = str(e)
                if "ThrottlingException" in error_str:
                    last_error = e
                    if attempt < len(self.RETRY_DELAYS):
                        delay = self.RETRY_DELAYS[attempt]
                        logger.warning(
                            f"Bedrock throttled (attempt {attempt + 1}), "
                            f"retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                        continue
                elif "ExpiredTokenException" in error_str:
                    raise  # Don't retry auth errors
                else:
                    last_error = e
                    if attempt < len(self.RETRY_DELAYS):
                        delay = self.RETRY_DELAYS[attempt]
                        logger.warning(
                            f"Bedrock error (attempt {attempt + 1}), "
                            f"retrying in {delay}s: {e}"
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise

        raise last_error  # type: ignore[misc]


class MockProvider:
    """Mock LLM provider for testing without API calls.

    Returns predefined responses based on persona role detection.
    """

    def __init__(self, responses: Optional[dict[str, str]] = None):
        self.responses = responses or {}
        self.call_log: list[dict] = []
        self.usage = TokenUsage()

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Return a mock response based on detected persona."""
        self.call_log.append({
            "system_prompt": system_prompt[:200],
            "user_message": user_message[:200],
            "temperature": temperature,
        })

        # Detect persona from system prompt and return appropriate mock
        if "Architect" in system_prompt:
            return json.dumps(self._mock_architect_response())
        elif "Builder" in system_prompt:
            return json.dumps(self._mock_builder_response())
        elif "Reviewer" in system_prompt:
            return json.dumps(self._mock_reviewer_response())
        elif "Benchmarker" in system_prompt:
            return json.dumps(self._mock_benchmarker_response())
        elif "Retrospective" in system_prompt:
            return json.dumps(self._mock_retrospective_response())
        return json.dumps({"response": "Mock response"})

    @staticmethod
    def _mock_architect_response() -> dict:
        """Mock response for the Architect persona."""
        return {
            "requirements": [
                {
                    "id": "REQ-001",
                    "text": "THE system SHALL have __init__.py files in all packages",
                    "priority": "must",
                }
            ],
            "design": {
                "approach": "Add missing __init__.py files for proper package structure",
                "components": ["Package initialization"],
                "data_flow": "File creation",
            },
            "tasks": [
                {
                    "id": "TASK-001",
                    "title": "Create __init__.py files",
                    "depends_on": [],
                    "effort": "S",
                }
            ],
            "title": "add-missing-init-files",
            "improvement_type": "fix",
            "rationale": "Missing __init__.py files prevent proper imports",
            "files_to_modify": [
                {"path": "src/__init__.py", "action": "create"},
                {"path": "src/core/__init__.py", "action": "create"},
            ],
            "expected_benchmark_impact": {"file_organization": 5.0},
        }

    @staticmethod
    def _mock_builder_response() -> dict:
        """Mock response for the Builder persona."""
        return {
            "improvement_type": "fix",
            "title": "Add missing __init__.py files",
            "rationale": "Package structure requires init files for imports",
            "changes": [
                {
                    "file": "src/__init__.py",
                    "content": '"""Self-improving agent source package."""\n',
                },
                {
                    "file": "src/core/__init__.py",
                    "content": '"""Core module for the self-improving agent."""\n',
                },
            ],
            "test_changes": [
                {
                    "file": "tests/test_imports.py",
                    "content": (
                        '"""Test that all packages are importable."""\n\n'
                        "def test_src_importable():\n"
                        '    """Verify src package can be imported."""\n'
                        "    import src\n"
                        "    assert src is not None\n\n"
                        "def test_core_importable():\n"
                        '    """Verify core module can be imported."""\n'
                        "    import src.core\n"
                        "    assert src.core is not None\n"
                    ),
                },
            ],
        }

    @staticmethod
    def _mock_reviewer_response() -> dict:
        """Mock response for the Reviewer persona."""
        return {
            "approved": True,
            "score": 0.85,
            "summary": "Clean change that improves project structure.",
            "findings": [
                {
                    "severity": "info",
                    "category": "style",
                    "file_path": "src/__init__.py",
                    "line_start": 1,
                    "line_end": 1,
                    "description": "Consider adding version info to __init__.py",
                    "suggestion": 'Add __version__ = "0.1.0"',
                    "auto_fixable": True,
                }
            ],
            "merge_recommendation": "approve",
        }

    @staticmethod
    def _mock_benchmarker_response() -> dict:
        """Mock response for the Benchmarker persona."""
        return {
            "current_scores": {"test_pass_rate": 90, "code_complexity": 85},
            "delta_from_best": {"test_pass_rate": 0, "code_complexity": 5},
            "trend_analysis": "improving",
            "correlations": [],
            "recommendations": ["Focus on test coverage next"],
        }

    @staticmethod
    def _mock_retrospective_response() -> dict:
        """Mock response for the Retrospective persona."""
        return {
            "generation_analyzed": 5,
            "acceptance_rate": 0.6,
            "top_patterns": [
                "Small focused changes have higher acceptance rate",
                "Test additions are almost always accepted",
            ],
            "process_changes": [],
            "stuck_detection": {
                "is_stuck": False,
                "evidence": "Steady improvement trend",
                "suggested_strategy": "Continue current approach",
            },
        }
