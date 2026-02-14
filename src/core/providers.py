"""LLM provider implementations for the self-improving agent.

Supports Anthropic (Claude), OpenAI, and a local mock for testing.
"""

from __future__ import annotations

import json
import os
from typing import Optional


class AnthropicProvider:
    """LLM provider using the Anthropic API (Claude)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self._client = None

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
        """Send a completion request to Claude."""
        client = self._get_client()
        response = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


class MockProvider:
    """Mock LLM provider for testing without API calls.

    Returns predefined responses based on persona role detection.
    """

    def __init__(self, responses: Optional[dict[str, str]] = None):
        self.responses = responses or {}
        self.call_log: list[dict] = []

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
