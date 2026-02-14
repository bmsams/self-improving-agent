"""AgentCore Runtime entrypoint for the self-improving agent.

Wraps the existing agent loop with BedrockAgentCoreApp to enable
hosting on Amazon Bedrock AgentCore Runtime. The runtime automatically:
- Creates an HTTP server on port 8080
- Provides /invocations endpoint for agent interaction
- Provides /ping endpoint for health checks
- Handles error responses per AWS standards

Usage:
    Local:   python runtime_entrypoint.py
    Runtime: opentelemetry-instrument python -m runtime_entrypoint
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent-runtime")

try:
    from bedrock_agentcore.runtime import BedrockAgentCoreApp
    app = BedrockAgentCoreApp()
except ImportError:
    logger.warning("bedrock-agentcore SDK not installed, runtime hosting unavailable")
    app = None


def _create_loop():
    """Create the improvement loop with current configuration."""
    from src.core.models import AgentConfig, AgentState
    from src.core.loop import ImprovementLoop
    from src.core.agentcore import AgentCoreConfig
    from src.benchmarks.runner import BenchmarkRunner
    from src.git_ops.git_manager import GitOps

    provider_name = os.environ.get("AGENT_PROVIDER", "anthropic")
    model = os.environ.get("AGENT_MODEL")

    if provider_name == "bedrock":
        from src.core.providers import BedrockProvider
        provider = BedrockProvider(model=model or "anthropic.claude-sonnet-4-20250514-v1:0")
    elif provider_name == "mock":
        from src.core.providers import MockProvider
        provider = MockProvider()
    else:
        from src.core.providers import AnthropicProvider
        provider = AnthropicProvider(model=model or "claude-sonnet-4-5-20250929")

    project_root = Path.cwd()
    config = AgentConfig(project_root=project_root)
    state = AgentState.load(project_root)
    benchmarks = BenchmarkRunner(project_root)
    git = GitOps(project_root)
    ac_config = AgentCoreConfig.from_env()

    return ImprovementLoop(
        config=config,
        state=state,
        provider=provider,
        benchmarks=benchmarks,
        git=git,
        auto_merge=os.environ.get("AGENT_AUTO_MERGE", "true").lower() == "true",
        agentcore_config=ac_config,
    )


if app is not None:
    @app.entrypoint
    async def invoke(payload: dict) -> dict:
        """Handle invocation requests from AgentCore Runtime.

        Payload format:
            {"action": "run", "max_generations": 1}
            {"action": "bench"}
            {"action": "status"}
        """
        action = payload.get("action", "run") if payload else "run"
        max_gens = payload.get("max_generations", 1) if payload else 1

        try:
            if action == "run":
                loop = _create_loop()
                result = await loop.run_generation()
                return {
                    "status": "completed",
                    "generation": result.get("generation", 0),
                    "accepted": result.get("accepted", False),
                    "benchmark_delta": result.get("benchmark_delta", 0),
                }

            elif action == "loop":
                loop = _create_loop()
                results = []
                for _ in range(max_gens):
                    result = await loop.run_generation()
                    results.append({
                        "generation": result.get("generation", 0),
                        "accepted": result.get("accepted", False),
                    })
                return {"status": "completed", "generations": results}

            elif action == "bench":
                from src.benchmarks.runner import BenchmarkRunner
                benchmarks = BenchmarkRunner(Path.cwd())
                scores = benchmarks.run_all()
                return {"status": "completed", "benchmarks": scores}

            elif action == "status":
                from src.core.models import AgentState
                state = AgentState.load(Path.cwd())
                return {
                    "status": "completed",
                    "generation": state.generation,
                    "total_accepted": state.total_accepted,
                    "total_rejected": state.total_rejected,
                }

            else:
                return {"status": "error", "error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error(f"Runtime invocation error: {e}")
            return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    if app is not None:
        app.run()
    else:
        print("bedrock-agentcore SDK not installed. Install with:")
        print("  pip install 'bedrock-agentcore<=0.1.5'")
