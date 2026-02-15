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
    from src.core.models import AgentConfig
    from src.core.loop import ImprovementLoop
    from src.core.agentcore import AgentCoreConfig

    # This entrypoint is primarily for Bedrock AgentCore Runtime, so default to Bedrock.
    # Local/dev usage can override via `AGENT_PROVIDER=anthropic|mock`.
    provider_name = os.environ.get("AGENT_PROVIDER", "bedrock")
    model = os.environ.get("AGENT_MODEL")

    if provider_name == "bedrock":
        from src.core.providers import BedrockProvider
        # Many Bedrock Claude models require using an inference profile ID/ARN instead of on-demand model IDs.
        provider = BedrockProvider(
            model_id=model or "us.anthropic.claude-opus-4-6-v1",
            region=os.environ.get("AWS_REGION", "us-east-1"),
        )
    elif provider_name == "mock":
        from src.core.providers import MockProvider
        provider = MockProvider()
    else:
        from src.core.providers import AnthropicProvider
        provider = AnthropicProvider(model=model or "claude-sonnet-4-5-20250929")

    project_root = Path.cwd()
    config = AgentConfig(
        project_root=project_root,
        auto_merge=os.environ.get("AGENT_AUTO_MERGE", "true").lower() == "true",
    )
    ac_config = AgentCoreConfig.from_env()

    return ImprovementLoop(
        config=config,
        llm=provider,
        project_root=project_root,
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
            {"action": "report", "generation": 7}
        """
        # When invoked through the AgentCore data plane, the runtime receives a wrapper
        # that includes a stringified `payload` field.
        raw = payload or {}
        user_payload = raw
        if isinstance(raw, dict) and "payload" in raw:
            inner = raw.get("payload")
            if isinstance(inner, dict):
                user_payload = inner
            elif isinstance(inner, str):
                try:
                    user_payload = json.loads(inner)
                except json.JSONDecodeError:
                    user_payload = {"prompt": inner}

        # Minimal diagnostics (keys only) to help debug invocation payload shapes.
        try:
            if isinstance(raw, dict):
                logger.info("Invoke raw keys: %s", sorted(raw.keys()))
                if "payload" in raw and isinstance(raw.get("payload"), str):
                    inner_s = raw.get("payload") or ""
                    logger.info("Invoke raw.payload length: %d", len(inner_s))
                    try:
                        inner_obj = json.loads(inner_s)
                        if isinstance(inner_obj, dict):
                            logger.info("Invoke inner keys: %s", sorted(inner_obj.keys()))
                    except json.JSONDecodeError:
                        pass
        except Exception:
            # Never fail an invocation due to logging issues.
            pass

        action = user_payload.get("action", "run") if isinstance(user_payload, dict) else "run"
        max_gens = user_payload.get("max_generations", 1) if isinstance(user_payload, dict) else 1

        try:
            if action == "run":
                loop = _create_loop()
                result = await loop.run_generation()
                return {
                    "status": "completed",
                    "generation": result.generation,
                    "accepted": result.accepted,
                    "benchmark_delta": result.benchmark_delta or 0,
                }

            elif action == "loop":
                loop = _create_loop()
                results = []
                for _ in range(max_gens):
                    result = await loop.run_generation()
                    results.append({
                        "generation": result.generation,
                        "accepted": result.accepted,
                    })
                return {"status": "completed", "generations": results}

            elif action == "bench":
                from src.benchmarks.runner import BenchmarkRunner
                benchmarks = BenchmarkRunner(Path.cwd())
                scores = benchmarks.run_all()
                return {"status": "completed", "benchmarks": scores}

            elif action == "status":
                from src.core.models import AgentState
                state = AgentState.load(Path.cwd() / ".agent_state.json")
                return {
                    "status": "completed",
                    "generation": state.generation,
                    "total_prs_merged": state.total_prs_merged,
                    "total_prs_rejected": state.total_prs_rejected,
                }

            elif action == "report":
                # AgentCore Runtime doesn't provide easy access to the container filesystem,
                # so expose a small slice of the structured per-generation report.
                reports_dir = Path.cwd() / "reports"
                if not reports_dir.exists():
                    return {"status": "error", "error": "No reports directory found"}

                gen = user_payload.get("generation") if isinstance(user_payload, dict) else None
                if gen is None:
                    candidates = sorted(
                        reports_dir.glob("gen-*.json"),
                        key=lambda p: int(p.stem.split("-", 1)[1]),
                    )
                    if not candidates:
                        return {"status": "error", "error": "No reports found"}
                    report_path = candidates[-1]
                else:
                    report_path = reports_dir / f"gen-{int(gen)}.json"
                    if not report_path.exists():
                        return {"status": "error", "error": f"Report not found: gen-{int(gen)}"}

                data = json.loads(report_path.read_text(encoding="utf-8"))
                return {
                    "status": "completed",
                    "report": {
                        "generation": data.get("generation"),
                        "accepted": data.get("accepted"),
                        "decision_reason": data.get("decision_reason"),
                        "total_benchmark_delta": data.get("total_benchmark_delta"),
                        "benchmark_deltas": data.get("benchmark_deltas", {}),
                        "review_approved": data.get("review_approved"),
                        "review_score": data.get("review_score"),
                        "review_findings_count": data.get("review_findings_count"),
                        "files_changed": data.get("files_changed", []),
                        "errors": data.get("errors", []),
                    },
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
