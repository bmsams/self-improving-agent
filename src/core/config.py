"""Configuration management for the self-improving agent.

Loads config from config/agent.toml, with CLI args and env vars as overrides.
Precedence: env vars > CLI args > config file > defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


@dataclass
class AgentSettings:
    """Merged configuration from all sources."""

    # Agent core
    # Default to Bedrock Opus inference profile for runtime-hosted usage.
    model: str = "us.anthropic.claude-opus-4-20250514-v1:0"
    provider: str = "bedrock"
    max_generations: int = 100
    auto_merge: bool = False
    benchmark_threshold: float = 0.0
    review_consensus: float = 0.6
    log_level: str = "INFO"

    # Benchmarks
    benchmarks_enabled: list[str] = field(default_factory=lambda: [
        "test_pass_rate", "code_complexity", "type_check",
        "code_coverage", "lint_score", "doc_coverage",
        "file_organization", "import_check", "security_scan",
    ])
    benchmarks_timeout: int = 60
    benchmarks_weights: dict[str, float] = field(default_factory=lambda: {
        "test_pass_rate": 2.0,
        "code_complexity": 1.0,
        "type_check": 1.0,
        "code_coverage": 1.0,
        "lint_score": 0.5,
        "doc_coverage": 0.5,
        "file_organization": 0.5,
        "import_check": 1.0,
        "security_scan": 1.5,
    })

    # AgentCore
    agentcore_enabled: bool = False
    agentcore_region: str = "us-east-1"
    agentcore_memory_namespace: str = "self-improving-agent"

    # Per-generation guardrails
    max_tokens_per_generation: int = 500_000  # Token budget (input + output)
    generation_timeout_seconds: int = 600     # Wall-clock timeout per generation
    max_concurrent_subagents: int = 5         # Max parallel sub-agents

    # Safety
    safety_max_files: int = 10
    safety_blocked_patterns: list[str] = field(default_factory=lambda: [
        "os.system", "subprocess.call", "eval(", "exec(",
    ])
    safety_require_tests: bool = True


def load_config(
    project_root: Path,
    cli_overrides: Optional[dict] = None,
) -> AgentSettings:
    """Load configuration with precedence: env > CLI > file > defaults.

    Args:
        project_root: Root directory of the project.
        cli_overrides: Dict of CLI argument overrides.

    Returns:
        Merged AgentSettings instance.
    """
    settings = AgentSettings()

    # 1. Load from config file
    config_path = project_root / "config" / "agent.toml"
    if config_path.exists():
        _apply_toml(settings, config_path)

    # 2. Apply CLI overrides
    if cli_overrides:
        _apply_cli(settings, cli_overrides)

    # 3. Apply environment variable overrides
    _apply_env(settings)

    return settings


def _apply_toml(settings: AgentSettings, path: Path) -> None:
    """Apply values from a TOML config file."""
    if tomllib is None:
        logger.warning("tomllib/tomli not available — skipping config file")
        return

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config from {path}: {e}")
        return

    agent = data.get("agent", {})
    if "model" in agent:
        settings.model = agent["model"]
    if "provider" in agent:
        settings.provider = agent["provider"]
    if "max_generations" in agent:
        settings.max_generations = int(agent["max_generations"])
    if "auto_merge" in agent:
        settings.auto_merge = bool(agent["auto_merge"])
    if "benchmark_threshold" in agent:
        settings.benchmark_threshold = float(agent["benchmark_threshold"])
    if "review_consensus" in agent:
        settings.review_consensus = float(agent["review_consensus"])
    if "log_level" in agent:
        settings.log_level = agent["log_level"]

    benchmarks = data.get("benchmarks", {})
    if "enabled" in benchmarks:
        settings.benchmarks_enabled = benchmarks["enabled"]
    if "timeout_seconds" in benchmarks:
        settings.benchmarks_timeout = int(benchmarks["timeout_seconds"])
    if "weights" in benchmarks:
        settings.benchmarks_weights.update(benchmarks["weights"])

    ac = data.get("agentcore", {})
    if "enabled" in ac:
        settings.agentcore_enabled = bool(ac["enabled"])
    if "region" in ac:
        settings.agentcore_region = ac["region"]
    if "memory_namespace" in ac:
        settings.agentcore_memory_namespace = ac["memory_namespace"]

    safety = data.get("safety", {})
    if "max_files_per_generation" in safety:
        settings.safety_max_files = int(safety["max_files_per_generation"])
    if "blocked_patterns" in safety:
        settings.safety_blocked_patterns = safety["blocked_patterns"]
    if "require_tests" in safety:
        settings.safety_require_tests = bool(safety["require_tests"])

    limits = data.get("limits", {})
    if "max_tokens_per_generation" in limits:
        settings.max_tokens_per_generation = int(limits["max_tokens_per_generation"])
    if "generation_timeout_seconds" in limits:
        settings.generation_timeout_seconds = int(limits["generation_timeout_seconds"])
    if "max_concurrent_subagents" in limits:
        settings.max_concurrent_subagents = int(limits["max_concurrent_subagents"])


def _apply_cli(settings: AgentSettings, overrides: dict) -> None:
    """Apply CLI argument overrides."""
    if "model" in overrides and overrides["model"]:
        settings.model = overrides["model"]
    if "provider" in overrides and overrides["provider"]:
        settings.provider = overrides["provider"]
    if "max_gen" in overrides and overrides["max_gen"] is not None:
        settings.max_generations = overrides["max_gen"]
    if "auto_merge" in overrides:
        settings.auto_merge = overrides["auto_merge"]
    if "agentcore" in overrides:
        settings.agentcore_enabled = overrides["agentcore"]
    if "agentcore_region" in overrides and overrides["agentcore_region"]:
        settings.agentcore_region = overrides["agentcore_region"]


def _apply_env(settings: AgentSettings) -> None:
    """Apply environment variable overrides."""
    env_map = {
        "AGENT_MODEL": "model",
        "AGENT_PROVIDER": "provider",
        "AGENT_MAX_GENERATIONS": "max_generations",
        "AGENT_AUTO_MERGE": "auto_merge",
        "AGENT_LOG_LEVEL": "log_level",
        "AGENTCORE_ENABLED": "agentcore_enabled",
        "AGENTCORE_REGION": "agentcore_region",
        "AGENT_MAX_TOKENS_PER_GEN": "max_tokens_per_generation",
        "AGENT_GEN_TIMEOUT": "generation_timeout_seconds",
    }
    for env_key, attr in env_map.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        current = getattr(settings, attr)
        if isinstance(current, bool):
            setattr(settings, attr, val.lower() in ("true", "1", "yes"))
        elif isinstance(current, int):
            try:
                setattr(settings, attr, int(val))
            except ValueError:
                logger.warning(f"Invalid int for {env_key}: {val}")
        elif isinstance(current, float):
            try:
                setattr(settings, attr, float(val))
            except ValueError:
                logger.warning(f"Invalid float for {env_key}: {val}")
        else:
            setattr(settings, attr, val)


def validate_config(settings: AgentSettings) -> list[str]:
    """Validate config and return list of issues (empty = valid)."""
    issues = []
    if settings.provider not in ("anthropic", "bedrock", "mock"):
        issues.append(f"Unknown provider: {settings.provider}")
    if settings.max_generations < 1:
        issues.append(f"max_generations must be >= 1, got {settings.max_generations}")
    if settings.benchmark_threshold < -100 or settings.benchmark_threshold > 100:
        issues.append(f"benchmark_threshold out of range: {settings.benchmark_threshold}")
    if not 0.0 <= settings.review_consensus <= 1.0:
        issues.append(f"review_consensus must be 0.0-1.0, got {settings.review_consensus}")
    if settings.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        issues.append(f"Invalid log_level: {settings.log_level}")
    if settings.safety_max_files < 1:
        issues.append(f"safety_max_files must be >= 1, got {settings.safety_max_files}")
    if settings.max_tokens_per_generation < 1000:
        issues.append(f"max_tokens_per_generation must be >= 1000, got {settings.max_tokens_per_generation}")
    if settings.generation_timeout_seconds < 10:
        issues.append(f"generation_timeout_seconds must be >= 10, got {settings.generation_timeout_seconds}")
    return issues


def format_config(settings: AgentSettings) -> str:
    """Format effective config for display."""
    lines = [
        "# Effective Configuration",
        "",
        "[agent]",
        f"  model             = {settings.model}",
        f"  provider          = {settings.provider}",
        f"  max_generations   = {settings.max_generations}",
        f"  auto_merge        = {settings.auto_merge}",
        f"  benchmark_threshold = {settings.benchmark_threshold}",
        f"  review_consensus  = {settings.review_consensus}",
        f"  log_level         = {settings.log_level}",
        "",
        "[benchmarks]",
        f"  enabled           = {settings.benchmarks_enabled}",
        f"  timeout_seconds   = {settings.benchmarks_timeout}",
        "  weights:",
    ]
    for name, weight in sorted(settings.benchmarks_weights.items()):
        lines.append(f"    {name:<20} = {weight}")
    lines.extend([
        "",
        "[agentcore]",
        f"  enabled           = {settings.agentcore_enabled}",
        f"  region            = {settings.agentcore_region}",
        f"  memory_namespace  = {settings.agentcore_memory_namespace}",
        "",
        "[limits]",
        f"  max_tokens_per_generation   = {settings.max_tokens_per_generation}",
        f"  generation_timeout_seconds  = {settings.generation_timeout_seconds}",
        "",
        "[safety]",
        f"  max_files         = {settings.safety_max_files}",
        f"  blocked_patterns  = {settings.safety_blocked_patterns}",
        f"  require_tests     = {settings.safety_require_tests}",
    ])
    return "\n".join(lines)
