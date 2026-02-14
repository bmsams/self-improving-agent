#!/usr/bin/env python3
"""Self-Improving AI Coding Agent — Main Entry Point.

Usage:
    python main.py run                  # Run one generation
    python main.py loop --max-gen 10    # Run continuous improvement loop
    python main.py bench                # Run benchmarks only
    python main.py status               # Show current state
    python main.py agentcore-status     # Show AgentCore service health
    python main.py history              # Show evolution history
    python main.py reset                # Reset to generation 0

    Flags:
      --mock              Use mock LLM (no API calls)
      --agentcore         Enable AWS Bedrock AgentCore services
      --auto-merge        Auto-merge without review gate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from src.core.models import AgentConfig, AgentState
from src.core.loop import ImprovementLoop
from src.core.providers import AnthropicProvider, MockProvider
from src.core.agentcore import AgentCoreConfig, AgentCoreServices
from src.benchmarks.runner import BenchmarkRunner
from src.git_ops.git_manager import GitOps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Self-Improving AI Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # run — single generation
    run_p = sub.add_parser("run", help="Run one improvement generation")
    run_p.add_argument("--mock", action="store_true", help="Use mock LLM provider")
    run_p.add_argument("--auto-merge", action="store_true", help="Auto-merge without review gate")
    run_p.add_argument("--agentcore", action="store_true", help="Enable AWS Bedrock AgentCore services")
    run_p.add_argument("--agentcore-region", default="us-east-1", help="AgentCore AWS region")

    # loop — continuous improvement
    loop_p = sub.add_parser("loop", help="Run continuous improvement loop")
    loop_p.add_argument("--max-gen", type=int, default=10, help="Maximum generations")
    loop_p.add_argument("--mock", action="store_true", help="Use mock LLM provider")
    loop_p.add_argument("--auto-merge", action="store_true", help="Auto-merge without review gate")
    loop_p.add_argument("--agentcore", action="store_true", help="Enable AWS Bedrock AgentCore services")
    loop_p.add_argument("--agentcore-region", default="us-east-1", help="AgentCore AWS region")

    # bench — run benchmarks
    sub.add_parser("bench", help="Run benchmark suite")

    # status — show state
    sub.add_parser("status", help="Show current agent state")

    # agentcore-status — show AgentCore health
    sub.add_parser("agentcore-status", help="Show AWS AgentCore services health")

    # history — show evolution log
    hist_p = sub.add_parser("history", help="Show evolution history")
    hist_p.add_argument("--last", type=int, default=20, help="Number of entries to show")

    # reset — reset state
    sub.add_parser("reset", help="Reset agent state to generation 0")

    return parser.parse_args()


def _make_agentcore_config(args: argparse.Namespace) -> AgentCoreConfig:
    """Build AgentCore config from CLI args."""
    if getattr(args, "agentcore", False):
        return AgentCoreConfig(
            enabled=True,
            region=getattr(args, "agentcore_region", "us-east-1"),
        )
    return AgentCoreConfig(enabled=False)


async def cmd_run(args: argparse.Namespace, root: Path) -> None:
    """Run a single improvement generation."""
    config = AgentConfig(
        project_root=root,
        auto_merge=getattr(args, "auto_merge", False),
    )
    llm = MockProvider() if args.mock else AnthropicProvider()
    ac_config = _make_agentcore_config(args)
    loop = ImprovementLoop(config, llm, root, agentcore_config=ac_config)

    if ac_config.enabled:
        health = loop.ac.health_check()
        logger.info(f"AgentCore services: {health}")

    entry = await loop.run_generation()
    status = "✅ ACCEPTED" if entry.accepted else "❌ REJECTED"
    logger.info(f"\n{'='*60}")
    logger.info(f"Generation {entry.generation}: {status}")
    logger.info(f"Action: {entry.action}")
    logger.info(f"Benchmark Delta: {entry.benchmark_delta or 0:+.1f}")
    logger.info(f"{'='*60}")


async def cmd_loop(args: argparse.Namespace, root: Path) -> None:
    """Run the continuous improvement loop."""
    config = AgentConfig(
        project_root=root,
        max_generations=args.max_gen,
        auto_merge=getattr(args, "auto_merge", False),
    )
    llm = MockProvider() if args.mock else AnthropicProvider()
    ac_config = _make_agentcore_config(args)
    loop = ImprovementLoop(config, llm, root, agentcore_config=ac_config)
    await loop.run_loop(max_generations=args.max_gen)


def cmd_bench(root: Path) -> None:
    """Run benchmark suite and display results."""
    runner = BenchmarkRunner(root)
    git = GitOps(root)
    suite = runner.run_all(git_sha=git.current_sha)
    report = runner.format_report(suite)
    print(report)
    print(f"\n{'='*60}")
    print(f"Total: {suite.total_score:.1f}/{suite.total_max:.1f} "
          f"({suite.total_score/suite.total_max*100:.1f}%)")
    print(f"Pass Rate: {suite.pass_rate*100:.1f}%")


def cmd_status(root: Path) -> None:
    """Display current agent state."""
    state = AgentState.load(root / ".agent_state.json")
    print(f"""
╔══════════════════════════════════════════════════╗
║         Self-Improving Agent Status              ║
╠══════════════════════════════════════════════════╣
║ Generation:      {state.generation:<32}║
║ PRs Created:     {state.total_prs_created:<32}║
║ PRs Merged:      {state.total_prs_merged:<32}║
║ PRs Rejected:    {state.total_prs_rejected:<32}║
║ Best Score:      {state.current_best_score:<32.1f}║
║ Acceptance Rate: {(state.total_prs_merged / max(1, state.total_prs_created) * 100):<31.1f}%║
╚══════════════════════════════════════════════════╝
""")


def cmd_history(root: Path, last: int = 20) -> None:
    """Display evolution history."""
    state = AgentState.load(root / ".agent_state.json")
    entries = state.evolution_log[-last:]
    if not entries:
        print("No evolution history yet. Run 'python main.py run' to start.")
        return
    print(f"\n{'Gen':>4} {'Status':>8} {'Delta':>8} {'Action'}")
    print(f"{'─'*4} {'─'*8} {'─'*8} {'─'*40}")
    for e in entries:
        status = "✅" if e.accepted else "❌"
        delta = f"{e.benchmark_delta or 0:+.1f}"
        print(f"{e.generation:>4} {status:>8} {delta:>8} {e.action[:40]}")


def cmd_reset(root: Path) -> None:
    """Reset agent state."""
    state_file = root / ".agent_state.json"
    if state_file.exists():
        state_file.unlink()
    history_file = root / ".benchmark_history.json"
    if history_file.exists():
        history_file.unlink()
    print("Agent state reset to generation 0.")


def cmd_agentcore_status() -> None:
    """Display AWS Bedrock AgentCore service health."""
    ac_config = AgentCoreConfig.from_env()
    services = AgentCoreServices(ac_config)
    health = services.health_check()
    policies = services.policy.get_policies_summary()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║       AWS Bedrock AgentCore — Service Health             ║
╠══════════════════════════════════════════════════════════╣
║ Region:            {ac_config.region:<37}║
║ Memory:            {'🟢 Connected' if health['memory'] else '🔴 Unavailable':<37}║
║ Code Interpreter:  {'🟢 Connected' if health['code_interpreter'] else '🔴 Unavailable':<37}║
║ Observability:     {'🟢 Active' if health['observability'] else '🟡 Local Only':<37}║
║ Evaluations:       {'🟢 Connected' if health['evaluations'] else '🟡 Local Only':<37}║
║ Gateway:           {'🟢 Connected' if health['gateway'] else '🔴 Not Configured':<37}║
║ Policy:            {'🟢 Active' if health['policy'] else '🔴 Disabled':<37}║
╠══════════════════════════════════════════════════════════╣
║ Safety Policies ({len(policies)} active):{'':>33}║""")
    for p in policies:
        name = p['name'][:50]
        print(f"║   • {name:<52}║")
    print(f"""╠══════════════════════════════════════════════════════════╣
║ Environment Variables:{'':>35}║
║   AGENTCORE_ENABLED   = {str(ac_config.enabled):<33}║
║   AGENTCORE_MEMORY_ID = {str(ac_config.memory_id or 'auto'):<33}║
║   AGENTCORE_GATEWAY_ID= {str(ac_config.gateway_id or 'not set'):<33}║
╚══════════════════════════════════════════════════════════╝
""")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    root = Path.cwd()

    if not args.command:
        print(__doc__)
        sys.exit(1)

    if args.command == "run":
        asyncio.run(cmd_run(args, root))
    elif args.command == "loop":
        asyncio.run(cmd_loop(args, root))
    elif args.command == "bench":
        cmd_bench(root)
    elif args.command == "status":
        cmd_status(root)
    elif args.command == "agentcore-status":
        cmd_agentcore_status()
    elif args.command == "history":
        cmd_history(root, args.last)
    elif args.command == "reset":
        cmd_reset(root)


if __name__ == "__main__":
    main()
