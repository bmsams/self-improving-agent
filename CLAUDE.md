# CLAUDE.md — Self-Improving Agent Project

## Project Overview
This is a self-improving AI coding agent that writes code, reviews its own PRs,
benchmarks performance, and iteratively improves its own codebase and instructions.

## Architecture
- Python 3.11+ with asyncio for agent orchestration
- Git for version control and PR simulation
- Strands-style SOPs for structured workflows
- Skills with Progressive Disclosure for context management
- JSON-based benchmark tracking and evolution logging

## Coding Standards
- Type hints on all functions
- Docstrings on all public methods (Google style)
- No function longer than 50 lines
- All new code must have corresponding tests
- Use `pathlib.Path` not `os.path`
- Prefer dataclasses and Pydantic models over raw dicts
- Async by default for agent operations

## Git Conventions
- Branch naming: `improve/{description}` for self-improvements
- Commit messages: `[agent] {type}: {description}` where type is feat/fix/refactor/bench/docs
- PRs require: description, test results, benchmark delta

## Review Criteria (for self-review)
- MUST: No regressions in benchmark scores
- MUST: All tests pass
- MUST: Type checking passes
- SHOULD: Improved code coverage
- SHOULD: Reduced complexity (cyclomatic)
- MAY: Performance improvements
- MAY: Better documentation

## File Organization
- `src/core/` — Core agent loop, state management
- `src/agents/` — Specialized agent personas (builder, reviewer, etc.)
- `src/benchmarks/` — Benchmark definitions and runners
- `src/git_ops/` — Git operations, PR management
- `src/mcp/` — MCP server definitions
- `sops/` — Standard Operating Procedures
- `skills/` — Progressive disclosure skills
- `steering/` — Steering documents
- `tests/` — Test suite
