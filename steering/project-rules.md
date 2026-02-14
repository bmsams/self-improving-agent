# Project Steering Rules

## Code Quality
- All Python files MUST use type hints on function signatures
- All public functions MUST have Google-style docstrings
- Maximum cyclomatic complexity per function: 10
- Maximum function length: 50 lines
- Prefer dataclasses over raw dictionaries for structured data
- Use pathlib.Path instead of os.path

## Git Workflow
- Branch naming: `improve/{slug}-{hash}`
- Commit format: `[agent] {type}: {description}`
- Valid types: feat, fix, refactor, perf, docs, bench, test
- PRs require benchmark comparison before merge

## Testing
- Every new function needs a corresponding test
- Tests use pytest with pytest-asyncio for async tests
- Mock external services (LLM, file system) in unit tests
- Integration tests use tempfile.TemporaryDirectory

## Architecture
- Agents communicate through well-defined data models (not raw strings)
- The improvement loop is the single entry point for all changes
- Benchmarks are the source of truth for quality measurement
- SOPs define the process; Skills define the knowledge

## Self-Improvement Rules
- Never modify benchmark definitions to game scores
- Never bypass the review step
- Never reduce test coverage
- Always log rationale for accepted/rejected changes
- Retrospective analysis every 5 generations
