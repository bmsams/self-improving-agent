# Contributing to Self-Improving Agent

## Development Setup

```bash
# Clone the repository
git clone https://github.com/bmsams/self-improving-agent.git
cd self-improving-agent

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Copy environment file
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Running Tests

```bash
make test          # Unit tests only (no API calls)
make test-live     # Integration tests (requires API key)
make lint          # Run ruff linter
make typecheck     # Run mypy
make bench         # Run benchmark suite
```

## Making Changes

1. Create a branch: `git checkout -b improve/your-description`
2. Make your changes
3. Run tests: `make test`
4. Run linter: `make lint`
5. Commit with the format: `[agent] type: description`
   - Types: `feat`, `fix`, `refactor`, `bench`, `docs`, `test`

## Project Structure

- `src/core/` — Core agent loop, state, config, safety
- `src/agents/` — Agent personas (architect, builder, reviewer)
- `src/benchmarks/` — Benchmark definitions and runner
- `src/git_ops/` — Git and GitHub operations
- `tests/` — Test suite
- `config/` — Configuration files
- `reports/` — Generation reports (auto-generated)

## Code Standards

- Type hints on all functions
- Docstrings on all public methods (Google style)
- No function longer than 50 lines
- All new code must have tests
- Use `pathlib.Path` not `os.path`
