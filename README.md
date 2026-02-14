# Self-Improving Agent

A self-improving AI coding agent that writes code, reviews its own PRs, benchmarks performance, and iteratively evolves its own codebase. It uses Claude to plan improvements, implement changes, review them, and decide whether to merge — all autonomously.

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env               # Add your ANTHROPIC_API_KEY
python main.py run --mock --auto-merge   # Test with mock LLM
python main.py run --auto-merge          # Run with real LLM
python main.py loop --max-gen 3 --auto-merge  # Multi-generation loop
```

## Architecture

The agent uses three core patterns:

- **Ouroboros Engine**: Self-modification loop (plan, implement, review, benchmark, merge/reject)
- **Dojo**: Empirical benchmarks drive evolution (9 benchmarks: tests, complexity, coverage, lint, types, docs, imports, security, file org)
- **Assembly Line**: Spec-driven pipeline with 5 personas (Architect, Builder, Reviewer, Benchmarker, Retrospective)

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details.

## CLI Reference

| Command | Description |
|---------|-------------|
| `python main.py run` | Run one improvement generation |
| `python main.py loop --max-gen N` | Run N improvement generations |
| `python main.py bench` | Run benchmark suite |
| `python main.py status` | Show current agent state |
| `python main.py history` | Show evolution history |
| `python main.py report N` | Show generation N report |
| `python main.py dashboard` | Show aggregate dashboard |
| `python main.py config show` | Show effective configuration |
| `python main.py config validate` | Validate config file |
| `python main.py reset` | Reset to generation 0 |

### Flags

| Flag | Description |
|------|-------------|
| `--mock` | Use mock LLM (no API calls) |
| `--provider {anthropic,bedrock,mock}` | Choose LLM provider |
| `--model NAME` | Override default model |
| `--auto-merge` | Auto-merge without review gate |
| `--dry-run` | Run without file writes or git commits |
| `--no-safety` | Disable safety guardrails (dev only) |
| `--github` | Enable GitHub remote operations |
| `--github-auto` | GitHub mode with auto-merge |
| `--agentcore` | Enable AWS Bedrock AgentCore |

## Configuration

Edit `config/agent.toml` to customize behavior. Environment variables override config file values. See `.env.example` for all options.

## FAQ

**Is it safe?**
Yes. Safety guardrails block dangerous patterns (eval, exec, pickle, shell=True), protect critical files (safety.py, merge decision logic), and validate Python syntax before writing. Cedar policies add 5 additional safety rules. Use `--no-safety` only for development.

**What if it breaks itself?**
Each change runs on a separate git branch. If benchmarks regress or tests fail, the change is rejected and the branch is discarded. The main branch is only updated when improvements pass all checks.

**How much does it cost per generation?**
Typically $0.01-$0.05 per generation with Claude Sonnet (3 LLM calls: plan, implement, review). Use `python main.py dashboard` to see cumulative costs.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
