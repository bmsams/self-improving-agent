# Production Readiness Task — Self-Improving AI Agent

## Context

You are working on a self-improving AI coding agent located at `./self-improving-agent/`. The system has a working architecture with 55 passing tests, but has only been validated with mock LLM responses. Your job is to make it production-ready — meaning it can run real improvement cycles against itself using live LLM calls, handle failures gracefully, and produce reliable results.

Read CLAUDE.md, docs/ARCHITECTURE.md, and docs/AGENTCORE.md first to understand the system before making any changes.

## Current State

**What works:**
- Mock provider cycle: plan → implement → review → benchmark → accept/reject
- 7 benchmarks run against real code (test_pass_rate, code_complexity, type_check, code_coverage, lint_score, doc_coverage, file_organization)
- Git branching, PR simulation, merge/reject flow
- Cedar policy enforcement (5 rules)
- Local evaluation scoring fallback
- State persistence with EvolutionEntry deserialization
- CLI: run, loop, bench, status, history, reset, agentcore-status
- 55 tests passing (25 core + 30 AgentCore)

**What has NOT been tested:**
- Real LLM calls (Anthropic API) — JSON parsing may fail on real output
- Multi-generation loops where the agent actually improves itself
- Error recovery mid-generation (what happens if the LLM returns garbage?)
- Concurrent git operations or merge conflicts
- AWS AgentCore services with real credentials
- Code Interpreter sandbox execution (always falls back to local)

## Task List

Complete these tasks in order. Each task should end with all tests still passing. Commit after each task.

---

### Task 1: Harden JSON Response Parsing

**Problem:** `_parse_json_response()` in `src/core/loop.py` is brittle. Real LLMs return markdown-wrapped JSON, partial JSON, JSON with trailing text, or sometimes no JSON at all. The mock provider returns perfectly structured dicts — real output won't.

**Requirements:**
1. Rewrite `_parse_json_response()` to handle:
   - JSON wrapped in ```json ... ``` fences (already handled, but verify)
   - JSON wrapped in ``` ... ``` without language tag
   - JSON with trailing explanation text after the closing `}`
   - JSON with leading explanation text before the opening `{`
   - Multiple JSON blocks in one response (take the first valid one)
   - Completely missing JSON (return structured error with raw text)
   - Nested JSON with unescaped strings that break parsing
2. Add a retry mechanism: if parsing fails, re-prompt the LLM once with "Please respond with ONLY valid JSON, no explanation" and the original prompt
3. Add a `_validate_response_schema()` method that checks the parsed dict has expected keys for each persona:
   - Architect: must have `title`, `files_to_modify`
   - Builder: must have `changes` (list with `file` and `content` keys)
   - Reviewer: must have `approved` (bool), `score` (float), `findings` (list)
   - If validation fails, return a safe default that will cause the generation to be rejected (not crash)
4. Write tests for all parsing edge cases (at least 10 new tests)

**Files to modify:** `src/core/loop.py`, `tests/test_agent.py`

---

### Task 2: Robust Error Recovery in the Generation Loop

**Problem:** If any phase fails mid-generation (LLM timeout, git conflict, benchmark crash), the system leaves orphan branches and corrupted state. The `run_generation()` method has a single try/except at the `run_loop()` level but nothing inside phases.

**Requirements:**
1. Wrap each phase in `run_generation()` with individual error handling
2. On any phase failure:
   - Log the error with full context (which phase, what input, what failed)
   - Clean up the git branch (switch to main, delete orphan branch)
   - Record a failed EvolutionEntry with `accepted=False` and error details in rationale
   - Increment generation counter (don't retry the same generation)
   - Return the failed entry (don't raise)
3. Add a `_cleanup_on_failure()` method that safely returns to main branch
4. Add exponential backoff for LLM calls (start at 1s, max 30s, 3 retries)
5. Add a `--dry-run` CLI flag that runs everything except actual file writes and git commits (useful for testing)
6. Handle the case where benchmarks take too long: add a timeout (default 120s per benchmark, configurable)
7. Write tests that simulate failures at each phase using mock provider

**Files to modify:** `src/core/loop.py`, `main.py`, `src/benchmarks/runner.py`, `tests/test_agent.py`

---

### Task 3: Real LLM Integration with Retry Logic

**Problem:** The `AnthropicProvider` has no error handling, no retries, no rate limiting, and no streaming support.

**Requirements:**
1. Add retry logic to `AnthropicProvider.complete()`:
   - Retry on `anthropic.RateLimitError` with exponential backoff
   - Retry on `anthropic.APIConnectionError` (network issues)
   - Don't retry on `anthropic.AuthenticationError` (fail fast)
   - Maximum 3 retries with 1s, 4s, 16s delays
2. Add token counting and cost tracking:
   - Track input_tokens, output_tokens per call
   - Log cumulative cost per generation (use Anthropic pricing: input $3/MTok, output $15/MTok for Sonnet)
   - Add cost info to EvolutionEntry or a separate cost log
3. Add a `BedrockProvider` class for AWS Bedrock (same interface as AnthropicProvider):
   - Use `boto3` with `bedrock-runtime` client
   - Support `invoke_model` with Claude model IDs
   - Same retry logic
4. Add a `--provider` CLI flag: `anthropic` (default), `bedrock`, `mock`
5. Add a `--model` CLI flag to override the default model
6. Write tests for retry behavior using mocked API responses

**Files to modify:** `src/core/providers.py`, `main.py`, `src/core/models.py`, `tests/test_agent.py`

---

### Task 4: Improve Benchmark Reliability

**Problem:** Benchmarks shell out to `pytest`, `mypy`, `ruff` and parse stdout. This is fragile — output format changes between versions, and benchmarks that fail hard return 0 instead of meaningful scores.

**Requirements:**
1. Pin tool versions in `pyproject.toml` (pytest, mypy, ruff with specific versions)
2. Add defensive parsing for each benchmark:
   - `_bench_test_pass_rate`: handle pytest XML output (`--junitxml`) instead of parsing stdout
   - `_bench_type_check`: handle mypy exit codes properly (0 = clean, 1 = errors, 2 = fatal)
   - `_bench_lint_score`: verify ruff JSON output schema before parsing
   - `_bench_code_coverage`: handle missing `coverage.json` gracefully (coverage-pytest not installed)
3. Add benchmark timeouts: each benchmark gets 60s max, returns a failing result on timeout instead of hanging
4. Add a `_bench_import_check` benchmark: verify all `src/` modules can be imported without errors
5. Add a `_bench_security_scan` benchmark: check for dangerous patterns (eval, exec, subprocess.call with shell=True, pickle.loads on untrusted data)
6. Make benchmarks configurable via `config/benchmarks.json`:
   ```json
   {
     "enabled": ["test_pass_rate", "code_complexity", "doc_coverage"],
     "timeouts": {"test_pass_rate": 120, "default": 60},
     "weights": {"test_pass_rate": 2.0, "lint_score": 0.5}
   }
   ```
7. Write tests for each new benchmark and timeout behavior

**Files to modify:** `src/benchmarks/runner.py`, `pyproject.toml`, `tests/test_agent.py`

---

### Task 5: Self-Improvement Validation (Integration Test)

**Problem:** The agent has never actually improved itself. We need an integration test that proves the full loop works end-to-end with a real LLM.

**Requirements:**
1. Create `tests/test_integration_live.py` (skipped unless `ANTHROPIC_API_KEY` is set)
2. Test scenario: Create a small target project in a temp dir with intentional issues:
   - A Python file missing type hints
   - A function with no docstring
   - A missing `__init__.py`
   - A test file with one failing test
3. Run ONE generation against this target project with the real LLM
4. Assert:
   - Generation completes without crash
   - EvolutionEntry is created with valid fields
   - State file is saved and loadable
   - Git has at least 2 commits (init + agent's change)
   - Benchmark scores are recorded (before and after)
   - The generation was either accepted or rejected with a valid reason
5. Create a `Makefile` with:
   - `make test` — run unit tests only (no API calls)
   - `make test-live` — run integration tests (requires API key)
   - `make lint` — run ruff
   - `make typecheck` — run mypy
   - `make bench` — run benchmarks
   - `make run` — single generation with mock
   - `make run-live` — single generation with real LLM

**Files to create:** `tests/test_integration_live.py`, `Makefile`

---

### Task 6: Logging and Observability

**Problem:** Current logging is basic `logger.info()` calls with no structure. You can't reconstruct what happened in a generation from logs alone.

**Requirements:**
1. Add structured JSON logging option (via `--log-format json` CLI flag):
   ```json
   {"ts": "...", "level": "INFO", "phase": "benchmark", "gen": 5, "msg": "...", "data": {...}}
   ```
2. Add a `GenerationReport` dataclass that captures everything about a generation:
   - Timestamps for each phase (start/end)
   - LLM calls (count, tokens, latency, cost)
   - Benchmark scores (before/after/delta for each)
   - Review summary and findings count
   - Eval scores
   - Decision and reason
   - Files changed
   - Errors encountered
3. Save each generation report as `reports/gen-{N}.json`
4. Add a `report` CLI command that pretty-prints a generation report
5. Add a `dashboard` CLI command that shows aggregate stats across all generations:
   - Acceptance rate over time
   - Average benchmark improvement per accepted PR
   - Most common rejection reasons
   - Cost per generation
   - Total cost so far
6. Write tests for report generation and dashboard output

**Files to create:** `src/core/reporting.py`, `reports/` directory
**Files to modify:** `src/core/loop.py`, `src/core/models.py`, `main.py`, `tests/test_agent.py`

---

### Task 7: Configuration and Environment Management

**Problem:** Config is scattered across CLI args, environment variables, and hardcoded defaults. No config file support.

**Requirements:**
1. Create `config/agent.toml` as the single source of truth:
   ```toml
   [agent]
   model = "claude-sonnet-4-5-20250929"
   provider = "anthropic"
   max_generations = 100
   auto_merge = false
   benchmark_threshold = 0.0
   review_consensus = 0.6
   log_level = "INFO"
   log_format = "text"

   [benchmarks]
   enabled = ["test_pass_rate", "code_complexity", "doc_coverage", "file_organization", "lint_score"]
   timeout_seconds = 60
   
   [benchmarks.weights]
   test_pass_rate = 2.0
   code_complexity = 1.0
   
   [agentcore]
   enabled = false
   region = "us-east-1"
   memory_namespace = "self-improving-agent"
   
   [safety]
   max_files_per_generation = 10
   blocked_patterns = ["os.system", "subprocess.call", "eval(", "exec("]
   require_tests = true
   ```
2. CLI args override config file, env vars override CLI args
3. Add `config validate` CLI command that checks the config file
4. Add `config show` CLI command that displays effective config (merged from all sources)
5. Write tests for config loading, merging, and validation

**Files to create:** `config/agent.toml`, `src/core/config.py`
**Files to modify:** `main.py`, `src/core/loop.py`, `tests/test_agent.py`

---

### Task 8: Safety Guardrails for Self-Modification

**Problem:** The agent modifies its own code. If it writes bad code that breaks its own loop, it could brick itself. Cedar policies exist but are only checked at file-write time — they don't catch semantic attacks.

**Requirements:**
1. Add a `SafetyValidator` class in `src/core/safety.py`:
   - `validate_before_write(filepath, content)` — check content before writing
   - Block writes that contain:
     - `os.system()`, `subprocess.Popen()` with `shell=True`
     - `eval()`, `exec()` on user/LLM-generated strings
     - `__import__()` calls
     - Modifications to `src/core/loop.py`'s `_make_merge_decision()` method
     - Modifications to safety.py itself
     - Import of `pickle` or `marshal` (deserialization attacks)
   - `validate_after_change(project_root)` — run after all changes applied:
     - Verify all existing tests still pass
     - Verify the agent can still import its own modules
     - Verify `main.py` can still parse args
   - If post-change validation fails, auto-rollback the branch
2. Add a `--no-safety` flag that disables guardrails (for development only, prints warning)
3. Integrate SafetyValidator into the loop between Phase 3 (implement) and Phase 4 (create PR)
4. Write thorough tests including adversarial cases

**Files to create:** `src/core/safety.py`
**Files to modify:** `src/core/loop.py`, `main.py`, `tests/test_agent.py`

---

### Task 9: GitHub Integration (Optional Remote)

**Problem:** The git operations are local-only. PRs are simulated objects. To run in production, you need real GitHub PRs with real reviews.

**Requirements:**
1. Add `GitHubIntegration` class in `src/git_ops/github.py`:
   - `push_branch(branch_name)` — push to remote
   - `create_pr(title, body, head, base)` — create real GitHub PR via API
   - `get_reviews(pr_number)` — fetch review comments
   - `merge_pr(pr_number)` — merge via API
   - `close_pr(pr_number)` — close without merge
   - Uses `GITHUB_TOKEN` env var for auth
   - Uses `GITHUB_REPO` env var for repo (format: `owner/repo`)
2. Add `--github` CLI flag that enables remote operations
3. When enabled:
   - Push branches to remote after commit
   - Create real PRs
   - Wait for human review (with configurable timeout, default 24h)
   - Auto-merge if review approved and benchmarks pass
4. When disabled: continue with local simulation (current behavior)
5. Add a `--github-auto` mode that creates PRs but auto-merges without waiting for human review
6. Write tests using mocked GitHub API responses

**Files to create:** `src/git_ops/github.py`
**Files to modify:** `src/core/loop.py`, `main.py`, `tests/test_agent.py`

---

### Task 10: Packaging and Deployment

**Requirements:**
1. Update `pyproject.toml`:
   - Add all dependencies with version pins (anthropic, boto3, pytest, mypy, ruff, pytest-cov, pytest-asyncio)
   - Add optional dependency groups: `[dev]`, `[aws]`, `[github]`
   - Add entry point: `self-improving-agent = "main:main"`
   - Set version to 1.0.0
2. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.12-slim
   # Install git, system deps
   # Copy project
   # Install dependencies
   # Set entrypoint
   ```
3. Create `docker-compose.yml` for local development with volume mounts
4. Create `.env.example` with all required environment variables documented
5. Update `README.md` with:
   - What this project does (1 paragraph)
   - Quick start (5 lines to get running)
   - Architecture overview (link to ARCHITECTURE.md)
   - Configuration reference
   - CLI reference (all commands and flags)
   - FAQ: "Is it safe?", "What if it breaks itself?", "How much does it cost per generation?"
6. Create `CONTRIBUTING.md` with development setup instructions
7. Ensure `make test` passes in Docker

**Files to create:** `Dockerfile`, `docker-compose.yml`, `.env.example`, `CONTRIBUTING.md`
**Files to modify:** `pyproject.toml`, `README.md`

---

## Completion Criteria

When all tasks are done:
1. `make test` — all unit tests pass (target: 100+ tests)
2. `make lint` — zero ruff issues
3. `make typecheck` — zero mypy errors (with `--strict` on `src/`)
4. `make test-live` — integration test passes with real API key
5. The agent can run 3 consecutive generations against itself without crashing:
   ```bash
   python main.py loop --max-gen 3 --auto-merge
   ```
6. Generation reports are saved to `reports/`
7. Docker build succeeds and container runs
8. All new code has docstrings, type hints, and tests

## Constraints

- Do NOT modify the core architecture (Ouroboros + Dojo + Assembly Line)
- Do NOT change the 5 Cedar safety policies
- Do NOT remove any existing tests
- Keep backward compatibility with the mock provider
- Every task should be a separate commit with a descriptive message
- Run tests after every change — never commit broken code
