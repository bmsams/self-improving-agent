# AWS Bedrock AgentCore Integration

## Overview

The self-improving agent integrates 6 AWS Bedrock AgentCore services to add
production-grade memory, sandboxing, observability, evaluation, tool access,
and safety enforcement. All services gracefully degrade when AWS credentials
are unavailable — the agent runs fully locally with mock/fallback behavior.

## Services

### 1. AgentCore Memory — Episodic Learning

**What it does**: Stores every generation outcome as episodic memory,
searchable across sessions and deployments.

**How the agent uses it**:
- **Planning phase**: Recalls similar past attempts to avoid repeating failures
- **Planning phase**: Retrieves top success patterns to guide strategy
- **Post-merge**: Stores outcome (accepted/rejected, benchmark delta, files changed)
- **Review meta-learning**: Stores which review findings were valid predictors

**Why it matters**: Without Memory, the agent only learns within a single session
via the evolution_log. With Memory, patterns survive restarts, redeployments,
and even team sharing — an agent running on a different branch can learn from
improvements discovered on main.

```python
# Recall similar past attempts before planning
past = await ac.memory.recall_similar_attempts("add type hints to utils")
# → [{"action": "Add type hints", "accepted": True, "benchmark_delta": 8.3}, ...]
```

### 2. AgentCore Code Interpreter — Sandboxed Execution

**What it does**: Runs benchmarks and validates generated code in isolated
sandbox environments with CPU/memory limits and automatic cleanup.

**How the agent uses it**:
- **Pre-apply validation**: Syntax-checks and import-validates code before writing
- **Benchmark execution**: Runs pytest/mypy/ruff in isolation so the agent can't
  game scores by modifying the runner on the host

**Why it matters**: The Builder generates code that gets executed. Without
sandboxing, a creative-but-misguided improvement could `os.system("rm -rf /")`
or modify benchmark definitions. The Code Interpreter prevents this.

```python
validation = await ac.code_interpreter.validate_generated_code(new_code)
# → {"syntax_valid": True, "imports_valid": True, "issues": []}
```

### 3. AgentCore Observability — OpenTelemetry Tracing

**What it does**: Creates structured traces for every generation cycle,
with child spans for each phase and attributes for LLM calls, benchmarks,
and decisions.

**How the agent uses it**:
- Parent span per generation (8-hour window supported)
- Child spans: benchmark_baseline → plan → implement → review → benchmark_post → evaluations
- Attributes: token usage, latency, model, persona, scores, decisions

**Why it matters**: When a generation fails or regresses, you can trace
exactly which phase went wrong, what the LLM said, and what benchmarks
changed — all in CloudWatch or Datadog.

```python
gen_span = ac.observability.start_generation_span(generation=42)
ac.observability.record_llm_call(gen_span, "claude-sonnet-4-5", "builder", 500, 2000, 1200.0)
ac.observability.record_decision(gen_span, "rejected", "benchmark regression -15.2")
```

### 4. AgentCore Evaluations — Quality Scoring

**What it does**: Scores each generation across 4 quality dimensions
(correctness, helpfulness, safety, spec_compliance) using built-in and
custom evaluators.

**How the agent uses it**:
- **Built-in**: Correctness, Helpfulness, Safety (via AgentCore when available)
- **Custom**: Benchmark Regression, Spec Compliance, Self-Consistency
- **Local fallback**: Scores computed from review scores + benchmark deltas

**Why it matters**: The merge decision uses review + benchmarks, but Evaluations
adds a meta-layer that catches subtler quality issues and publishes to
CloudWatch for long-term trend analysis.

```python
scores = await ac.evaluations.evaluate_generation(gen, plan, impl, review, comparison)
# → {"correctness": 0.85, "helpfulness": 0.9, "safety": 1.0, "overall": 0.88}
```

### 5. AgentCore Gateway — MCP Tool Integration

**What it does**: Converts APIs into MCP-compatible tools that the agent
can discover and invoke securely.

**How the agent uses it**:
- `github_create_pr`: Creates real GitHub PRs (beyond local git simulation)
- `github_get_reviews`: Fetches human review feedback from GitHub
- Extensible: add Jira, Slack, dependency scanning, etc.

**Why it matters**: Moves the agent from local-only git operations to
real-world CI/CD integration.

### 6. AgentCore Policy — Cedar Safety Boundaries

**What it does**: Enforces 5 Cedar policies that prevent the agent from
gaming scores, bypassing reviews, or damaging infrastructure.

**Policies**:
| Policy | Rule |
|--------|------|
| no_benchmark_modification | Cannot modify benchmarks/runner.py |
| no_test_deletion | Cannot delete files in tests/ |
| no_safety_bypass | Cannot modify merge decision logic |
| change_limit_per_generation | Maximum 10 file changes |
| no_steering_self_modification | Only Retrospective can modify CLAUDE.md/steering |

**How it works**: Every `file_write` and `file_delete` goes through
`policy.check_action()` before execution. Blocked actions immediately
reject the generation.

```python
allowed, reason = ac.policy.check_action("file_write", "benchmarks/runner.py", "builder")
# → (False, "Policy: Cannot modify benchmark runner")
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Improvement Loop (9 phases)                   │
├─────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│Benchmark│  Plan    │Implement │  Review  │ Evaluate │  Merge   │
│         │          │          │          │          │          │
│  ┌──────┤ ┌───────┐│ ┌───────┐│          │ ┌───────┐│          │
│  │Code  │ │Memory ││ │Policy ││          │ │Evals  ││          │
│  │Interp│ │Recall ││ │Check  ││          │ │Score  ││          │
│  └──────┤ └───────┘│ │       ││          │ └───────┘│          │
│         │          │ │Code   ││          │          │ ┌───────┐│
│         │          │ │Interp ││          │          │ │Memory ││
│         │          │ │Valid. ││          │          │ │Store  ││
│         │          │ └───────┘│          │          │ └───────┘│
├─────────┴──────────┴──────────┴──────────┴──────────┴──────────┤
│                   AgentCore Observability (spans)               │
├─────────────────────────────────────────────────────────────────┤
│                    CloudWatch / Datadog / OTEL                  │
└─────────────────────────────────────────────────────────────────┘
```

## CLI Usage

```bash
# Run with AgentCore services enabled
python main.py run --mock --agentcore --agentcore-region us-east-1

# Continuous loop with AgentCore
python main.py loop --max-gen 20 --agentcore

# Check AgentCore service health
python main.py agentcore-status
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| AGENTCORE_ENABLED | true | Enable AgentCore services |
| AWS_REGION | us-east-1 | AWS region for AgentCore |
| AGENTCORE_MEMORY_NS | self-improving-agent | Memory namespace |
| AGENTCORE_MEMORY_ID | auto | Memory store ID |
| AGENTCORE_GATEWAY_ID | - | Gateway ID for tools |
| AGENTCORE_POLICY_STORE_ID | - | Policy store for Cedar |

## Test Coverage

30 tests cover all 6 services:
- Policy: 12 tests (all Cedar rules + edge cases)
- Evaluations: 3 tests (good/bad/mediocre generations)
- Memory: 3 tests (graceful degradation)
- Code Interpreter: 2 tests (sandbox fallback)
- Observability: 3 tests (NoOp span, recording)
- Gateway: 3 tests (tool registration, invocation)
- Services: 2 tests (initialization, health check)
