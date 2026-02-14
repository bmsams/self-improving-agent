# Self-Improving AI Coding Agent: Architecture & Research

## Research Summary

### Claude Code
- Terminal-native agentic coding tool with CLAUDE.md for project steering
- Supports MCP servers, custom slash commands, hooks (PreToolUse/PostToolUse/Stop)
- Agent SDK for building custom agents; subagents for parallel work
- Checkpoints for safe rollback; plan mode for think-before-code
- Pipe-friendly Unix philosophy: `git diff | claude -p "review for security issues"`
- Key insight: Skills > MCP for most dev workflows (progressive context disclosure)

### AWS Kiro IDE
- Spec-driven development: requirements.md → design.md → tasks.md
- Uses EARS notation (Easy Approach to Requirements Syntax) for requirements
- Steering files (.kiro/steering/) for persistent project knowledge
- Agent hooks: event-driven automations on file save/create/delete
- Property-based testing: generates hundreds of random test cases from specs
- Autonomous agent: maintains context across sessions, learns from PR feedback
- CLI + IDE share same steering files and MCP configuration

### AWS Strands Agents
- Model-driven agent framework: `Agent(system_prompt, tools)` in few lines
- Multi-agent patterns: Agents-as-Tools, Handoffs, Swarms, Graphs
- Agent SOPs: markdown-based workflows with RFC 2119 constraints (MUST/SHOULD/MAY)
- Skills system with Progressive Disclosure (3 phases: metadata → instructions → resources)
- Eval SOP: Plan → Data → Eval → Report with LLM-as-Judge scoring
- Compatible with Claude Skills, Kiro steering, Cursor rules
- SOPs convert to Skills via `strands-agents-sops skills`

### Context Management Patterns
1. **CLAUDE.md / Steering Files**: Persistent project knowledge loaded every session
2. **Skills (Progressive Disclosure)**: Phase 1 metadata (~100 tokens), Phase 2 full instructions (<5K tokens), Phase 3 resources (on demand)
3. **MCP Context**: External tool connections (GitHub, Jira, databases)
4. **SOPs**: Structured natural-language workflows with constraints
5. **Specs**: Requirements → Design → Tasks cascade

### Tight Feedback Loops
- **Code → Test → Fix**: TDD with agent writing failing tests, then fixing
- **Code → Review → Improve**: PR agent reviews, developer (or agent) fixes
- **Code → Benchmark → Optimize**: Run evals, measure, iterate
- **Spec → Implement → Validate**: Property-based testing against specs
- **Hooks → Auto-actions**: PostToolUse hooks for auto-format, auto-lint, auto-test

---

## 5 Approaches to Building a Self-Improving Coding AI Agent

### Approach 1: "The Ouroboros" — Single Agent Self-Modification Loop

**Philosophy**: One agent, one repo, continuous self-consumption.

The agent writes code, commits it to its own repository, opens a PR against
itself, reviews its own PR using a separate "reviewer persona" system prompt,
either merges or rejects with feedback, then iterates. The agent maintains a
`CHANGELOG.md` and `BENCHMARKS.md` that it updates after each cycle.

**Loop**: Code → Commit → PR → Self-Review → Accept/Reject → Benchmark → Learn

**Key Mechanism**: The agent has two modes activated via different system prompts:
- **Builder Mode**: Writes features, fixes bugs, adds capabilities
- **Reviewer Mode**: Reviews PRs with strict criteria, security focus, performance analysis

The agent alternates between these modes, creating an adversarial dynamic where
the reviewer makes the builder better and vice versa.

**Strengths**: Simplest to implement, no coordination overhead
**Weaknesses**: Limited perspective diversity, potential echo chamber

---

### Approach 2: "The Parliament" — Multi-Agent Voting System

**Philosophy**: Multiple specialized agents debate and vote on code changes.

Uses Strands' Agents-as-Tools pattern to create a parliament of specialists:
- **Architect Agent**: Reviews design decisions and system structure
- **Security Agent**: Scans for vulnerabilities, injection points
- **Performance Agent**: Profiles code, benchmarks, suggests optimizations
- **Style Agent**: Enforces coding standards, readability, documentation
- **Test Agent**: Writes comprehensive tests, finds edge cases

Each agent reviews every PR. A **Merge Controller** tallies votes and only
merges when consensus threshold is met (e.g., 4/5 approve). Rejected PRs
get consolidated feedback and are re-submitted.

**Loop**: Code → PR → Parallel Review (5 agents) → Vote → Merge/Reject → Metrics

**Key Mechanism**: Weighted voting based on each agent's historical accuracy.
Agents that catch real bugs get higher voting power over time.

**Strengths**: Diverse perspectives, catches more issues
**Weaknesses**: Higher token cost, potential deadlocks

---

### Approach 3: "The Dojo" — Benchmark-Driven Evolution

**Philosophy**: The agent improves by competing against itself on benchmarks.

Uses Strands Eval SOP to create a continuous improvement cycle:
1. Agent writes code to solve benchmark tasks (SWE-bench style)
2. Results are scored and tracked in a leaderboard
3. Agent analyzes its failures, identifies patterns
4. Agent modifies its own SOPs, skills, and steering documents
5. Next iteration runs with updated configuration
6. Compare scores; keep improvements, revert regressions

The agent literally evolves its own instruction set based on empirical evidence.

**Loop**: Benchmark → Score → Analyze Failures → Update SOPs/Skills → Re-benchmark → Compare

**Key Mechanism**: The agent maintains a `evolution_log.json` that tracks which
SOP/skill changes led to score improvements. Over time, it builds a model of
what makes its own instructions effective.

**Strengths**: Empirically grounded, measurable improvement
**Weaknesses**: Benchmark-gaming risk, narrow optimization

---

### Approach 4: "The Assembly Line" — Spec-Driven Pipeline

**Philosophy**: Kiro-inspired spec-driven development applied recursively.

The agent operates as a full software development pipeline:
1. **Product Manager Agent**: Writes requirements in EARS notation
2. **Architect Agent**: Produces design.md with system architecture
3. **Developer Agent**: Implements tasks from tasks.md
4. **QA Agent**: Runs property-based tests against requirements
5. **DevOps Agent**: Manages git, PRs, CI/CD, deployment
6. **Retrospective Agent**: Analyzes what went well/poorly, updates process

Each agent has its own SOP and steering files. The pipeline runs in a loop,
with the Retrospective Agent modifying the other agents' SOPs after each cycle.

**Loop**: Spec → Design → Implement → Test → Deploy → Retrospect → Update Pipeline

**Key Mechanism**: The Retrospective Agent uses Strands Eval SOP to measure
pipeline efficiency and code quality, then modifies upstream agents' instructions.

**Strengths**: Production-realistic, comprehensive coverage
**Weaknesses**: Complex orchestration, slow cycles

---

### Approach 5: "The Genetic Algorithm" — Population-Based Optimization

**Philosophy**: Maintain a population of agent configurations and evolve them.

Create N variations of the agent, each with slightly different:
- System prompts and SOPs
- Tool preferences and strategies
- Review criteria and thresholds
- Code style preferences

Run all N agents on the same task set. Score results. The top performers
"reproduce" by having their configurations mixed and mutated. Poor performers
are eliminated. Over generations, the population converges on optimal
configurations.

**Loop**: Spawn Population → Task Assignment → Score → Select → Crossover → Mutate → Repeat

**Key Mechanism**: Configuration genes stored as markdown files. Crossover =
combining sections from two parent configs. Mutation = LLM-generated variations
of existing instructions.

**Strengths**: Explores wide solution space, avoids local optima
**Weaknesses**: Extremely token-intensive, complex infrastructure

---

## Chosen Implementation: Hybrid of Approaches 1 + 3 + 4

We implement the **Ouroboros Core** (self-modification loop) with
**Dojo Benchmarking** (empirical measurement) running inside an
**Assembly Line Pipeline** (spec-driven structure).

This gives us:
- Simple inner loop (Ouroboros) that's easy to debug
- Empirical grounding (Dojo) that prevents drift
- Production-realistic structure (Assembly Line) that mirrors real dev
