# Self-Review Skill

## Overview
Specialized skill for reviewing code changes against quality criteria,
producing structured findings with severity ratings, and making merge
recommendations. Implements the Reviewer persona of the self-improving agent.

## When to Use
- After any code change before merging
- When validating auto-generated code
- During retrospective analysis of past changes
- When evaluating proposed architectural modifications

## Instructions

### Review Process
1. **Read the diff** — Understand what changed and why
2. **Check correctness** — Logic errors, edge cases, error handling
3. **Check security** — Injection, auth, data exposure, input validation
4. **Check performance** — Complexity, allocations, I/O patterns
5. **Check style** — Naming, structure, documentation, type hints
6. **Check tests** — Coverage, edge cases, assertions quality
7. **Score and recommend** — Overall quality score and merge decision

### Scoring Rubric
- **0.9-1.0**: Excellent. Clean, well-tested, improves codebase.
- **0.7-0.9**: Good. Minor issues that don't block merge.
- **0.5-0.7**: Acceptable. Some issues to address.
- **0.3-0.5**: Needs work. Significant issues found.
- **0.0-0.3**: Reject. Fundamental problems.

### Merge Decision Matrix
| Critical Findings | Benchmark Delta | Decision          |
|-------------------|-----------------|-------------------|
| 0                 | Positive        | APPROVE           |
| 0                 | Neutral (±2%)   | APPROVE with note |
| 0                 | Negative (>5%)  | REQUEST CHANGES   |
| 1+                | Any             | REQUEST CHANGES   |

## Resources
- `references/review-checklist.md` — Detailed review checklist
- `references/common-findings.md` — Catalog of common issues
- `scripts/analyze-diff.sh` — Script for diff analysis
