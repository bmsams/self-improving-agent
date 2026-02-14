# Code Self-Improvement

## Overview
This SOP guides an AI agent through one complete self-improvement cycle:
benchmark → plan → implement → review → validate → accept/reject.

## Parameters
- **project_root** (required): Path to the project to improve
- **focus_area** (optional, default: "auto"): Specific area to improve ("correctness"|"quality"|"performance"|"style"|"auto")
- **max_attempts** (optional, default: 3): Maximum implementation attempts before giving up
- **mode** (optional, default: "interactive"): "interactive" for human-in-loop, "autonomous" for full auto

## Steps

### Step 1: Baseline Measurement
Run the full benchmark suite against the current main branch.

**Constraints:**
- You MUST record all benchmark scores before making any changes
- You MUST be on the main branch with a clean working tree
- You MUST store the baseline in the evolution log

**Output:** Benchmark suite results with scores for each metric

### Step 2: Improvement Planning
Analyze benchmarks and evolution history to identify the highest-impact change.

**Constraints:**
- You MUST consider the last 5 generations' outcomes when planning
- You MUST NOT repeat a recently rejected approach without modification
- You SHOULD target the lowest-scoring benchmark area
- You SHOULD prefer small, focused changes over large refactors
- You MAY propose new benchmarks if existing ones seem insufficient

**Output:** Improvement spec with requirements, design, and tasks (Kiro-style)

### Step 3: Branch and Implement
Create an improvement branch and implement the planned change.

**Constraints:**
- You MUST create a new git branch named `improve/{description}`
- You MUST write tests alongside any new code
- You MUST keep all functions under 50 lines
- You MUST add type hints to all new functions
- You MUST add docstrings to all public functions
- You SHOULD follow the existing code style
- You MUST NOT modify test infrastructure or benchmark definitions

**Output:** Committed code changes on the improvement branch

### Step 4: Self-Review
Review the changes using the Reviewer persona with strict criteria.

**Constraints:**
- You MUST review all changed files
- You MUST check for security vulnerabilities
- You MUST check for performance regressions
- You MUST verify test coverage of new code
- You SHOULD identify any architectural concerns
- You MUST produce a structured review with severity ratings

**Output:** Review result with findings, score, and merge recommendation

### Step 5: Post-Change Validation
Run the full benchmark suite on the improvement branch.

**Constraints:**
- You MUST run the exact same benchmarks as Step 1
- You MUST compare results against the baseline
- You MUST flag any regressions greater than 5%
- You SHOULD document which specific changes caused score deltas

**Output:** Comparison report with deltas for each benchmark

### Step 6: Merge Decision
Decide whether to merge the improvement based on reviews and benchmarks.

**Merge Criteria (ALL must be met):**
- You MUST have no critical review findings
- You MUST have no benchmark regressions > 5%
- You MUST have all tests passing
- You SHOULD have net positive benchmark delta

**If approved:**
- Merge the branch into main with a no-ff merge
- Tag the commit with generation number and score
- Update the evolution log

**If rejected:**
- Record rejection reason in evolution log
- Delete the improvement branch
- Return to main

**Output:** Merge decision with rationale

### Step 7: Retrospective (Every 5 Generations)
Analyze patterns in the evolution history to improve the process itself.

**Constraints:**
- You MUST analyze acceptance/rejection patterns
- You MUST identify which change types are most effective
- You SHOULD recommend process modifications
- You MAY modify steering documents based on evidence
- You MUST NOT lower safety thresholds

**Output:** Process change recommendations with evidence

## Progress Tracking

### Completion Criteria
- One complete cycle (Steps 1-6) executed
- Evolution log updated
- State file saved
- Git tag applied (if merged)

### Success Metrics
- Benchmark score improvement (primary)
- Code quality metrics improvement
- Acceptance rate trend (target: 40-70%)
- No regressions in merged changes
