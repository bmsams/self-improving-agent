"""Core configuration and data models for the self-improving agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class AgentRole(Enum):
    """Roles an agent can assume during the improvement loop."""
    BUILDER = "builder"
    REVIEWER = "reviewer"
    BENCHMARKER = "benchmarker"
    ARCHITECT = "architect"
    RETROSPECTIVE = "retrospective"


class PRStatus(Enum):
    """Status of a pull request in the self-improvement loop."""
    DRAFT = "draft"
    OPEN = "open"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    REJECTED = "rejected"
    MERGED = "merged"


class Severity(Enum):
    """Review finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ReviewFinding:
    """A single finding from a code review."""
    severity: Severity
    category: str  # e.g., "security", "performance", "style", "logic"
    file_path: str
    line_start: int
    line_end: int
    description: str
    suggestion: str
    auto_fixable: bool = False


@dataclass
class ReviewResult:
    """Complete result of a code review."""
    pr_id: str
    reviewer_role: AgentRole
    approved: bool
    findings: list[ReviewFinding] = field(default_factory=list)
    summary: str = ""
    score: float = 0.0  # 0.0 to 1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def blocking(self) -> bool:
        return self.critical_count > 0 or not self.approved


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    benchmark_name: str
    score: float
    max_score: float
    passed: bool
    duration_seconds: float
    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def percentage(self) -> float:
        return (self.score / self.max_score * 100) if self.max_score > 0 else 0.0


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results from a single run."""
    suite_id: str
    results: list[BenchmarkResult] = field(default_factory=list)
    git_sha: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total_score(self) -> float:
        return sum(r.score for r in self.results)

    @property
    def total_max(self) -> float:
        return sum(r.max_score for r in self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


@dataclass
class PullRequest:
    """Represents a self-improvement pull request."""
    pr_id: str
    branch_name: str
    title: str
    description: str
    status: PRStatus = PRStatus.DRAFT
    files_changed: list[str] = field(default_factory=list)
    reviews: list[ReviewResult] = field(default_factory=list)
    benchmark_before: Optional[BenchmarkSuite] = None
    benchmark_after: Optional[BenchmarkSuite] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    merged_at: Optional[str] = None

    @property
    def benchmark_delta(self) -> Optional[float]:
        if self.benchmark_before and self.benchmark_after:
            before = self.benchmark_before.total_score
            after = self.benchmark_after.total_score
            return after - before
        return None

    @property
    def all_approved(self) -> bool:
        return all(r.approved for r in self.reviews) and len(self.reviews) > 0

    @property
    def has_blocking(self) -> bool:
        return any(r.blocking for r in self.reviews)


@dataclass
class EvolutionEntry:
    """Tracks a single evolution step of the agent."""
    generation: int
    action: str  # What the agent did
    target_file: str  # What file was modified
    rationale: str  # Why this change was made
    benchmark_delta: Optional[float] = None
    accepted: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AgentState:
    """Full state of the self-improving agent."""
    generation: int = 0
    total_prs_created: int = 0
    total_prs_merged: int = 0
    total_prs_rejected: int = 0
    evolution_log: list[EvolutionEntry] = field(default_factory=list)
    benchmark_history: list[BenchmarkSuite] = field(default_factory=list)
    current_best_score: float = 0.0
    active_branch: str = "main"

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> AgentState:
        if path.exists():
            data = json.loads(path.read_text())
            # Reconstruct EvolutionEntry objects from dicts
            if "evolution_log" in data:
                rebuilt = []
                for e in data["evolution_log"]:
                    if isinstance(e, dict):
                        rebuilt.append(EvolutionEntry(
                            **{k: v for k, v in e.items()
                               if k in EvolutionEntry.__dataclass_fields__}
                        ))
                    else:
                        rebuilt.append(e)
                data["evolution_log"] = rebuilt
            # Reconstruct BenchmarkSuite/BenchmarkResult from dicts
            if "benchmark_history" in data:
                rebuilt_suites = []
                for suite_data in data["benchmark_history"]:
                    if isinstance(suite_data, dict):
                        results = []
                        for r in suite_data.get("results", []):
                            if isinstance(r, dict):
                                results.append(BenchmarkResult(
                                    **{k: v for k, v in r.items()
                                       if k in BenchmarkResult.__dataclass_fields__}
                                ))
                        suite_fields = {
                            k: v for k, v in suite_data.items()
                            if k in BenchmarkSuite.__dataclass_fields__ and k != "results"
                        }
                        suite_fields["results"] = results
                        rebuilt_suites.append(BenchmarkSuite(**suite_fields))
                data["benchmark_history"] = rebuilt_suites
            return cls(**{k: v for k, v in data.items()
                         if k in cls.__dataclass_fields__})
        return cls()


@dataclass
class AgentConfig:
    """Configuration for the self-improving agent."""
    project_root: Path = Path(".")
    state_file: str = ".agent_state.json"
    max_generations: int = 100
    benchmark_threshold: float = 0.0  # Min delta to accept a change
    review_consensus_threshold: float = 0.6  # % of reviewers that must approve
    max_retries_per_pr: int = 3
    auto_merge: bool = False  # If True, merge without human approval
    # Used for local/dev defaults; runtime_entrypoint sets provider/model explicitly.
    model_name: str = "us.anthropic.claude-opus-4-20250514-v1:0"
    log_level: str = "INFO"

    # Per-generation guardrails
    max_tokens_per_generation: int = 500_000  # Token budget (input + output)
    generation_timeout_seconds: int = 600     # Wall-clock timeout per generation
    max_concurrent_subagents: int = 5         # Max parallel sub-agents per generation

    # Reviewer weights (for parliament voting)
    reviewer_weights: dict[str, float] = field(default_factory=lambda: {
        "architect": 1.0,
        "security": 1.5,
        "performance": 1.0,
        "style": 0.5,
        "test": 1.2,
    })
