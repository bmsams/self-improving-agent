"""Generation reporting and observability for the self-improving agent.

Captures structured data about each generation cycle, saves reports
to disk, and provides dashboard-style aggregate views.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PhaseTimestamp:
    """Start and end times for a single phase."""

    phase: str
    start: str = ""
    end: str = ""
    duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class LLMCallRecord:
    """Record of a single LLM call within a generation."""

    persona: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0


@dataclass
class GenerationReport:
    """Complete structured report of one generation cycle.

    Captures everything needed to reconstruct what happened:
    - Phase timings
    - LLM calls (count, tokens, cost)
    - Benchmark scores before/after/delta
    - Review summary
    - Decision and reason
    - Files changed
    - Errors encountered
    """

    generation: int
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str = ""
    duration_seconds: float = 0.0

    # Phase timings
    phases: list[PhaseTimestamp] = field(default_factory=list)

    # LLM usage
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    # Benchmarks
    benchmark_before: dict = field(default_factory=dict)
    benchmark_after: dict = field(default_factory=dict)
    benchmark_deltas: dict = field(default_factory=dict)
    total_benchmark_delta: float = 0.0

    # Review
    review_approved: bool = False
    review_score: float = 0.0
    review_findings_count: int = 0
    review_summary: str = ""

    # Decision
    accepted: bool = False
    decision_reason: str = ""

    # Changes
    files_changed: list[str] = field(default_factory=list)
    action: str = ""

    # Errors
    errors: list[str] = field(default_factory=list)

    def start_phase(self, phase: str) -> PhaseTimestamp:
        """Start tracking a phase. Returns the phase object."""
        ts = PhaseTimestamp(
            phase=phase,
            start=datetime.now(timezone.utc).isoformat(),
        )
        self.phases.append(ts)
        return ts

    def end_phase(self, phase_ts: PhaseTimestamp, error: Optional[str] = None) -> None:
        """End tracking a phase."""
        phase_ts.end = datetime.now(timezone.utc).isoformat()
        if phase_ts.start:
            start_dt = datetime.fromisoformat(phase_ts.start)
            end_dt = datetime.fromisoformat(phase_ts.end)
            phase_ts.duration_seconds = (end_dt - start_dt).total_seconds()
        if error:
            phase_ts.error = error
            self.errors.append(f"{phase_ts.phase}: {error}")

    def record_llm_call(
        self, persona: str, input_tok: int, output_tok: int,
        latency: float, cost: float,
    ) -> None:
        """Record an LLM call."""
        self.llm_calls.append(LLMCallRecord(
            persona=persona,
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_seconds=round(latency, 2),
            cost_usd=round(cost, 6),
        ))
        self.total_input_tokens += input_tok
        self.total_output_tokens += output_tok
        self.total_cost_usd += cost

    def finalize(self) -> None:
        """Mark the report as complete."""
        self.finished_at = datetime.now(timezone.utc).isoformat()
        if self.started_at:
            start_dt = datetime.fromisoformat(self.started_at)
            end_dt = datetime.fromisoformat(self.finished_at)
            self.duration_seconds = (end_dt - start_dt).total_seconds()

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return asdict(self)

    def save(self, reports_dir: Path) -> Path:
        """Save report to reports/gen-{N}.json."""
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"gen-{self.generation}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


def load_report(path: Path) -> GenerationReport:
    """Load a generation report from a JSON file."""
    data = json.loads(path.read_text())
    phases = [PhaseTimestamp(**p) for p in data.pop("phases", [])]
    llm_calls = [LLMCallRecord(**c) for c in data.pop("llm_calls", [])]
    report = GenerationReport(**{
        k: v for k, v in data.items()
        if k in GenerationReport.__dataclass_fields__
    })
    report.phases = phases
    report.llm_calls = llm_calls
    return report


def load_all_reports(reports_dir: Path) -> list[GenerationReport]:
    """Load all generation reports from a directory, sorted by generation."""
    reports = []
    if not reports_dir.exists():
        return reports
    for f in sorted(reports_dir.glob("gen-*.json")):
        try:
            reports.append(load_report(f))
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to load report {f}: {e}")
    return reports


def format_report(report: GenerationReport) -> str:
    """Pretty-print a single generation report."""
    lines = [
        f"# Generation {report.generation} Report",
        f"**Started**: {report.started_at}",
        f"**Duration**: {report.duration_seconds:.1f}s",
        f"**Result**: {'ACCEPTED' if report.accepted else 'REJECTED'}",
        f"**Action**: {report.action}",
        "",
    ]

    # Phase timings
    lines.append("## Phase Timings")
    for p in report.phases:
        status = "ERROR" if p.error else "OK"
        lines.append(f"  {p.phase:.<30} {p.duration_seconds:>6.1f}s [{status}]")

    # LLM usage
    lines.append("")
    lines.append("## LLM Usage")
    lines.append(f"  Calls: {len(report.llm_calls)}")
    lines.append(f"  Input tokens:  {report.total_input_tokens:,}")
    lines.append(f"  Output tokens: {report.total_output_tokens:,}")
    lines.append(f"  Total cost:    ${report.total_cost_usd:.4f}")

    # Benchmarks
    lines.append("")
    lines.append("## Benchmark Delta")
    lines.append(f"  Total delta: {report.total_benchmark_delta:+.1f}")
    for name, delta in report.benchmark_deltas.items():
        marker = "+" if delta > 0 else ""
        lines.append(f"  {name:.<30} {marker}{delta:.1f}")

    # Review
    lines.append("")
    lines.append("## Review")
    lines.append(f"  Approved: {report.review_approved}")
    lines.append(f"  Score:    {report.review_score:.2f}")
    lines.append(f"  Findings: {report.review_findings_count}")

    # Files
    if report.files_changed:
        lines.append("")
        lines.append("## Files Changed")
        for f in report.files_changed:
            lines.append(f"  - {f}")

    # Errors
    if report.errors:
        lines.append("")
        lines.append("## Errors")
        for e in report.errors:
            lines.append(f"  - {e}")

    return "\n".join(lines)


def format_dashboard(reports: list[GenerationReport]) -> str:
    """Format aggregate dashboard across all generations."""
    if not reports:
        return "No generation reports found. Run 'python main.py run' first."

    total = len(reports)
    accepted = sum(1 for r in reports if r.accepted)
    rejected = total - accepted
    acceptance_rate = accepted / total if total > 0 else 0.0

    total_cost = sum(r.total_cost_usd for r in reports)
    avg_cost = total_cost / total if total > 0 else 0.0

    accepted_deltas = [r.total_benchmark_delta for r in reports if r.accepted]
    avg_delta = sum(accepted_deltas) / len(accepted_deltas) if accepted_deltas else 0.0

    total_duration = sum(r.duration_seconds for r in reports)
    avg_duration = total_duration / total if total > 0 else 0.0

    # Rejection reasons
    rejection_reasons: dict[str, int] = {}
    for r in reports:
        if not r.accepted and r.decision_reason:
            reason = r.decision_reason[:50]
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    lines = [
        "# Self-Improving Agent Dashboard",
        "",
        f"  Generations:      {total}",
        f"  Accepted:         {accepted} ({acceptance_rate:.0%})",
        f"  Rejected:         {rejected}",
        "",
        f"  Total cost:       ${total_cost:.4f}",
        f"  Avg cost/gen:     ${avg_cost:.4f}",
        f"  Total time:       {total_duration:.0f}s",
        f"  Avg time/gen:     {avg_duration:.1f}s",
        "",
        f"  Avg benchmark delta (accepted): {avg_delta:+.1f}",
        "",
    ]

    if rejection_reasons:
        lines.append("## Top Rejection Reasons")
        for reason, count in sorted(
            rejection_reasons.items(), key=lambda x: -x[1]
        )[:5]:
            lines.append(f"  {count:>3}x  {reason}")
        lines.append("")

    # Score trend
    lines.append("## Benchmark Score Trend")
    for r in reports[-20:]:
        bar_len = int(r.total_benchmark_delta / 2) if r.total_benchmark_delta > 0 else 0
        bar = "+" * min(bar_len, 30) if r.total_benchmark_delta >= 0 else "-" * min(abs(int(r.total_benchmark_delta / 2)), 30)
        status = "OK" if r.accepted else "XX"
        lines.append(
            f"  Gen {r.generation:>3} [{status}] {r.total_benchmark_delta:>+6.1f} {bar}"
        )

    return "\n".join(lines)
