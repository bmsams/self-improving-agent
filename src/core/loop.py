"""Core self-improvement loop — the Ouroboros Engine.

This is the heart of the self-improving agent. It orchestrates:
1. Benchmark current state (Dojo)
2. Plan improvement (Architect)
3. Implement change (Builder)
4. Review change (Reviewer)
5. Validate with benchmarks (Dojo)
6. Accept or reject (Git ops)
7. Learn from outcome (Retrospective)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional, Protocol

from src.core.models import (
    AgentConfig, AgentRole, AgentState, BenchmarkSuite,
    EvolutionEntry, PRStatus, PullRequest, ReviewResult,
    ReviewFinding, Severity,
)
from src.agents.personas import get_persona, AgentPersona
from src.benchmarks.runner import BenchmarkRunner
from src.git_ops.git_manager import GitOps
from src.core.agentcore import AgentCoreServices, AgentCoreConfig
from src.core.reporting import GenerationReport

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM interaction — swap in any provider."""

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Send a completion request and return the response text."""
        ...


class ImprovementLoop:
    """The main self-improvement loop controller.

    Implements the hybrid Ouroboros + Dojo + Assembly Line approach.
    Integrates AWS Bedrock AgentCore for:
    - Memory: episodic learning across generations
    - Code Interpreter: sandboxed benchmark execution
    - Observability: OpenTelemetry tracing of all decisions
    - Evaluations: quality scoring (correctness, helpfulness, safety)
    - Gateway: MCP tool integration (GitHub, analysis)
    - Policy: Cedar-based safety boundaries
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: LLMProvider,
        project_root: Optional[Path] = None,
        agentcore_config: Optional[AgentCoreConfig] = None,
        dry_run: bool = False,
    ):
        self.config = config
        self.llm = llm
        self.root = project_root or config.project_root
        self.dry_run = dry_run
        self.git = GitOps(self.root)
        self.benchmarks = BenchmarkRunner(self.root)
        self.state = AgentState.load(self.root / config.state_file)

        # Initialize AgentCore services (gracefully degrades if unavailable)
        self.ac = AgentCoreServices(agentcore_config)

    async def run_generation(self) -> EvolutionEntry:
        """Execute one complete improvement generation.

        Each phase is individually wrapped with error handling. On failure:
        - Logs full context (phase, input, error)
        - Cleans up orphan git branches
        - Records a failed EvolutionEntry
        - Increments generation counter
        - Returns the failed entry (does not raise)
        """
        gen = self.state.generation + 1
        branch = None
        logger.info(f"═══ Generation {gen} starting ═══")

        gen_span = self.ac.observability.start_generation_span(gen)
        report = GenerationReport(generation=gen)

        # ── Phase 1: Benchmark baseline ──
        try:
            logger.info("Phase 1: Benchmarking current state...")
            phase_ts = report.start_phase("benchmark_baseline")
            bench_span = self.ac.observability.start_phase_span("benchmark_baseline", gen_span)
            baseline = self.benchmarks.run_all(git_sha=self.git.current_sha)
            for r in baseline.results:
                self.ac.observability.record_benchmark(
                    bench_span, r.benchmark_name, r.score, r.max_score,
                )
            bench_span.end()
            report.end_phase(phase_ts)
            report.benchmark_before = {
                r.benchmark_name: r.score for r in baseline.results
            }
            logger.info(f"Baseline score: {baseline.total_score:.1f}/{baseline.total_max:.1f}")
        except Exception as e:
            logger.error(f"Phase 1 (baseline benchmark) failed: {e}", exc_info=True)
            report.end_phase(phase_ts, error=str(e))
            self._save_report(report)
            return self._record_failure(gen, "benchmark_baseline", str(e), gen_span)

        # ── Phase 2: Plan improvement ──
        try:
            phase_ts = report.start_phase("plan")
            plan_span = self.ac.observability.start_phase_span("plan", gen_span)
            logger.info("Phase 2: Planning improvement (memory-informed)...")
            plan = await self._plan_improvement_with_memory(baseline)
            plan_span.end()
            report.end_phase(phase_ts)
            report.action = plan.get("title", "unknown")
        except Exception as e:
            logger.error(f"Phase 2 (planning) failed: {e}", exc_info=True)
            report.end_phase(phase_ts, error=str(e))
            self._save_report(report)
            return self._record_failure(gen, "plan", str(e), gen_span)

        # ── Phase 3: Implement on branch ──
        try:
            phase_ts = report.start_phase("implement")
            logger.info("Phase 3: Implementing change...")
            impl_span = self.ac.observability.start_phase_span("implement", gen_span)
            branch = self.git.create_improvement_branch(plan.get("title", "unnamed"))
            implementation = await self._implement_change(plan)

            # Policy check
            files_changed_count = 0
            policy_blocked = False
            reason = ""
            for change in implementation.get("changes", []) + implementation.get("test_changes", []):
                file_path = change.get("file", "")
                allowed, reason = self.ac.policy.check_action(
                    action="file_write",
                    resource_path=file_path,
                    agent_role="builder",
                    context={"files_changed_count": files_changed_count},
                )
                if not allowed:
                    logger.warning(f"Policy blocked: {reason} ({file_path})")
                    policy_blocked = True
                    break
                files_changed_count += 1

            if policy_blocked:
                self._cleanup_on_failure(branch)
                entry = EvolutionEntry(
                    generation=gen,
                    action=plan.get("title", "policy-blocked"),
                    target_file="POLICY_VIOLATION",
                    rationale=reason,
                    benchmark_delta=0,
                    accepted=False,
                )
                self.ac.observability.record_decision(gen_span, "rejected", reason)
                gen_span.end()
                self.state.generation = gen
                self.state.total_prs_created += 1
                self.state.total_prs_rejected += 1
                self.state.evolution_log.append(entry)
                self.state.save(self.root / self.config.state_file)
                logger.info(f"═══ Generation {gen} POLICY-BLOCKED ═══")
                return entry

            # Code validation
            for change in implementation.get("changes", []):
                if change.get("file", "").endswith(".py"):
                    validation = await self.ac.code_interpreter.validate_generated_code(
                        change.get("content", "")
                    )
                    if not validation.get("fallback") and not validation.get("syntax_valid"):
                        logger.warning(
                            f"Code validation failed for {change['file']}: "
                            f"{validation.get('issues', [])}"
                        )

            if not self.dry_run:
                self._apply_changes(implementation)
                sha = self.git.commit_changes(
                    f"[agent] {plan.get('improvement_type', 'feat')}: "
                    f"{plan.get('title', 'improvement')}"
                )
            else:
                sha = self.git.current_sha
                logger.info("[DRY RUN] Skipping file writes and git commit")

            impl_span.end()
            report.end_phase(phase_ts)
            report.files_changed = [
                c.get("file", "unknown") for c in implementation.get("changes", [])
            ]
        except Exception as e:
            logger.error(f"Phase 3 (implement) failed: {e}", exc_info=True)
            report.end_phase(phase_ts, error=str(e))
            self._cleanup_on_failure(branch)
            self._save_report(report)
            return self._record_failure(gen, "implement", str(e), gen_span)

        # ── Phase 4: Create PR ──
        try:
            phase_ts = report.start_phase("create_pr")
            pr = self.git.create_pr(
                title=plan.get("title", "Self-improvement"),
                description=self._format_pr_description(plan, baseline),
            )
            pr.benchmark_before = baseline
            report.end_phase(phase_ts)
            logger.info(f"Created {pr.pr_id}: {pr.title}")
        except Exception as e:
            logger.error(f"Phase 4 (create PR) failed: {e}", exc_info=True)
            report.end_phase(phase_ts, error=str(e))
            self._cleanup_on_failure(branch)
            self._save_report(report)
            return self._record_failure(gen, "create_pr", str(e), gen_span)

        # ── Phase 5: Review ──
        try:
            phase_ts = report.start_phase("review")
            logger.info("Phase 5: Reviewing change...")
            review_span = self.ac.observability.start_phase_span("review", gen_span)
            review = await self._review_change(pr)
            pr.reviews.append(review)
            review_span.end()
            report.end_phase(phase_ts)
            report.review_approved = review.approved
            report.review_score = review.score
            report.review_findings_count = len(review.findings)
            report.review_summary = review.summary
        except Exception as e:
            logger.error(f"Phase 5 (review) failed: {e}", exc_info=True)
            report.end_phase(phase_ts, error=str(e))
            self._cleanup_on_failure(branch)
            self._save_report(report)
            return self._record_failure(gen, "review", str(e), gen_span)

        # ── Phase 6: Post-change benchmarks ──
        try:
            phase_ts = report.start_phase("benchmark_post")
            logger.info("Phase 6: Post-change benchmarks...")
            post_span = self.ac.observability.start_phase_span("benchmark_post", gen_span)

            sandbox_result = await self.ac.code_interpreter.run_benchmarks_sandboxed(
                str(self.root),
                "from src.benchmarks.runner import BenchmarkRunner; print('ok')",
            )
            post_bench = self.benchmarks.run_all(git_sha=sha)

            pr.benchmark_after = post_bench
            comparison = self.benchmarks.compare(baseline, post_bench)
            for r in post_bench.results:
                self.ac.observability.record_benchmark(
                    post_span, r.benchmark_name, r.score, r.max_score,
                )
            post_span.end()
            report.end_phase(phase_ts)
            report.benchmark_after = {
                r.benchmark_name: r.score for r in post_bench.results
            }
            report.total_benchmark_delta = comparison.get("total_delta", 0)
            for item in comparison.get("improved", []) + comparison.get("regressed", []):
                report.benchmark_deltas[item["name"]] = item["delta"]
        except Exception as e:
            logger.error(f"Phase 6 (post benchmarks) failed: {e}", exc_info=True)
            report.end_phase(phase_ts, error=str(e))
            self._cleanup_on_failure(branch)
            self._save_report(report)
            return self._record_failure(gen, "benchmark_post", str(e), gen_span)

        # ── Phase 7: Evaluations ──
        try:
            eval_span = self.ac.observability.start_phase_span("evaluations", gen_span)
            eval_scores = await self.ac.evaluations.evaluate_generation(
                generation=gen,
                plan=plan,
                implementation=implementation,
                review={"score": review.score, "approved": review.approved},
                benchmark_comparison=comparison,
            )
            logger.info(
                f"Eval scores: correctness={eval_scores.get('correctness', 0):.2f} "
                f"helpfulness={eval_scores.get('helpfulness', 0):.2f} "
                f"safety={eval_scores.get('safety', 0):.2f} "
                f"overall={eval_scores.get('overall', 0):.2f}"
            )
            eval_span.end()
        except Exception as e:
            logger.error(f"Phase 7 (evaluations) failed: {e}", exc_info=True)
            eval_scores = {"overall": 0.0}

        # ── Phase 8: Accept or reject ──
        accepted = self._make_merge_decision(pr, comparison)
        entry = EvolutionEntry(
            generation=gen,
            action=plan.get("title", "unknown"),
            target_file=", ".join(
                f.get("path", str(f)) if isinstance(f, dict) else str(f)
                for f in plan.get("files_to_modify", ["unknown"])[:3]
            ),
            rationale=plan.get("rationale", ""),
            benchmark_delta=comparison.get("total_delta", 0),
            accepted=accepted,
        )

        self.ac.observability.record_decision(
            gen_span,
            "merged" if accepted else "rejected",
            f"delta={comparison.get('total_delta', 0):+.1f}, "
            f"review_score={review.score:.2f}, "
            f"eval_overall={eval_scores.get('overall', 0):.2f}",
        )

        if not self.dry_run:
            if accepted:
                logger.info(f"ACCEPTED — merging (delta: {comparison['total_delta']:+.1f})")
                self.git.merge_pr(pr)
                self.state.total_prs_merged += 1
                self.state.current_best_score = max(
                    self.state.current_best_score, post_bench.total_score
                )
            else:
                logger.info(f"REJECTED — discarding (delta: {comparison['total_delta']:+.1f})")
                self.git.reject_pr(pr)
                self.state.total_prs_rejected += 1
        else:
            logger.info(f"[DRY RUN] Would {'merge' if accepted else 'reject'} PR")

        # ── Phase 9: Update state and learn ──
        self.state.generation = gen
        self.state.total_prs_created += 1
        self.state.evolution_log.append(entry)
        self.state.benchmark_history.append(post_bench)
        self.state.save(self.root / self.config.state_file)

        await self.ac.memory.store_generation_outcome(
            generation=gen,
            action=plan.get("title", "unknown"),
            accepted=accepted,
            benchmark_delta=comparison.get("total_delta", 0),
            review_summary=review.summary,
            files_changed=[
                c.get("file", "unknown") for c in implementation.get("changes", [])
            ],
        )
        for finding in review.findings:
            await self.ac.memory.store_review_pattern(
                finding_category=finding.category,
                finding_severity=finding.severity.value,
                was_valid=not accepted,
                description=finding.description,
            )

        if gen % 5 == 0:
            logger.info("Phase 9: Running retrospective...")
            try:
                await self._run_retrospective()
            except Exception as e:
                logger.error(f"Retrospective failed (non-fatal): {e}")

        self.git.tag_generation(gen, post_bench.total_score)
        gen_span.end()

        # Finalize and save report
        report.accepted = accepted
        report.decision_reason = (
            f"delta={comparison.get('total_delta', 0):+.1f}, "
            f"review={'approved' if review.approved else 'rejected'}"
        )
        report.finalize()
        self._save_report(report)

        logger.info(f"═══ Generation {gen} complete ═══")
        return entry

    def _save_report(self, report: GenerationReport) -> None:
        """Save a generation report to the reports/ directory."""
        try:
            report.finalize()
            reports_dir = self.root / "reports"
            path = report.save(reports_dir)
            logger.info(f"Generation report saved: {path}")
        except Exception as e:
            logger.warning(f"Failed to save generation report: {e}")

    def _cleanup_on_failure(self, branch: Optional[str] = None) -> None:
        """Safely return to main branch and clean up orphan branches."""
        try:
            current = self.git.current_branch
            if current not in ("main", "master"):
                self.git.switch_to_main()
            if branch and branch not in ("main", "master"):
                self.git._run("branch", "-D", branch, check=False)
        except Exception as cleanup_err:
            logger.error(f"Cleanup failed: {cleanup_err}")

    def _record_failure(
        self, gen: int, phase: str, error: str, gen_span: object,
    ) -> EvolutionEntry:
        """Record a failed generation and return the entry."""
        entry = EvolutionEntry(
            generation=gen,
            action=f"FAILED:{phase}",
            target_file="N/A",
            rationale=f"Phase '{phase}' failed: {error[:200]}",
            benchmark_delta=0,
            accepted=False,
        )
        self.ac.observability.record_decision(gen_span, "failed", f"{phase}: {error[:100]}")
        gen_span.end()
        self.state.generation = gen
        self.state.total_prs_created += 1
        self.state.total_prs_rejected += 1
        self.state.evolution_log.append(entry)
        self.state.save(self.root / self.config.state_file)
        logger.info(f"═══ Generation {gen} FAILED at {phase} ═══")
        return entry

    async def run_loop(self, max_generations: Optional[int] = None) -> None:
        """Run the improvement loop continuously with error recovery."""
        max_gen = max_generations or self.config.max_generations
        consecutive_failures = 0
        while self.state.generation < max_gen:
            try:
                entry = await self.run_generation()
                logger.info(
                    f"Gen {entry.generation}: "
                    f"{'ACCEPTED' if entry.accepted else 'REJECTED'} "
                    f"{entry.action} "
                    f"(delta: {entry.benchmark_delta or 0:+.1f})"
                )
                if entry.action.startswith("FAILED:"):
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
            except Exception as e:
                logger.error(f"Generation failed unexpectedly: {e}", exc_info=True)
                self._cleanup_on_failure()
                consecutive_failures += 1

            if consecutive_failures >= 3:
                logger.error(
                    f"Stopping loop: {consecutive_failures} consecutive failures"
                )
                break

            if consecutive_failures > 0:
                backoff = min(30, 2 ** consecutive_failures)
                logger.info(f"Backoff: waiting {backoff}s before next generation")
                time.sleep(backoff)

    # ─── Private Methods ──────────────────────────────────────────

    async def _llm_complete_json(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        persona: str = "",
    ) -> dict:
        """Call LLM, parse JSON, validate schema, retry once on failure."""
        response = await self.llm.complete(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = self._parse_json_response(response)

        if "error" in parsed and "raw" in parsed:
            logger.info("JSON parse failed — retrying with explicit JSON instruction")
            retry_msg = (
                "Your previous response could not be parsed as JSON. "
                "Please respond with ONLY valid JSON, no explanation.\n\n"
                f"Original request:\n{user_message}"
            )
            response = await self.llm.complete(
                system_prompt=system_prompt,
                user_message=retry_msg,
                temperature=max(0.1, temperature - 0.2),
                max_tokens=max_tokens,
            )
            parsed = self._parse_json_response(response)

        if persona:
            parsed = self._validate_response_schema(parsed, persona)

        return parsed

    async def _plan_improvement(self, baseline: BenchmarkSuite) -> dict:
        """Use Architect persona to plan the next improvement (no memory)."""
        persona = get_persona(AgentRole.ARCHITECT)
        context = self._build_planning_context(baseline)
        return await self._llm_complete_json(
            system_prompt=persona.get_full_prompt(),
            user_message=context,
            temperature=persona.temperature,
            max_tokens=persona.max_tokens,
            persona="architect",
        )

    async def _plan_improvement_with_memory(self, baseline: BenchmarkSuite) -> dict:
        """Use Architect persona + AgentCore Memory to plan improvement.

        Recalls similar past attempts to avoid repeating failures
        and to build on successful patterns.
        """
        persona = get_persona(AgentRole.ARCHITECT)
        context = self._build_planning_context(baseline)

        # ── AgentCore Memory: enrich planning context with past experience ──
        past_attempts = await self.ac.memory.recall_similar_attempts(context, limit=5)
        success_patterns = await self.ac.memory.get_success_patterns(limit=5)

        if past_attempts:
            context += "\n\n## Relevant Past Attempts (from AgentCore Memory)\n"
            for attempt in past_attempts:
                status = "✅ accepted" if attempt.get("accepted") else "❌ rejected"
                context += (
                    f"- Gen {attempt.get('generation', '?')}: "
                    f"{attempt.get('action', '?')} → {status} "
                    f"(delta: {attempt.get('benchmark_delta', 0):+.1f})\n"
                )

        if success_patterns:
            context += "\n## Top Success Patterns\n"
            for pattern in success_patterns[:3]:
                context += (
                    f"- {pattern.get('action', '?')} "
                    f"(delta: +{pattern.get('benchmark_delta', 0):.1f}): "
                    f"files={pattern.get('files_changed', [])}\n"
                )

        return await self._llm_complete_json(
            system_prompt=persona.get_full_prompt(),
            user_message=context,
            temperature=persona.temperature,
            max_tokens=persona.max_tokens,
            persona="architect",
        )

    async def _implement_change(self, plan: dict) -> dict:
        """Use Builder persona to implement the planned change."""
        persona = get_persona(AgentRole.BUILDER)
        context = json.dumps(plan, indent=2)
        return await self._llm_complete_json(
            system_prompt=persona.get_full_prompt(),
            user_message=f"Implement this improvement plan:\n\n{context}",
            temperature=persona.temperature,
            max_tokens=8192,
            persona="builder",
        )

    async def _review_change(self, pr: PullRequest) -> ReviewResult:
        """Use Reviewer persona to review the PR."""
        persona = get_persona(AgentRole.REVIEWER)
        diff = self.git.get_diff()
        context = (
            f"## PR: {pr.title}\n"
            f"**Description**: {pr.description}\n"
            f"**Files Changed**: {', '.join(pr.files_changed)}\n\n"
            f"## Diff\n```\n{diff[:8000]}\n```"
        )
        review_data = await self._llm_complete_json(
            system_prompt=persona.get_full_prompt(),
            user_message=f"Review this pull request:\n\n{context}",
            temperature=persona.temperature,
            max_tokens=persona.max_tokens,
            persona="reviewer",
        )
        return self._build_review_result(pr.pr_id, review_data)

    async def _run_retrospective(self) -> None:
        """Run retrospective analysis with AgentCore Memory enrichment.

        Uses episodic memory to identify long-term patterns that
        wouldn't be visible in a single session's evolution log.
        """
        persona = get_persona(AgentRole.RETROSPECTIVE)
        recent = self.state.evolution_log[-10:]

        # ── AgentCore Memory: pull success patterns for richer retrospective ──
        success_patterns = await self.ac.memory.get_success_patterns(limit=10)

        context = (
            f"## Evolution Log (Last 10 Generations)\n"
            f"{json.dumps([e.__dict__ for e in recent], indent=2, default=str)}\n\n"
            f"## Current Best Score: {self.state.current_best_score:.1f}\n"
            f"## Acceptance Rate: {self._acceptance_rate():.1%}\n"
        )

        if success_patterns:
            context += "\n## Historical Success Patterns (from AgentCore Memory)\n"
            for p in success_patterns[:5]:
                context += (
                    f"- {p.get('action', '?')}: delta=+{p.get('benchmark_delta', 0):.1f}, "
                    f"files={p.get('files_changed', [])}\n"
                )

        # ── AgentCore Policy: include safety rules context ──
        policies = self.ac.policy.get_policies_summary()
        context += f"\n## Active Safety Policies: {len(policies)}\n"
        for pol in policies:
            context += f"- {pol['name']}: {pol['description']}\n"

        # ── AgentCore Observability health ──
        health = self.ac.health_check()
        context += f"\n## AgentCore Services Health: {health}\n"

        retro = await self._llm_complete_json(
            system_prompt=persona.get_full_prompt(),
            user_message=f"Analyze our progress and recommend process changes:\n\n{context}",
            temperature=persona.temperature,
            max_tokens=persona.max_tokens,
        )
        logger.info(f"Retrospective insights: {json.dumps(retro, indent=2)}")

    def _apply_changes(self, implementation: dict) -> None:
        """Write implementation changes to disk."""
        for change in implementation.get("changes", []):
            file_path = self.root / change["file"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(change["content"])

        for test in implementation.get("test_changes", []):
            test_path = self.root / test["file"]
            test_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.write_text(test["content"])

    def _make_merge_decision(self, pr: PullRequest, comparison: dict) -> bool:
        """Decide whether to merge based on reviews and benchmarks."""
        # Check review approval
        if pr.has_blocking:
            logger.info("Merge blocked: blocking review findings")
            return False

        # Check benchmark regression
        total_delta = comparison.get("total_delta", 0)
        if total_delta < -self.config.benchmark_threshold:
            logger.info(f"Merge blocked: benchmark regression ({total_delta:+.1f})")
            return False

        # Check for regressions in individual benchmarks
        regressions = comparison.get("regressed", [])
        critical_regressions = [r for r in regressions if abs(r["delta"]) > 10]
        if critical_regressions:
            logger.info(f"Merge blocked: critical regressions in {[r['name'] for r in critical_regressions]}")
            return False

        return pr.all_approved or self.config.auto_merge

    def _build_planning_context(self, baseline: BenchmarkSuite) -> str:
        """Build context for the planning phase."""
        report = self.benchmarks.format_report(baseline)
        recent = self.state.evolution_log[-5:]
        recent_str = "\n".join(
            f"  Gen {e.generation}: {'✅' if e.accepted else '❌'} {e.action} (delta: {e.benchmark_delta or 0:+.1f})"
            for e in recent
        )
        return (
            f"## Current Benchmark Scores\n{report}\n\n"
            f"## Recent History\n{recent_str}\n\n"
            f"## Current Generation: {self.state.generation}\n"
            f"## Best Score Ever: {self.state.current_best_score:.1f}\n\n"
            f"Based on this data, what is the highest-impact improvement "
            f"you can make to this codebase?"
        )

    def _format_pr_description(self, plan: dict, baseline: BenchmarkSuite) -> str:
        """Format a PR description from the plan."""
        return (
            f"## {plan.get('title', 'Improvement')}\n\n"
            f"**Type**: {plan.get('improvement_type', 'unknown')}\n"
            f"**Rationale**: {plan.get('rationale', 'N/A')}\n\n"
            f"### Baseline Scores\n"
            f"Total: {baseline.total_score:.1f}/{baseline.total_max:.1f}\n"
            f"Pass Rate: {baseline.pass_rate:.1%}\n\n"
            f"### Expected Impact\n"
            f"{json.dumps(plan.get('expected_benchmark_impact', {}), indent=2)}"
        )

    def _build_review_result(self, pr_id: str, data: dict) -> ReviewResult:
        """Convert raw review data to ReviewResult model."""
        findings = []
        for f in data.get("findings", []):
            try:
                findings.append(ReviewFinding(
                    severity=Severity(f.get("severity", "info")),
                    category=f.get("category", "general"),
                    file_path=f.get("file_path", "unknown"),
                    line_start=f.get("line_start", 0),
                    line_end=f.get("line_end", 0),
                    description=f.get("description", ""),
                    suggestion=f.get("suggestion", ""),
                    auto_fixable=f.get("auto_fixable", False),
                ))
            except (ValueError, KeyError):
                continue

        return ReviewResult(
            pr_id=pr_id,
            reviewer_role=AgentRole.REVIEWER,
            approved=data.get("approved", False),
            findings=findings,
            summary=data.get("summary", ""),
            score=data.get("score", 0.0),
        )

    def _acceptance_rate(self) -> float:
        """Calculate the historical acceptance rate."""
        if not self.state.evolution_log:
            return 0.0
        accepted = sum(1 for e in self.state.evolution_log if e.accepted)
        return accepted / len(self.state.evolution_log)

    @staticmethod
    def _parse_json_response(response: str) -> dict:
        """Extract JSON from an LLM response, handling real-world edge cases.

        Handles: markdown fences (```json and ```), leading/trailing text,
        multiple JSON blocks (takes first valid), completely missing JSON.
        """
        text = response.strip()

        # Strategy 1: Extract from ```json ... ``` fences
        json_fence = re.search(r"```json\s*\n?(.*?)```", text, re.DOTALL)
        if json_fence:
            try:
                return json.loads(json_fence.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 2: Extract from ``` ... ``` fences (no language tag)
        plain_fence = re.search(r"```\s*\n?(.*?)```", text, re.DOTALL)
        if plain_fence:
            try:
                return json.loads(plain_fence.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find first valid JSON object by brace matching
        parsed = ImprovementLoop._extract_first_json_object(text)
        if parsed is not None:
            return parsed

        # Strategy 4: Try the whole text as-is
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        logger.warning(f"Failed to parse JSON from response: {text[:200]}...")
        return {"error": "Failed to parse response", "raw": text[:500]}

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[dict]:
        """Find the first valid JSON object in text using brace matching."""
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escape_next = False
            for i in range(start, len(text)):
                ch = text[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\":
                    if in_string:
                        escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            start = text.find("{", start + 1)
        return None

    @staticmethod
    def _validate_response_schema(data: dict, persona: str) -> dict:
        """Validate that parsed JSON has the expected keys for a persona.

        Returns the data if valid, or a safe default that causes rejection.
        """
        if "error" in data and "raw" in data:
            return data

        if persona == "architect":
            required = {"title", "files_to_modify"}
            if not required.issubset(data.keys()):
                logger.warning(f"Architect response missing keys: {required - data.keys()}")
                return {
                    "title": data.get("title", "invalid-plan"),
                    "files_to_modify": data.get("files_to_modify", []),
                    "improvement_type": data.get("improvement_type", "unknown"),
                    "rationale": data.get("rationale", "Schema validation failed"),
                    "_schema_error": True,
                }

        elif persona == "builder":
            if "changes" not in data or not isinstance(data.get("changes"), list):
                logger.warning("Builder response missing 'changes' list")
                return {
                    "changes": [],
                    "test_changes": [],
                    "_schema_error": True,
                }
            for change in data["changes"]:
                if not isinstance(change, dict):
                    continue
                if "file" not in change or "content" not in change:
                    logger.warning(f"Builder change missing file/content: {change.keys()}")
                    return {
                        "changes": [],
                        "test_changes": [],
                        "_schema_error": True,
                    }

        elif persona == "reviewer":
            valid = True
            if "approved" not in data or not isinstance(data.get("approved"), bool):
                valid = False
            if "score" not in data:
                valid = False
            elif not isinstance(data.get("score"), (int, float)):
                valid = False
            if "findings" not in data or not isinstance(data.get("findings"), list):
                valid = False
            if not valid:
                logger.warning("Reviewer response has invalid schema")
                return {
                    "approved": False,
                    "score": 0.0,
                    "findings": [],
                    "summary": "Schema validation failed — auto-rejecting",
                    "_schema_error": True,
                }

        return data
