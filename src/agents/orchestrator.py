"""Sub-agent orchestrator for parallel task execution.

Enables the self-improving agent to spawn concurrent sub-agents,
each with their own persona and LLM context, and gather results.

Supports:
- Fan-out: dispatch N tasks to N sub-agents in parallel
- Fan-in: collect and merge results with configurable strategies
- Timeout per sub-agent with graceful degradation
- Token budget sharing across sub-agents within a generation
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol

from src.agents.personas import AgentPersona, get_persona
from src.core.models import AgentRole

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM interaction."""

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str: ...


class SubAgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class SubAgentTask:
    """A unit of work to be executed by a sub-agent."""
    task_id: str
    persona: AgentPersona
    user_message: str
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def create(
        persona: AgentPersona,
        user_message: str,
        max_tokens: int = 4096,
        timeout_seconds: float = 120.0,
        **metadata: Any,
    ) -> SubAgentTask:
        return SubAgentTask(
            task_id=f"sub-{uuid.uuid4().hex[:8]}",
            persona=persona,
            user_message=user_message,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            metadata=metadata,
        )


@dataclass
class SubAgentResult:
    """Result from a single sub-agent execution."""
    task_id: str
    status: SubAgentStatus
    response: str = ""
    error: str = ""
    duration_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == SubAgentStatus.COMPLETED


class MergeStrategy:
    """Strategies for combining results from parallel sub-agents."""

    @staticmethod
    def collect_all(results: list[SubAgentResult]) -> list[SubAgentResult]:
        """Return all results as-is (no merging)."""
        return results

    @staticmethod
    def first_success(results: list[SubAgentResult]) -> Optional[SubAgentResult]:
        """Return the first successful result."""
        for r in results:
            if r.succeeded:
                return r
        return None

    @staticmethod
    def majority_vote(
        results: list[SubAgentResult],
        extract_vote: Callable[[SubAgentResult], bool],
    ) -> bool:
        """Return True if majority of successful results vote True."""
        votes = [extract_vote(r) for r in results if r.succeeded]
        if not votes:
            return False
        return sum(votes) > len(votes) / 2


class SubAgentOrchestrator:
    """Orchestrates parallel sub-agent execution.

    Each sub-agent gets its own persona prompt and runs as an independent
    asyncio task against the shared LLM provider. Results are gathered
    with configurable timeout and error handling.
    """

    def __init__(
        self,
        llm: LLMProvider,
        max_concurrent: int = 5,
        default_timeout: float = 120.0,
    ):
        self.llm = llm
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.default_timeout = default_timeout

    async def run_parallel(
        self,
        tasks: list[SubAgentTask],
    ) -> list[SubAgentResult]:
        """Execute multiple sub-agent tasks concurrently.

        Returns results in the same order as the input tasks.
        Failed or timed-out tasks return with appropriate status
        rather than raising exceptions.
        """
        if not tasks:
            return []

        logger.info(f"Dispatching {len(tasks)} sub-agent tasks in parallel")
        coros = [self._execute_task(task) for task in tasks]
        results = await asyncio.gather(*coros, return_exceptions=False)
        succeeded = sum(1 for r in results if r.succeeded)
        logger.info(
            f"Sub-agents complete: {succeeded}/{len(tasks)} succeeded"
        )
        return results

    async def run_sequential(
        self,
        tasks: list[SubAgentTask],
    ) -> list[SubAgentResult]:
        """Execute sub-agent tasks one at a time (for dependent work)."""
        results = []
        for task in tasks:
            result = await self._execute_task(task)
            results.append(result)
            if not result.succeeded:
                logger.warning(
                    f"Sub-agent {task.task_id} failed, stopping sequential chain"
                )
                break
        return results

    async def fan_out_fan_in(
        self,
        tasks: list[SubAgentTask],
        merge: Callable[[list[SubAgentResult]], Any] = MergeStrategy.collect_all,
    ) -> Any:
        """Fan-out tasks in parallel, then fan-in with a merge strategy."""
        results = await self.run_parallel(tasks)
        return merge(results)

    async def _execute_task(self, task: SubAgentTask) -> SubAgentResult:
        """Execute a single sub-agent task with timeout and error handling."""
        start = time.monotonic()
        timeout = task.timeout_seconds or self.default_timeout

        async with self._semaphore:
            logger.debug(
                f"Sub-agent {task.task_id} starting "
                f"(persona={task.persona.name}, timeout={timeout}s)"
            )
            try:
                response = await asyncio.wait_for(
                    self.llm.complete(
                        system_prompt=task.persona.get_full_prompt(),
                        user_message=task.user_message,
                        temperature=task.persona.temperature,
                        max_tokens=task.max_tokens,
                    ),
                    timeout=timeout,
                )
                duration = time.monotonic() - start
                logger.debug(
                    f"Sub-agent {task.task_id} completed in {duration:.1f}s"
                )
                return SubAgentResult(
                    task_id=task.task_id,
                    status=SubAgentStatus.COMPLETED,
                    response=response,
                    duration_seconds=duration,
                    metadata=task.metadata,
                )

            except asyncio.TimeoutError:
                duration = time.monotonic() - start
                logger.warning(
                    f"Sub-agent {task.task_id} timed out after {duration:.1f}s"
                )
                return SubAgentResult(
                    task_id=task.task_id,
                    status=SubAgentStatus.TIMED_OUT,
                    error=f"Timed out after {timeout}s",
                    duration_seconds=duration,
                    metadata=task.metadata,
                )

            except Exception as e:
                duration = time.monotonic() - start
                logger.error(
                    f"Sub-agent {task.task_id} failed: {e}", exc_info=True
                )
                return SubAgentResult(
                    task_id=task.task_id,
                    status=SubAgentStatus.FAILED,
                    error=str(e),
                    duration_seconds=duration,
                    metadata=task.metadata,
                )
