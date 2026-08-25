from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .audit import JsonlAuditLogger
from .budget import BudgetManager
from .critic import CriticSynthesizer
from .ledger import EvidenceLedger
from .models import Critique, ResearchResult, ResearchTask, TaskStatus
from .planner import Planner
from .researcher import Researcher


@dataclass(frozen=True, slots=True)
class PipelineResult:
    report: str
    ledger: EvidenceLedger
    tasks: list[ResearchTask]
    initial_critique: Critique
    final_critique: Critique


class ResearchPipeline:
    def __init__(
        self,
        *,
        planner: Planner,
        researcher: Researcher,
        critic: CriticSynthesizer,
        ledger: EvidenceLedger,
        budget: BudgetManager,
        audit: JsonlAuditLogger,
    ) -> None:
        self.planner = planner
        self.researcher = researcher
        self.critic = critic
        self.ledger = ledger
        self.budget = budget
        self.audit = audit

    def run(self, query: str) -> PipelineResult:
        tasks = self.planner.plan(query)
        primary_results = self._run_tasks(tasks)
        self._write_results(primary_results)

        can_follow_up = (
            self.budget.remaining_pages() > 0
            and self.budget.remaining_searches() > 0
            and self.budget.config.max_depth == 1
        )
        initial = self.critic.critique(
            original_query=query,
            tasks=tasks,
            claims=self.ledger.claims(),
            allow_followups=can_follow_up,
        )
        followups = initial.followup_tasks
        if followups:
            tasks.extend(followups)
            followup_results = self._run_tasks(followups)
            self._write_results(followup_results)
        self.budget.check_time()
        final = self.critic.critique(
            original_query=query,
            tasks=tasks,
            claims=self.ledger.claims(),
            allow_followups=False,
        )
        report = self.critic.synthesize(
            original_query=query,
            tasks=tasks,
            claims=self.ledger.claims(),
            observations=self.ledger.observations(),
            remaining_gaps=final.remaining_gaps,
        )
        return PipelineResult(
            report=report,
            ledger=self.ledger,
            tasks=tasks,
            initial_critique=initial,
            final_critique=final,
        )

    def _run_tasks(self, tasks: list[ResearchTask]) -> list[ResearchResult]:
        if not tasks:
            return []
        executor = ThreadPoolExecutor(max_workers=min(len(tasks), 4))
        futures: dict[Future[ResearchResult], ResearchTask] = {
            executor.submit(self.researcher.research, task): task for task in tasks
        }
        results: list[ResearchResult] = []
        try:
            for future in as_completed(
                futures, timeout=max(0.001, self.budget.remaining_seconds())
            ):
                task = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    task.status = TaskStatus.FAILED
                    message = f"{type(exc).__name__}: {exc}"
                    self.audit.log("researcher.task_failed", task_id=task.id, error=message)
                    results.append(
                        ResearchResult(
                            task_id=task.id,
                            observations=[],
                            explorations=[],
                            errors=[message],
                        )
                    )
        except TimeoutError:
            for future, task in futures.items():
                if not future.done():
                    future.cancel()
                    task.status = TaskStatus.FAILED
                    results.append(
                        ResearchResult(
                            task_id=task.id,
                            observations=[],
                            explorations=[],
                            errors=["wall-clock timeout exhausted"],
                        )
                    )
            self.audit.log("pipeline.timeout", budget=self.budget.snapshot())
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return sorted(results, key=lambda result: result.task_id)

    def _write_results(self, results: list[ResearchResult]) -> None:
        for result in results:
            self.ledger.add_observations(result.observations)
            self.audit.log(
                "ledger.task_result_written",
                task_id=result.task_id,
                observation_count=len(result.observations),
                exploration_count=len(result.explorations),
                errors=result.errors,
            )
