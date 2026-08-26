from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..agents.critic import CriticSynthesizer
from ..agents.planner import InvalidResearchQuery, Planner
from ..agents.researcher import FetchGate, Researcher
from ..domain.budget import BudgetConfig, BudgetManager
from ..domain.ledger import EvidenceLedger
from ..domain.models import (
    Priority,
    ResearchResult,
    ResearchTask,
    TaskStatus,
    to_primitive,
)
from ..domain.urls import UrlRegistry
from ..infrastructure.audit import JsonlAuditLogger
from ..infrastructure.providers import (
    BatchPageExtractor,
    PageFetcher,
    ProviderError,
    SearchClient,
    StructuredModel,
    provider_error_context,
)
from .pipeline import ResearchPipeline


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    run_dir: Path
    report_path: Path
    ledger_path: Path
    events_path: Path
    run_path: Path
    report: str
    status: str


def run_query(
    *,
    query: str,
    output_root: Path,
    config: BudgetConfig,
    model: StructuredModel,
    search: SearchClient,
    fetcher: PageFetcher,
    batch_extractor: BatchPageExtractor | None = None,
    approved_tasks: list[ResearchTask] | None = None,
) -> RunArtifacts:
    if not query.strip():
        raise ValueError("research query must not be empty")
    query_hash = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:8]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / f"{timestamp}-{query_hash}"
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    events_path = run_dir / "events.jsonl"
    report_path = run_dir / "report.md"
    ledger_path = run_dir / "ledger.json"
    run_path = run_dir / "run.json"

    audit = JsonlAuditLogger(events_path)
    budget = BudgetManager(config)
    ledger = EvidenceLedger(audit)
    urls = UrlRegistry()
    pipeline = ResearchPipeline(
        planner=Planner(model, budget, audit),
        researcher=Researcher(
            model=model,
            search=search,
            fetcher=fetcher,
            budget=budget,
            audit=audit,
            urls=urls,
            fetch_gate=FetchGate(config.max_concurrent_fetches),
            batch_extractor=batch_extractor,
            evidence_batch_size=4 if config.max_pages >= 100 else 1,
        ),
        critic=CriticSynthesizer(model, budget, audit),
        ledger=ledger,
        budget=budget,
        audit=audit,
    )
    audit.log("run.started", query=query, budget_config=config)
    try:
        result = pipeline.run(query, approved_tasks=approved_tasks)
        _write_json_atomic(ledger_path, ledger.dump())
        _write_text_atomic(report_path, result.report)
        result_errors = {
            item.task_id: item.errors for item in result.research_results if item.errors
        }
        status = _completion_status(result.tasks, result.research_results)
        run_record = {
            "status": status,
            "query": query,
            "budget_config": config,
            "budget_used": budget.snapshot(),
            "tasks": result.tasks,
            "initial_critique": result.initial_critique,
            "final_critique": result.final_critique,
            "domain_counts": urls.domain_counts(),
            "research_errors": result_errors,
        }
        if status == "failed":
            failure = next(
                (item for item in result.research_results if item.errors),
                None,
            )
            run_record["error"] = (
                failure.errors[0] if failure is not None else "Research could not complete."
            )
            run_record["error_code"] = (
                (failure.error_code or "research_failed")
                if failure is not None
                else "research_failed"
            )
            provider, retryable = provider_error_context(run_record["error_code"])
            run_record["error_provider"] = provider
            run_record["error_retryable"] = retryable
        _write_json_atomic(run_path, run_record)
        audit.log(
            "run.completed" if status != "failed" else "run.failed",
            status=status,
            budget=budget.snapshot(),
            claim_count=len(ledger.claims()),
            observation_count=len(ledger.observations()),
        )
        return RunArtifacts(
            run_dir=run_dir,
            report_path=report_path,
            ledger_path=ledger_path,
            events_path=events_path,
            run_path=run_path,
            report=result.report,
            status=status,
        )
    except Exception as exc:
        audit.log("run.failed", error_type=type(exc).__name__, error=str(exc))
        public_error = (
            exc.public_message if isinstance(exc, ProviderError) else "Research could not complete."
        )
        error_code = exc.code if isinstance(exc, ProviderError) else "research_failed"
        error_provider, error_retryable = provider_error_context(error_code)
        _write_json_atomic(
            ledger_path,
            ledger.dump(),
        )
        _write_json_atomic(
            run_path,
            {
                "status": "failed",
                "query": query,
                "budget_config": config,
                "budget_used": budget.snapshot(),
                "error_type": type(exc).__name__,
                "error": public_error,
                "error_code": error_code,
                "error_provider": error_provider,
                "error_retryable": error_retryable,
            },
        )
        raise


def _completion_status(
    tasks: list[ResearchTask],
    results: list[ResearchResult],
) -> str:
    primary_ids = {task.id for task in tasks if task.depth == 0}
    failed_primary_ids = {
        task.id
        for task in tasks
        if task.id in primary_ids and task.status is TaskStatus.FAILED
    }
    primary_observation_count = sum(
        len(item.observations) for item in results if item.task_id in primary_ids
    )
    if (
        primary_ids
        and failed_primary_ids == primary_ids
        and primary_observation_count == 0
    ):
        return "failed"
    if any(item.errors for item in results):
        return "completed_with_errors"
    return "completed"


def create_plan(
    *,
    query: str,
    output_path: Path,
    config: BudgetConfig,
    model: StructuredModel,
) -> list[ResearchTask]:
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    audit = JsonlAuditLogger(output_path.with_suffix(".events.jsonl"))
    try:
        tasks = Planner(model, BudgetManager(config), audit).plan(query)
    except InvalidResearchQuery as exc:
        _write_json_atomic(
            output_path,
            {"query": query, "rejected": True, "reason": str(exc), "tasks": []},
        )
        return []
    except ProviderError as exc:
        provider, retryable = provider_error_context(exc.code)
        _write_json_atomic(
            output_path,
            {
                "query": query,
                "status": "failed",
                "error": exc.public_message,
                "error_code": exc.code,
                "error_provider": provider,
                "error_retryable": retryable,
                "tasks": [],
            },
        )
        raise
    _write_json_atomic(output_path, {"query": query, "rejected": False, "tasks": tasks})
    return tasks


def load_plan(path: Path, expected_query: str) -> list[ResearchTask]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("query") != expected_query:
        raise ValueError("approved plan does not match research query")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("approved plan tasks are unavailable")
    tasks = [
        ResearchTask(
            id=str(item["id"]),
            question=str(item["question"]),
            rationale=str(item["rationale"]),
            priority=Priority(str(item["priority"])),
            page_budget_share=float(item["page_budget_share"]),
            parent_task_id=item.get("parent_task_id"),
            depth=int(item.get("depth", 0)),
            status=TaskStatus.PENDING,
        )
        for item in raw_tasks
        if isinstance(item, dict)
    ]
    if len(tasks) != 4:
        raise ValueError("approved plan must contain exactly four tasks")
    return tasks


def _write_json_atomic(path: Path, value: object) -> None:
    _write_text_atomic(
        path,
        json.dumps(to_primitive(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)
    temporary.replace(path)
