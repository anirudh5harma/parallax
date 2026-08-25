from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .audit import JsonlAuditLogger
from .budget import BudgetConfig, BudgetManager
from .critic import CriticSynthesizer
from .ledger import EvidenceLedger
from .models import to_primitive
from .pipeline import ResearchPipeline
from .planner import Planner
from .providers import PageFetcher, SearchClient, StructuredModel
from .researcher import FetchGate, Researcher
from .urls import UrlRegistry


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
) -> RunArtifacts:
    if not query.strip():
        raise ValueError("research query must not be empty")
    query_hash = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:8]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / f"{timestamp}-{query_hash}"
    run_dir.mkdir(parents=True, exist_ok=False)
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
        ),
        critic=CriticSynthesizer(model, budget, audit),
        ledger=ledger,
        budget=budget,
        audit=audit,
    )
    audit.log("run.started", query=query, budget_config=config)
    try:
        result = pipeline.run(query)
        _write_json_atomic(ledger_path, ledger.dump())
        _write_text_atomic(report_path, result.report)
        primary_ids = {task.id for task in result.tasks if task.depth == 0}
        result_errors = {
            item.task_id: item.errors for item in result.research_results if item.errors
        }
        failed_primary_ids = {
            task.id
            for task in result.tasks
            if task.id in primary_ids and task.status.value == "failed"
        }
        if primary_ids and failed_primary_ids == primary_ids:
            status = "failed"
        elif result_errors:
            status = "completed_with_errors"
        else:
            status = "completed"
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
                "error": str(exc),
            },
        )
        raise


def _write_json_atomic(path: Path, value: object) -> None:
    _write_text_atomic(
        path,
        json.dumps(to_primitive(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)
