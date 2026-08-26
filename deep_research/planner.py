from __future__ import annotations

from typing import Any

from .audit import JsonlAuditLogger
from .budget import BudgetManager
from .models import Priority, ResearchTask
from .providers import StructuredModel


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "disposition": {
            "type": "string",
            "enum": ["researchable", "reject"],
        },
        "reason": {
            "type": "string",
            "minLength": 2,
            "description": "Concise planning or rejection reason, preferably under 40 words.",
        },
        "tasks": {
            "type": "array",
            "minItems": 0,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "minLength": 5},
                    "rationale": {"type": "string", "minLength": 5},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "page_budget_share": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                    },
                },
                "required": ["question", "rationale", "priority", "page_budget_share"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["disposition", "reason", "tasks"],
    "additionalProperties": False,
}


class InvalidResearchQuery(ValueError):
    pass


class Planner:
    def __init__(
        self,
        model: StructuredModel,
        budget: BudgetManager,
        audit: JsonlAuditLogger,
    ) -> None:
        self.model = model
        self.budget = budget
        self.audit = audit

    def plan(self, query: str) -> list[ResearchTask]:
        if not query.strip():
            raise ValueError("research query must not be empty")
        self.budget.check_time()
        payload = self.model.generate_json(
            system_prompt=(
                "You are Planner, one of exactly three roles. Decompose only; do not research. "
                "First decide whether the input is a meaningful, evidence-answerable research "
                "question. Reject gibberish, greetings, action-only requests, requests for "
                "secrets, and instructions unrelated to research. Accept controversial or "
                "sensitive topics when they can be researched from public evidence. For a "
                "researchable input, return four focused, non-overlapping questions. For a "
                "rejected input, return no tasks and a concise rewrite suggestion."
            ),
            user_prompt=(
                f"Original query: {query}\n"
                f"Global page ceiling: {self.budget.config.max_pages}. "
                "Allocate rough shares whose total is close to 1."
            ),
            schema_name="research_plan",
            schema=PLAN_SCHEMA,
            timeout_seconds=self.budget.remaining_seconds(),
        )
        disposition = str(payload["disposition"])
        reason = str(payload["reason"])
        raw_tasks = payload.get("tasks")
        if disposition == "reject":
            if raw_tasks not in ([], None):
                raise ValueError("rejected query must not include research tasks")
            raise InvalidResearchQuery(reason)
        if disposition != "researchable":
            raise ValueError("planner returned an invalid query disposition")
        if not isinstance(raw_tasks, list) or len(raw_tasks) != 4:
            raise ValueError("planner must return exactly four tasks")
        raw_shares = [float(item["page_budget_share"]) for item in raw_tasks]
        total_share = sum(raw_shares)
        if total_share <= 0:
            raise ValueError("planner page shares must have positive total")
        tasks: list[ResearchTask] = []
        for index, (item, raw_share) in enumerate(zip(raw_tasks, raw_shares), start=1):
            self.budget.reserve_primary_task()
            tasks.append(
                ResearchTask(
                    id=f"T{index}",
                    question=str(item["question"]),
                    rationale=str(item["rationale"]),
                    priority=Priority(str(item["priority"])),
                    page_budget_share=raw_share / total_share,
                )
            )
        self.audit.log(
            "planner.plan_created",
            original_query=query,
            tasks=tasks,
            budget=self.budget.snapshot(),
        )
        return tasks

    def accept(self, query: str, tasks: list[ResearchTask]) -> list[ResearchTask]:
        if len(tasks) != 4:
            raise ValueError("approved plan must contain exactly four tasks")
        if any(task.depth != 0 or task.parent_task_id is not None for task in tasks):
            raise ValueError("approved plan may only contain primary tasks")
        if len({task.id for task in tasks}) != len(tasks):
            raise ValueError("approved plan task IDs must be unique")
        for task in tasks:
            self.budget.reserve_primary_task()
        self.audit.log(
            "planner.plan_created",
            original_query=query,
            tasks=tasks,
            approved_preview=True,
            budget=self.budget.snapshot(),
        )
        return tasks
