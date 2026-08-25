from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .audit import JsonlAuditLogger
from .budget import BudgetExceeded, BudgetManager
from .models import (
    Critique,
    EvidenceClaim,
    EvidenceObservation,
    Priority,
    ResearchTask,
)
from .providers import StructuredModel
from .synthesizer import build_synthesis_context, render_report


def _critique_schema(max_followups: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "coverage": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "assessment": {"type": "string"},
                    },
                    "required": ["task_id", "assessment"],
                    "additionalProperties": False,
                },
            },
            "contested_claim_ids": {"type": "array", "items": {"type": "string"}},
            "remaining_gaps": {"type": "array", "items": {"type": "string"}},
            "followups": {
                "type": "array",
                "maxItems": max_followups,
                "items": {
                    "type": "object",
                    "properties": {
                        "parent_task_id": {"type": "string"},
                        "question": {"type": "string", "minLength": 5},
                        "rationale": {"type": "string", "minLength": 5},
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "page_budget_share": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": [
                        "parent_task_id",
                        "question",
                        "rationale",
                        "priority",
                        "page_budget_share",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["coverage", "contested_claim_ids", "remaining_gaps", "followups"],
        "additionalProperties": False,
    }


FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "synthesis": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claim_id", "synthesis", "source_ids"],
    "additionalProperties": False,
}

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "main_findings": {"type": "array", "items": FINDING_SCHEMA},
        "contested_findings": {"type": "array", "items": FINDING_SCHEMA},
        "weak_evidence": {"type": "array", "items": FINDING_SCHEMA},
        "remaining_gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "executive_summary",
        "main_findings",
        "contested_findings",
        "weak_evidence",
        "remaining_gaps",
    ],
    "additionalProperties": False,
}


def _report_schema(
    claim_ids: list[str],
    source_ids: list[str],
) -> dict[str, Any]:
    schema = deepcopy(REPORT_SCHEMA)
    for section in ("main_findings", "contested_findings", "weak_evidence"):
        properties = schema["properties"][section]["items"]["properties"]
        if claim_ids:
            properties["claim_id"]["enum"] = claim_ids
        if source_ids:
            properties["source_ids"]["items"]["enum"] = source_ids
    return schema


class CriticSynthesizer:
    """Combined third role: one bounded critique pass, final check, then synthesis."""

    def __init__(
        self,
        model: StructuredModel,
        budget: BudgetManager,
        audit: JsonlAuditLogger,
    ) -> None:
        self.model = model
        self.budget = budget
        self.audit = audit

    def critique(
        self,
        *,
        original_query: str,
        tasks: list[ResearchTask],
        claims: list[EvidenceClaim],
        allow_followups: bool,
    ) -> Critique:
        max_followups = self.budget.config.max_followup_tasks if allow_followups else 0
        payload = self.model.generate_json(
            system_prompt=(
                "You are Critic/Synthesizer, one of exactly three roles. Assess coverage and "
                "visible disagreement. Emit at most two high-value follow-ups only when allowed. "
                "Do not resolve contradictions silently."
            ),
            user_prompt=json.dumps(
                {
                    "original_query": original_query,
                    "tasks": [
                        {
                            "id": task.id,
                            "question": task.question,
                            "priority": task.priority.value,
                            "depth": task.depth,
                        }
                        for task in tasks
                    ],
                    "claims": [
                        {
                            "claim_id": claim.claim_id,
                            "text": claim.text,
                            "confidence": claim.confidence_tag.value,
                            "supporting_domains": claim.supporting_domain_count,
                            "contradicting_domains": claim.contradicting_domain_count,
                            "disagreement": claim.disagreement_flag,
                            "task_ids": claim.task_ids,
                        }
                        for claim in claims
                    ],
                    "remaining_pages": self.budget.remaining_pages(),
                    "remaining_searches": self.budget.remaining_searches(),
                    "followups_allowed": allow_followups,
                },
                ensure_ascii=False,
            ),
            schema_name="initial_critique" if allow_followups else "final_critique",
            schema=_critique_schema(max_followups),
            timeout_seconds=self.budget.remaining_seconds(),
        )
        raw_followups = payload.get("followups")
        if not isinstance(raw_followups, list) or len(raw_followups) > max_followups:
            raise ValueError("critic exceeded follow-up limit")
        if not allow_followups and raw_followups:
            raise ValueError("final critic check cannot recurse")

        tasks_by_id = {task.id: task for task in tasks}
        followups: list[ResearchTask] = []
        for item in raw_followups:
            parent_id = str(item["parent_task_id"])
            parent = tasks_by_id.get(parent_id)
            if parent is None or parent.depth != 0 or parent.parent_task_id is not None:
                raise ValueError("follow-up parent must be a primary task")
            try:
                self.budget.reserve_followup_task(parent_depth=parent.depth)
            except BudgetExceeded as exc:
                self.audit.log(
                    "critic.followup_rejected",
                    parent_task_id=parent_id,
                    reason=str(exc),
                )
                break
            followup = ResearchTask(
                id=f"F{len(followups) + 1}",
                question=str(item["question"]),
                rationale=str(item["rationale"]),
                priority=Priority(str(item["priority"])),
                page_budget_share=float(item["page_budget_share"]),
                parent_task_id=parent_id,
                depth=1,
            )
            followups.append(followup)
            self.audit.log("critic.followup_created", task=followup)

        coverage: dict[str, str] = {}
        raw_coverage = payload.get("coverage", [])
        if not isinstance(raw_coverage, list):
            raise ValueError("critic coverage must be a list")
        for item in raw_coverage:
            task_id = str(item["task_id"])
            if task_id in tasks_by_id:
                coverage[task_id] = str(item["assessment"])
        for task in tasks:
            coverage.setdefault(task.id, "not assessed")
        valid_claim_ids = {claim.claim_id for claim in claims}
        contested = [
            str(item)
            for item in payload.get("contested_claim_ids", [])
            if str(item) in valid_claim_ids
        ]
        gaps = [str(item) for item in payload.get("remaining_gaps", [])]
        return Critique(
            coverage_by_task=coverage,
            contested_claim_ids=contested,
            remaining_gaps=gaps,
            followup_tasks=followups,
        )

    def synthesize(
        self,
        *,
        original_query: str,
        tasks: list[ResearchTask],
        claims: list[EvidenceClaim],
        observations: list[EvidenceObservation],
        remaining_gaps: list[str],
    ) -> str:
        context = build_synthesis_context(claims, observations)
        if not claims:
            payload: dict[str, object] = {
                "executive_summary": (
                    "No page produced clear, on-topic evidence strong enough to support "
                    "a finding. The run therefore remains insufficient rather than "
                    "inventing a narrative."
                ),
                "main_findings": [],
                "contested_findings": [],
                "weak_evidence": [],
                "remaining_gaps": remaining_gaps,
            }
            report = render_report(payload, context, remaining_gaps=remaining_gaps)
            self.audit.log(
                "synthesis.completed",
                report_word_count=len(report.split()),
                source_count=0,
                budget=self.budget.snapshot(),
            )
            return report
        system_prompt = (
            "You are Critic/Synthesizer, one of exactly three roles. Produce a concise "
            "800-1800 word report. Preserve confidence tiers and disagreement. Use only "
            "provided claim IDs and source IDs. Each finding may cite only source IDs listed "
            "inside that claim's allowed_source_ids. Put source IDs only in finding source_ids "
            "or finding synthesis, never in the executive summary. Never create numeric "
            "confidence scores."
        )
        report_input = {
            "original_query": original_query,
            "planner_tasks": [
                {"id": task.id, "question": task.question} for task in tasks
            ],
            "structured_claims": context.packet,
            "remaining_gaps": remaining_gaps,
        }
        schema = _report_schema(
            sorted(context.claims_by_id),
            sorted(context.source_urls, key=lambda item: int(item[1:])),
        )
        payload = self.model.generate_json(
            system_prompt=system_prompt,
            user_prompt=json.dumps(report_input, ensure_ascii=False),
            schema_name="final_report",
            schema=schema,
            timeout_seconds=self.budget.remaining_seconds(),
        )
        try:
            report = render_report(payload, context, remaining_gaps=remaining_gaps)
        except ValueError as exc:
            self.audit.log(
                "synthesis.validation_retry",
                reason=" ".join(str(exc).split())[:300],
            )
            repair_input = {
                **report_input,
                "invalid_report": payload,
                "validation_error": " ".join(str(exc).split())[:300],
                "repair_instruction": (
                    "Regenerate the complete report once. Correct the validation error. "
                    "Do not add claims or sources."
                ),
            }
            payload = self.model.generate_json(
                system_prompt=system_prompt,
                user_prompt=json.dumps(repair_input, ensure_ascii=False),
                schema_name="final_report",
                schema=schema,
                timeout_seconds=self.budget.remaining_seconds(),
            )
            report = render_report(payload, context, remaining_gaps=remaining_gaps)
        self.audit.log(
            "synthesis.completed",
            report_word_count=len(report.split()),
            source_count=len(context.source_urls),
            budget=self.budget.snapshot(),
        )
        return report
