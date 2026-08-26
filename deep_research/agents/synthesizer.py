from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass

from ..domain.models import EvidenceClaim, EvidenceObservation


class CitationError(ValueError):
    pass


MAX_REPORT_WORDS = 1_800


@dataclass(frozen=True, slots=True)
class SynthesisContext:
    packet: list[dict[str, object]]
    source_urls: dict[str, str]
    claims_by_id: dict[str, EvidenceClaim]
    allowed_sources_by_claim: dict[str, set[str]]
    contested_claim_ids: frozenset[str]
    omitted_contested_count: int


def build_synthesis_context(
    claims: list[EvidenceClaim],
    observations: list[EvidenceObservation],
    *,
    critic_contested_claim_ids: list[str] | None = None,
    max_claims: int = 24,
    max_contested_claims: int = 6,
    max_contested_title_words: int = 300,
    max_sources_per_polarity: int = 3,
) -> SynthesisContext:
    urls = sorted({observation.source_url for observation in observations})
    url_to_id = {url: f"S{index}" for index, url in enumerate(urls, start=1)}
    source_urls = {source_id: url for url, source_id in url_to_id.items()}
    observations_by_id = {
        observation.observation_id: observation for observation in observations
    }
    confidence_rank = {"High": 0, "Moderate": 1, "Low": 2, "Insufficient": 3}
    critic_contested = set(critic_contested_claim_ids or [])
    effective_contested = {
        claim.claim_id
        for claim in claims
        if claim.disagreement_flag
        or (
            claim.claim_id in critic_contested
            and bool(claim.contradicting_observations)
        )
    }
    ranked_claims = sorted(
        claims,
        key=lambda claim: (
            0 if claim.claim_id in effective_contested else 1,
            confidence_rank[claim.confidence_tag.value],
            -claim.supporting_domain_count,
            claim.claim_id,
        ),
    )
    contested_candidates = [
        claim for claim in ranked_claims if claim.claim_id in effective_contested
    ]
    contested: list[EvidenceClaim] = []
    contested_title_words = 0
    for claim in contested_candidates:
        title_words = len(claim.text.split())
        if contested and contested_title_words + title_words > max_contested_title_words:
            continue
        contested.append(claim)
        contested_title_words += title_words
        if len(contested) >= max_contested_claims:
            break
    uncontested = [
        claim for claim in ranked_claims if claim.claim_id not in effective_contested
    ]
    selected_ids = {claim.claim_id for claim in contested}
    balanced: list[EvidenceClaim] = []
    task_ids = sorted(
        {task_id for claim in uncontested for task_id in claim.task_ids},
        key=lambda task_id: (not task_id.startswith("T"), task_id),
    )
    task_buckets = {
        task_id: [claim for claim in uncontested if task_id in claim.task_ids]
        for task_id in task_ids
    }
    while len(contested) + len(balanced) < max_claims:
        added = False
        for task_id in task_ids:
            bucket = task_buckets[task_id]
            while bucket and bucket[0].claim_id in selected_ids:
                bucket.pop(0)
            if not bucket:
                continue
            claim = bucket.pop(0)
            selected_ids.add(claim.claim_id)
            balanced.append(claim)
            added = True
            if len(contested) + len(balanced) >= max_claims:
                break
        if not added:
            break
    for claim in uncontested:
        if len(contested) + len(balanced) >= max_claims:
            break
        if claim.claim_id not in selected_ids:
            selected_ids.add(claim.claim_id)
            balanced.append(claim)
    ranked = contested + balanced
    packet: list[dict[str, object]] = []
    allowed_sources_by_claim: dict[str, set[str]] = {}
    for claim in ranked:
        support = [
            observations_by_id[item]
            for item in claim.supporting_observations
            if item in observations_by_id
        ]
        contradict = [
            observations_by_id[item]
            for item in claim.contradicting_observations
            if item in observations_by_id
        ]
        neutral = [
            observations_by_id[item]
            for item in claim.neutral_observations
            if item in observations_by_id
        ]
        all_observations = support + contradict + neutral
        allowed_sources_by_claim[claim.claim_id] = {
            url_to_id[observation.source_url] for observation in all_observations
        }

        def compact(items: list[EvidenceObservation]) -> list[dict[str, str]]:
            selected: list[EvidenceObservation] = []
            selected_ids: set[str] = set()
            seen_domains: set[str] = set()
            for item in items:
                if item.source_domain in seen_domains:
                    continue
                selected.append(item)
                selected_ids.add(item.observation_id)
                seen_domains.add(item.source_domain)
                if len(selected) >= max_sources_per_polarity:
                    break
            if len(selected) < max_sources_per_polarity:
                for item in items:
                    if item.observation_id in selected_ids:
                        continue
                    selected.append(item)
                    if len(selected) >= max_sources_per_polarity:
                        break
            return [
                {
                    "source_id": url_to_id[item.source_url],
                    "excerpt": item.excerpt,
                    "source_type": item.source_type or "other",
                }
                for item in selected
            ]

        packet.append(
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "confidence": claim.confidence_tag.value,
                "supporting_domain_count": claim.supporting_domain_count,
                "contradicting_domain_count": claim.contradicting_domain_count,
                "disagreement": claim.claim_id in effective_contested,
                "support": compact(support),
                "contradiction": compact(contradict),
                "neutral": compact(neutral),
            }
        )
    return SynthesisContext(
        packet=packet,
        source_urls=source_urls,
        claims_by_id={claim.claim_id: claim for claim in ranked},
        allowed_sources_by_claim=allowed_sources_by_claim,
        contested_claim_ids=frozenset(
            claim.claim_id for claim in contested
        ),
        omitted_contested_count=len(contested_candidates) - len(contested),
    )


def repair_report_citations(
    payload: dict[str, object],
    context: SynthesisContext,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    repaired = deepcopy(payload)
    repairs: list[dict[str, object]] = []
    for section in ("main_findings", "contested_findings", "weak_evidence"):
        items = repaired.get(section, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            claim_id = str(item.get("claim_id", ""))
            allowed = context.allowed_sources_by_claim.get(claim_id)
            source_ids = item.get("source_ids")
            if allowed is None or not isinstance(source_ids, list) or any(
                not isinstance(source_id, str) for source_id in source_ids
            ):
                continue
            removed = set(source_ids) - allowed
            valid_source_ids = set(source_ids) & allowed
            synthesis = str(item.get("synthesis", ""))

            def replace_citation(match: re.Match[str]) -> str:
                source_id = match.group(1)
                if source_id in allowed:
                    valid_source_ids.add(source_id)
                    return match.group(0)
                removed.add(source_id)
                return ""

            synthesis = re.sub(r"\[\s*(S\d+)\s*\]", replace_citation, synthesis)
            synthesis = re.sub(r"\(\s*\)|\[\s*\]", "", synthesis)
            item["synthesis"] = " ".join(synthesis.split())
            item["source_ids"] = sorted(
                valid_source_ids,
                key=lambda source_id: int(source_id[1:]),
            )
            if removed:
                repairs.append(
                    {
                        "claim_id": claim_id,
                        "removed_source_count": len(removed),
                    }
                )
    return repaired, repairs


def build_fallback_report_payload(
    context: SynthesisContext,
    remaining_gaps: list[str],
) -> dict[str, object]:
    """Build a bounded, fully source-bound report when model repair remains invalid."""
    main: list[dict[str, object]] = []
    contested: list[dict[str, object]] = []
    weak: list[dict[str, object]] = []
    finding_word_budget = 900
    finding_words = 0
    for packet_claim in context.packet:
        claim_id = str(packet_claim["claim_id"])
        claim = context.claims_by_id[claim_id]
        allowed = context.allowed_sources_by_claim[claim_id]
        if not allowed:
            continue
        source_ids: list[str] = []
        for polarity in ("support", "contradiction", "neutral"):
            observations = packet_claim.get(polarity, [])
            if not isinstance(observations, list):
                continue
            for observation in observations[:2]:
                if not isinstance(observation, dict):
                    continue
                source_id = str(observation.get("source_id", ""))
                if source_id in allowed and source_id not in source_ids:
                    source_ids.append(source_id)
        if not source_ids:
            source_ids = sorted(allowed, key=lambda item: int(item[1:]))[:2]
        statement = _truncate_words(_display_claim_text(claim.text), 35)
        finding = {
            "claim_id": claim_id,
            "synthesis": (
                f"Sources disagree about this finding: {statement}"
                if claim_id in context.contested_claim_ids
                else statement
            ),
            "source_ids": source_ids,
        }
        if claim_id in context.contested_claim_ids:
            contested.append(finding)
            finding_words += len(claim.text.split()) + len(statement.split()) + 20
        elif finding_words + len(claim.text.split()) + len(statement.split()) + 20 > finding_word_budget:
            continue
        elif claim.confidence_tag.value == "Insufficient":
            if len(weak) < 4:
                weak.append(finding)
                finding_words += len(claim.text.split()) + len(statement.split()) + 20
        elif len(main) < 8:
            main.append(finding)
            finding_words += len(claim.text.split()) + len(statement.split()) + 20

    summary_items = [*main[:2], *contested[:1], *weak[:1]]
    summary = _truncate_words(
        " ".join(str(item["synthesis"]) for item in summary_items),
        100,
    ) or "Available evidence is insufficient to support a reliable answer."
    return {
        "executive_summary": summary,
        "main_findings": main,
        "contested_findings": contested,
        "weak_evidence": weak,
        "remaining_gaps": [_truncate_words(gap, 40) for gap in remaining_gaps[:6]],
    }


def contextualize_remaining_gaps(
    context: SynthesisContext,
    remaining_gaps: list[str],
) -> list[str]:
    gaps = [
        _truncate_words(cleaned, 40)
        for gap in remaining_gaps
        if (cleaned := _strip_internal_claim_ids(gap))
    ]
    if context.omitted_contested_count:
        notice = (
            f"{context.omitted_contested_count} additional contested findings "
            "remain outside this concise report."
        )
        return [*gaps[:5], notice]
    return gaps[:6]


def _truncate_words(value: str, limit: int) -> str:
    words = value.split()
    return " ".join(words[:limit]) + ("…" if len(words) > limit else "")


def _strip_internal_claim_ids(value: str) -> str:
    claim_id = r"C[0-9a-fA-F]{10}"
    cleaned = re.sub(
        rf"[\[(]\s*{claim_id}(?:\s*,\s*{claim_id})*\s*[\])]",
        "",
        value,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(rf"`?\b{claim_id}\b`?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"``", "", cleaned)
    cleaned = re.sub(r"\(\s*\)|\[\s*\]", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return " ".join(cleaned.split())


def _display_claim_text(value: str) -> str:
    without_ids = re.sub(r"\[\s*S\d+\s*\]", "", value)
    without_empty_marks = re.sub(r"\(\s*\)|\[\s*\]", "", without_ids)
    return " ".join(without_empty_marks.split())


def render_report(
    payload: dict[str, object],
    context: SynthesisContext,
    *,
    remaining_gaps: list[str] | None = None,
) -> str:
    _validate_section_membership(payload, context)
    citation_payload = {
        key: value for key, value in payload.items() if key != "remaining_gaps"
    }
    serialized = json.dumps(citation_payload, ensure_ascii=False)
    cited_in_text = _inline_source_ids(serialized)
    unknown = cited_in_text - set(context.source_urls)
    if unknown:
        raise CitationError(f"unknown source IDs: {sorted(unknown)}")
    summary_citations = _inline_source_ids(str(payload.get("executive_summary", "")))
    if summary_citations:
        raise CitationError("executive summary cannot contain unscoped source IDs")

    sections: list[str] = [
        "# Research Report",
        "",
        "## Executive Summary",
        "",
        str(payload.get("executive_summary", "No supported summary available.")),
        "",
        "## Main Findings",
        "",
    ]
    cited: set[str] = set()
    _render_claim_findings(
        sections,
        payload.get("main_findings", []),
        context,
        cited,
    )
    sections.extend(["## Contested Findings", ""])
    _render_claim_findings(
        sections,
        payload.get("contested_findings", []),
        context,
        cited,
    )
    sections.extend(["## Weak / Insufficient Evidence", ""])
    weak_items = payload.get("weak_evidence", [])
    if not isinstance(weak_items, list) or not weak_items:
        sections.extend(["None identified.", ""])
    else:
        for item in weak_items:
            if not isinstance(item, dict):
                raise ValueError("weak evidence item must be an object")
            claim = _validated_claim_item(item, context, cited)
            sections.extend(
                [
                    f"### {_display_claim_text(claim.text)}",
                    "",
                    f"- Confidence: {claim.confidence_tag.value}",
                    f"- Support: {claim.supporting_domain_count} distinct domains",
                    f"- Contradiction: {claim.contradicting_domain_count} distinct domains",
                    f"- Sources: {', '.join(_claim_source_ids(item))}",
                    "",
                    str(item.get("synthesis", "Evidence remains insufficient.")),
                    "",
                ]
            )
    unbound_citations = cited_in_text - cited
    if unbound_citations:
        raise CitationError(
            f"source IDs are not bound to a claim finding: {sorted(unbound_citations)}"
        )
    sections.extend(["## Remaining Gaps", ""])
    gaps = (
        remaining_gaps
        if remaining_gaps is not None
        else payload.get("remaining_gaps", [])
    )
    if isinstance(gaps, list) and gaps:
        sections.extend(f"- {gap}" for gap in gaps)
    else:
        sections.append("- No material gaps identified by final critic check.")
    sections.extend(["", "## Sources", ""])
    if cited:
        for source_id in sorted(cited, key=lambda item: int(item[1:])):
            sections.append(f"- {source_id}: {context.source_urls[source_id]}")
    else:
        sections.append("- No sources cited because evidence was insufficient.")
    report = "\n".join(sections).strip() + "\n"
    word_count = len(report.split())
    if word_count > MAX_REPORT_WORDS:
        raise ValueError(
            f"report exceeds {MAX_REPORT_WORDS}-word limit: {word_count} words"
        )
    return report


def _validate_section_membership(
    payload: dict[str, object],
    context: SynthesisContext,
) -> None:
    section_ids: dict[str, set[str]] = {}
    for section in ("main_findings", "contested_findings", "weak_evidence"):
        raw_items = payload.get(section, [])
        if not isinstance(raw_items, list):
            raise ValueError(f"{section} must be a list")
        ids = {
            str(item.get("claim_id", ""))
            for item in raw_items
            if isinstance(item, dict)
        }
        section_ids[section] = ids
    disputed = set(context.contested_claim_ids)
    misplaced = disputed & (
        section_ids["main_findings"] | section_ids["weak_evidence"]
    )
    if misplaced:
        raise ValueError(f"disputed claims outside contested_findings: {sorted(misplaced)}")
    non_disputed = section_ids["contested_findings"] - disputed
    if non_disputed:
        raise ValueError(f"non-disputed claims in contested_findings: {sorted(non_disputed)}")
    missing = disputed - section_ids["contested_findings"]
    if missing:
        raise ValueError(f"disputed claims missing from contested_findings: {sorted(missing)}")


def _render_claim_findings(
    sections: list[str],
    raw_items: object,
    context: SynthesisContext,
    cited: set[str],
) -> None:
    if not isinstance(raw_items, list) or not raw_items:
        sections.extend(["None identified.", ""])
        return
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("finding must be an object")
        claim = _validated_claim_item(item, context, cited)
        sections.extend(
            [
                f"### {_display_claim_text(claim.text)}",
                "",
                f"- Confidence: {claim.confidence_tag.value}",
                f"- Support: {claim.supporting_domain_count} distinct domains",
                f"- Contradiction: {claim.contradicting_domain_count} distinct domains",
                f"- Sources: {', '.join(_claim_source_ids(item))}",
                "",
                str(item["synthesis"]),
                "",
            ]
        )


def _validated_claim_item(
    item: dict[str, object],
    context: SynthesisContext,
    cited: set[str],
) -> EvidenceClaim:
    claim_id = str(item.get("claim_id", ""))
    claim = context.claims_by_id.get(claim_id)
    if claim is None:
        raise ValueError(f"unknown claim ID: {claim_id}")
    source_ids = item.get("source_ids", [])
    if not isinstance(source_ids, list) or any(
        not isinstance(source_id, str) for source_id in source_ids
    ):
        raise CitationError("source_ids must be a string list")
    text_source_ids = _inline_source_ids(str(item.get("synthesis", "")))
    claim_source_ids = set(source_ids) | text_source_ids
    if not claim_source_ids:
        raise CitationError(f"claim finding has no citation: {claim_id}")
    invalid = claim_source_ids - context.allowed_sources_by_claim[claim_id]
    if invalid:
        raise CitationError(
            f"source IDs do not support claim {claim_id}: {sorted(invalid)}"
        )
    cited.update(claim_source_ids)
    return claim


def _claim_source_ids(item: dict[str, object]) -> list[str]:
    source_ids = {
        *[str(source_id) for source_id in item.get("source_ids", [])],
        *_inline_source_ids(str(item.get("synthesis", ""))),
    }
    return sorted(source_ids, key=lambda source_id: int(source_id[1:]))


def _inline_source_ids(value: str) -> set[str]:
    return set(re.findall(r"\[\s*(S\d+)\s*\]", value))
