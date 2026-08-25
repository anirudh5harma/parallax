from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .models import EvidenceClaim, EvidenceObservation


class CitationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SynthesisContext:
    packet: list[dict[str, object]]
    source_urls: dict[str, str]
    claims_by_id: dict[str, EvidenceClaim]
    allowed_sources_by_claim: dict[str, set[str]]


def build_synthesis_context(
    claims: list[EvidenceClaim],
    observations: list[EvidenceObservation],
    *,
    max_claims: int = 60,
    max_sources_per_polarity: int = 5,
) -> SynthesisContext:
    urls = sorted({observation.source_url for observation in observations})
    url_to_id = {url: f"S{index}" for index, url in enumerate(urls, start=1)}
    source_urls = {source_id: url for url, source_id in url_to_id.items()}
    observations_by_id = {
        observation.observation_id: observation for observation in observations
    }
    confidence_rank = {"High": 0, "Moderate": 1, "Low": 2, "Insufficient": 3}
    ranked = sorted(
        claims,
        key=lambda claim: (
            0 if claim.disagreement_flag else 1,
            confidence_rank[claim.confidence_tag.value],
            -claim.supporting_domain_count,
            claim.claim_id,
        ),
    )[:max_claims]
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
            return [
                {
                    "source_id": url_to_id[item.source_url],
                    "excerpt": item.excerpt,
                    "source_type": item.source_type or "other",
                }
                for item in items[:max_sources_per_polarity]
            ]

        packet.append(
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "confidence": claim.confidence_tag.value,
                "supporting_domain_count": claim.supporting_domain_count,
                "contradicting_domain_count": claim.contradicting_domain_count,
                "disagreement": claim.disagreement_flag,
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
    )


def render_report(
    payload: dict[str, object],
    context: SynthesisContext,
    *,
    remaining_gaps: list[str] | None = None,
) -> str:
    serialized = json.dumps(payload, ensure_ascii=False)
    cited_in_text = set(re.findall(r"\bS\d+\b", serialized))
    unknown = cited_in_text - set(context.source_urls)
    if unknown:
        raise CitationError(f"unknown source IDs: {sorted(unknown)}")

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
                    f"### {claim.text}",
                    "",
                    f"- Confidence: {claim.confidence_tag.value}",
                    f"- Support: {claim.supporting_domain_count} distinct domains",
                    f"- Contradiction: {claim.contradicting_domain_count} distinct domains",
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
    payload_gaps = payload.get("remaining_gaps", [])
    if remaining_gaps is not None and payload_gaps != remaining_gaps:
        raise ValueError("report gaps must match the final critic check")
    gaps = remaining_gaps if remaining_gaps is not None else payload_gaps
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
    return "\n".join(sections).strip() + "\n"


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
                f"### {claim.text}",
                "",
                f"- Confidence: {claim.confidence_tag.value}",
                f"- Support: {claim.supporting_domain_count} distinct domains",
                f"- Contradiction: {claim.contradicting_domain_count} distinct domains",
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
    text_source_ids = set(re.findall(r"\bS\d+\b", str(item.get("synthesis", ""))))
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
