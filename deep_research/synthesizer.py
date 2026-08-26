from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass

from .models import EvidenceClaim, EvidenceObservation


class CitationError(ValueError):
    pass


MAX_REPORT_WORDS = 1_800


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
                source_id = re.search(r"S\d+", match.group(0))
                if source_id and source_id.group(0) in allowed:
                    valid_source_ids.add(source_id.group(0))
                    return match.group(0)
                if source_id:
                    removed.add(source_id.group(0))
                return ""

            synthesis = re.sub(r"\[?\bS\d+\b\]?", replace_citation, synthesis)
            synthesis = re.sub(r"\(\s*\)|\[\s*\]", "", synthesis)
            item["synthesis"] = " ".join(synthesis.split())
            fallback_added = False
            if not valid_source_ids and allowed:
                valid_source_ids.add(
                    min(allowed, key=lambda source_id: int(source_id[1:]))
                )
                fallback_added = True
            item["source_ids"] = sorted(
                valid_source_ids,
                key=lambda source_id: int(source_id[1:]),
            )
            if removed or fallback_added:
                repairs.append(
                    {
                        "claim_id": claim_id,
                        "removed_source_count": len(removed),
                        "fallback_added": fallback_added,
                    }
                )
    return repaired, repairs


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
    cited_in_text = set(re.findall(r"\bS\d+\b", serialized))
    unknown = cited_in_text - set(context.source_urls)
    if unknown:
        raise CitationError(f"unknown source IDs: {sorted(unknown)}")
    summary_citations = set(
        re.findall(r"\bS\d+\b", str(payload.get("executive_summary", "")))
    )
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
                    f"### {claim.text}",
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
    disputed = {
        claim_id
        for claim_id, claim in context.claims_by_id.items()
        if claim.disagreement_flag
    }
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
                f"### {claim.text}",
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


def _claim_source_ids(item: dict[str, object]) -> list[str]:
    source_ids = {
        *[str(source_id) for source_id in item.get("source_ids", [])],
        *re.findall(r"\bS\d+\b", str(item.get("synthesis", ""))),
    }
    return sorted(source_ids, key=lambda source_id: int(source_id[1:]))
