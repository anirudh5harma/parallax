from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable
from typing import Any, Protocol

from .models import ConfidenceTag, EvidenceClaim, EvidenceObservation, Polarity


class AuditSink(Protocol):
    def log(self, event: str, **data: Any) -> None: ...


def _claim_key(statement: str) -> str:
    normalized = unicodedata.normalize("NFKC", statement).casefold()
    key = " ".join("".join(character if character.isalnum() else " " for character in normalized).split())
    if not key:
        raise ValueError("claim statement has no alphanumeric content")
    return key


class EvidenceLedger:
    """Single-writer in-memory ledger with deterministic exact claim grouping."""

    def __init__(self, audit: AuditSink) -> None:
        self._audit = audit
        self._claims: dict[str, EvidenceClaim] = {}
        self._observations: dict[str, EvidenceObservation] = {}

    def add_observations(self, observations: Iterable[EvidenceObservation]) -> None:
        for observation in observations:
            if observation.observation_id in self._observations:
                continue
            key = _claim_key(observation.statement)
            claim = self._claims.get(key)
            if claim is None:
                digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
                claim = EvidenceClaim(claim_id=f"C{digest}", text=observation.statement)
                self._claims[key] = claim
                self._audit.log("ledger.claim_created", claim=claim)

            self._observations[observation.observation_id] = observation
            target = {
                Polarity.SUPPORT: claim.supporting_observations,
                Polarity.CONTRADICT: claim.contradicting_observations,
                Polarity.NEUTRAL: claim.neutral_observations,
            }[observation.polarity]
            target.append(observation.observation_id)
            if observation.task_id not in claim.task_ids:
                claim.task_ids.append(observation.task_id)
            self._recompute(claim)
            event = (
                "ledger.contradiction_added"
                if observation.polarity is Polarity.CONTRADICT
                else "observation.extracted"
            )
            self._audit.log(event, claim_id=claim.claim_id, observation=observation)

    def _recompute(self, claim: EvidenceClaim) -> None:
        support_domains = {
            self._observations[item].source_domain
            for item in claim.supporting_observations
        }
        contradiction_domains = {
            self._observations[item].source_domain
            for item in claim.contradicting_observations
        }
        claim.supporting_domain_count = len(support_domains)
        claim.contradicting_domain_count = len(contradiction_domains)
        claim.disagreement_flag = bool(support_domains and contradiction_domains)

        if not support_domains:
            claim.confidence_tag = ConfidenceTag.INSUFFICIENT
        elif len(contradiction_domains) >= 2:
            claim.confidence_tag = ConfidenceTag.LOW
        elif len(support_domains) >= 3:
            claim.confidence_tag = ConfidenceTag.HIGH
        elif len(support_domains) >= 2:
            claim.confidence_tag = ConfidenceTag.MODERATE
        else:
            claim.confidence_tag = ConfidenceTag.LOW

    def claims(self) -> list[EvidenceClaim]:
        return sorted(self._claims.values(), key=lambda claim: claim.claim_id)

    def observations(self) -> list[EvidenceObservation]:
        return sorted(
            self._observations.values(), key=lambda observation: observation.observation_id
        )

    def dump(self) -> dict[str, object]:
        return {
            "confidence_rules": {
                "High": "at least 3 supporting domains and at most 1 contradicting domain",
                "Moderate": "2 supporting domains and at most 1 contradicting domain",
                "Low": "1 supporting domain or at least 2 contradicting domains",
                "Insufficient": "no supporting domains",
            },
            "claims": self.claims(),
            "observations": self.observations(),
        }
