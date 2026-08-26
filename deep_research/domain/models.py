from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import urlsplit

ResearchMode = Literal["fast", "deep"]


class Priority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Polarity(StrEnum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    NEUTRAL = "neutral"


class ConfidenceTag(StrEnum):
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    INSUFFICIENT = "Insufficient"


class FetchStatus(StrEnum):
    FETCHED = "fetched"
    FAILED = "failed"
    SKIPPED_DUPLICATE = "skipped_duplicate"


def _required(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


@dataclass(slots=True)
class ResearchTask:
    id: str
    question: str
    rationale: str
    priority: Priority
    page_budget_share: float
    parent_task_id: str | None = None
    depth: int = 0
    status: TaskStatus = TaskStatus.PENDING

    def __post_init__(self) -> None:
        _required(self.id, "id")
        _required(self.question, "question")
        _required(self.rationale, "rationale")
        if self.depth not in (0, 1):
            raise ValueError("task depth must be 0 or 1")
        if self.depth == 1 and not self.parent_task_id:
            raise ValueError("depth-1 task requires parent_task_id")
        if self.depth == 0 and self.parent_task_id is not None:
            raise ValueError("primary task cannot have parent_task_id")
        if not 0 < self.page_budget_share <= 1:
            raise ValueError("page_budget_share must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class SearchQuery:
    task_id: str
    query_text: str
    rationale: str

    def __post_init__(self) -> None:
        _required(self.task_id, "task_id")
        _required(self.query_text, "query_text")
        _required(self.rationale, "rationale")


@dataclass(frozen=True, slots=True)
class SearchResult:
    url: str
    title: str
    snippet: str = ""


@dataclass(frozen=True, slots=True)
class PageExploration:
    url: str
    normalized_url: str
    domain: str
    fetch_status: FetchStatus
    task_id: str
    content_hash: str | None = None


@dataclass(frozen=True, slots=True)
class FetchedPage:
    url: str
    normalized_url: str
    domain: str
    title: str
    text: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class EvidenceObservation:
    observation_id: str
    task_id: str
    source_url: str
    source_domain: str
    statement: str
    polarity: Polarity
    excerpt: str
    source_type: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "task_id",
            "source_domain",
            "statement",
            "excerpt",
        ):
            _required(getattr(self, name), name)
        parsed = urlsplit(self.source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source_url must be an HTTP(S) URL")


@dataclass(slots=True)
class EvidenceClaim:
    claim_id: str
    text: str
    supporting_observations: list[str] = field(default_factory=list)
    contradicting_observations: list[str] = field(default_factory=list)
    neutral_observations: list[str] = field(default_factory=list)
    supporting_domain_count: int = 0
    contradicting_domain_count: int = 0
    confidence_tag: ConfidenceTag = ConfidenceTag.INSUFFICIENT
    disagreement_flag: bool = False
    task_ids: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class Critique:
    coverage_by_task: dict[str, str]
    contested_claim_ids: list[str]
    remaining_gaps: list[str]
    followup_tasks: list[ResearchTask]


@dataclass(frozen=True, slots=True)
class ResearchResult:
    task_id: str
    observations: list[EvidenceObservation]
    explorations: list[PageExploration]
    errors: list[str]
    error_code: str | None = None


def to_primitive(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_primitive(item) for item in value]
    return value
