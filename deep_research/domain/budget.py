from __future__ import annotations

import threading
import time
from dataclasses import dataclass

MAX_PRIMARY_TASKS = 4
MAX_FOLLOWUP_TASKS = 2
MAX_RESEARCH_TASKS = 6
MAX_SEARCHES = 150
MAX_PAGES = 600
MAX_CONCURRENT_FETCHES = 12
MAX_DEPTH = 1
MAX_TIMEOUT_SECONDS = 1800.0


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    max_primary_tasks: int = 4
    max_followup_tasks: int = 2
    max_research_tasks: int = 6
    max_searches: int = 24
    max_pages: int = 40
    max_concurrent_fetches: int = 8
    max_depth: int = 1
    wall_clock_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        limits = {
            "max_primary_tasks": (self.max_primary_tasks, 1, MAX_PRIMARY_TASKS),
            "max_followup_tasks": (self.max_followup_tasks, 0, MAX_FOLLOWUP_TASKS),
            "max_research_tasks": (self.max_research_tasks, 1, MAX_RESEARCH_TASKS),
            "max_searches": (self.max_searches, 1, MAX_SEARCHES),
            "max_pages": (self.max_pages, 1, MAX_PAGES),
            "max_concurrent_fetches": (
                self.max_concurrent_fetches,
                1,
                MAX_CONCURRENT_FETCHES,
            ),
            "max_depth": (self.max_depth, 0, MAX_DEPTH),
            "wall_clock_timeout_seconds": (
                self.wall_clock_timeout_seconds,
                0.001,
                MAX_TIMEOUT_SECONDS,
            ),
        }
        for name, (value, minimum, maximum) in limits.items():
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        if self.max_primary_tasks + self.max_followup_tasks > self.max_research_tasks:
            raise ValueError("task class ceilings exceed max_research_tasks")

    @classmethod
    def serious(cls) -> BudgetConfig:
        return cls(
            max_searches=100,
            max_pages=600,
            max_concurrent_fetches=12,
            wall_clock_timeout_seconds=1800,
        )


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    primary_tasks: int
    followup_tasks: int
    searches: int
    pages: int
    elapsed_seconds: float

    @property
    def research_tasks(self) -> int:
        return self.primary_tasks + self.followup_tasks


class BudgetManager:
    """Atomic, fail-closed global budget accounting."""

    def __init__(self, config: BudgetConfig) -> None:
        self.config = config
        self._started = time.monotonic()
        self._primary_tasks = 0
        self._followup_tasks = 0
        self._searches = 0
        self._pages = 0
        self._lock = threading.Lock()

    def check_time(self) -> None:
        if time.monotonic() - self._started >= self.config.wall_clock_timeout_seconds:
            raise BudgetExceeded("wall-clock timeout exhausted")

    def reserve_primary_task(self) -> None:
        self._reserve_task(followup=False)

    def reserve_followup_task(self, *, parent_depth: int) -> None:
        if parent_depth >= self.config.max_depth:
            raise BudgetExceeded("follow-up would exceed max depth")
        self._reserve_task(followup=True)

    def _reserve_task(self, *, followup: bool) -> None:
        with self._lock:
            self.check_time()
            total = self._primary_tasks + self._followup_tasks
            if total >= self.config.max_research_tasks:
                raise BudgetExceeded("research task budget exhausted")
            if followup:
                if self._followup_tasks >= self.config.max_followup_tasks:
                    raise BudgetExceeded("follow-up task budget exhausted")
                self._followup_tasks += 1
            else:
                if self._primary_tasks >= self.config.max_primary_tasks:
                    raise BudgetExceeded("primary task budget exhausted")
                self._primary_tasks += 1

    def reserve_search(self) -> None:
        self._reserve("_searches", self.config.max_searches, "search")

    def reserve_page(self) -> None:
        self._reserve("_pages", self.config.max_pages, "page")

    def _reserve(self, attribute: str, ceiling: int, label: str) -> None:
        with self._lock:
            self.check_time()
            current = getattr(self, attribute)
            if current >= ceiling:
                raise BudgetExceeded(f"{label} budget exhausted")
            setattr(self, attribute, current + 1)

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                primary_tasks=self._primary_tasks,
                followup_tasks=self._followup_tasks,
                searches=self._searches,
                pages=self._pages,
                elapsed_seconds=time.monotonic() - self._started,
            )

    def remaining_pages(self) -> int:
        return max(0, self.config.max_pages - self.snapshot().pages)

    def remaining_searches(self) -> int:
        return max(0, self.config.max_searches - self.snapshot().searches)

    def remaining_seconds(self) -> float:
        return max(
            0.0,
            self.config.wall_clock_timeout_seconds
            - (time.monotonic() - self._started),
        )
