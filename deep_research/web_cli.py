from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from .api.http import create_app
from .api.sessions import ResearchSessionService
from .domain.budget import BudgetConfig


def _scaled_reserve(configured: int, baseline: int, reserve: int) -> int:
    return min(max(0, configured - 1), round(configured * reserve / baseline))


def main(argv: list[str] | None = None) -> int:
    fast = BudgetConfig.fast()
    deep = BudgetConfig.deep()
    parser = argparse.ArgumentParser(
        prog="deep-research-api",
        description="Local Parallax research API.",
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--fast-max-searches", type=int, default=fast.max_searches)
    parser.add_argument("--fast-max-sources", type=int, default=fast.max_sources)
    parser.add_argument("--fast-max-pages", type=int, default=fast.max_pages)
    parser.add_argument("--deep-max-searches", type=int, default=deep.max_searches)
    parser.add_argument("--deep-max-sources", type=int, default=deep.max_sources)
    parser.add_argument("--deep-max-pages", type=int, default=deep.max_pages)
    parser.add_argument(
        "--max-concurrent-fetches",
        type=int,
        default=fast.max_concurrent_fetches,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=fast.wall_clock_timeout_seconds,
    )
    parser.add_argument(
        "--deep-timeout",
        type=float,
        default=deep.wall_clock_timeout_seconds,
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.host not in {"127.0.0.1", "0.0.0.0"}:
        parser.error("--host must be 127.0.0.1 or 0.0.0.0")
    try:
        configs = {
            "fast": BudgetConfig(
                max_followup_tasks=1,
                max_research_tasks=5,
                max_searches=args.fast_max_searches,
                max_sources=args.fast_max_sources,
                max_pages=args.fast_max_pages,
                followup_source_reserve=_scaled_reserve(
                    args.fast_max_sources,
                    fast.max_sources,
                    fast.followup_source_reserve,
                ),
                followup_page_reserve=_scaled_reserve(
                    args.fast_max_pages,
                    fast.max_pages,
                    fast.followup_page_reserve,
                ),
                max_concurrent_fetches=args.max_concurrent_fetches,
                wall_clock_timeout_seconds=args.timeout,
            ),
            "deep": BudgetConfig(
                max_searches=args.deep_max_searches,
                max_sources=args.deep_max_sources,
                max_pages=args.deep_max_pages,
                followup_source_reserve=_scaled_reserve(
                    args.deep_max_sources,
                    deep.max_sources,
                    deep.followup_source_reserve,
                ),
                followup_page_reserve=_scaled_reserve(
                    args.deep_max_pages,
                    deep.max_pages,
                    deep.followup_page_reserve,
                ),
                max_concurrent_fetches=args.max_concurrent_fetches,
                wall_clock_timeout_seconds=args.deep_timeout,
            ),
        }
    except ValueError as exc:
        parser.error(str(exc))
    service = ResearchSessionService(output_root=Path("runs/web"), configs=configs)
    uvicorn.run(
        create_app(service),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
