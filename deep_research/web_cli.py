from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .budget import BudgetConfig
from .web_sessions import ResearchSessionService
from .webapp import create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="deep-research-api",
        description="Local Parallax research API.",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-searches", type=int, default=24)
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--max-concurrent-fetches", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        config = BudgetConfig(
            max_searches=args.max_searches,
            max_pages=args.max_pages,
            max_concurrent_fetches=args.max_concurrent_fetches,
            wall_clock_timeout_seconds=args.timeout,
        )
    except ValueError as exc:
        parser.error(str(exc))
    service = ResearchSessionService(output_root=Path("runs/web"), config=config)
    uvicorn.run(
        create_app(service),
        host="127.0.0.1",
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
