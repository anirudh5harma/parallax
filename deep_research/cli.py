from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from .application.runtime import worker_script_path, worker_timeout
from .application.service import create_plan, load_plan, run_query
from .domain.budget import BudgetConfig
from .infrastructure.providers import (
    BedrockConverseModel,
    HttpPageFetcher,
    TavilyExtractClient,
    TavilySearchClient,
)


class ConfigurationError(ValueError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deep-research",
        description="Bounded web research with a transparent evidence ledger.",
    )
    parser.add_argument("query", help="Research question")
    parser.add_argument(
        "--profile",
        choices=("dev", "fast", "deep", "serious"),
        default="dev",
        help="Budget preset (default: dev)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--max-searches", type=int, default=None)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-followup-tasks", type=int, default=None)
    parser.add_argument("--max-concurrent-fetches", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None, help="Wall-clock seconds")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan-output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--approved-plan", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def config_from_args(args: argparse.Namespace) -> BudgetConfig:
    profiles = {
        "dev": BudgetConfig,
        "fast": BudgetConfig.fast,
        "deep": BudgetConfig.deep,
        "serious": BudgetConfig.serious,
    }
    config = profiles[args.profile]()
    replacements = {
        "max_searches": args.max_searches,
        "max_sources": args.max_sources,
        "max_pages": args.max_pages,
        "max_followup_tasks": args.max_followup_tasks,
        "max_concurrent_fetches": args.max_concurrent_fetches,
        "wall_clock_timeout_seconds": args.timeout,
    }
    return replace(
        config,
        **{key: value for key, value in replacements.items() if value is not None},
    )


def _worker_timeout(total_seconds: float) -> float:
    """Backward-compatible alias for callers of the original CLI helper."""
    return worker_timeout(total_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        model_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        search_key = os.environ.get("TAVILY_API_KEY")
        if not model_key or (not args.plan_only and not search_key):
            missing = [
                name
                for name, value in (
                    ("AWS_BEARER_TOKEN_BEDROCK", model_key),
                    ("TAVILY_API_KEY", search_key if not args.plan_only else "unused"),
                )
                if not value
            ]
            raise ConfigurationError(
                "missing required environment variables: " + ", ".join(missing)
            )
        command_args = list(argv) if argv is not None else sys.argv[1:]
        worker_path = worker_script_path()
        command = [sys.executable, "-I", str(worker_path), *command_args]
        worker_environment = os.environ.copy()
        worker_environment["DEEP_RESEARCH_INTERNAL_TIMEOUT"] = str(
            _worker_timeout(config.wall_clock_timeout_seconds)
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                env=worker_environment,
                timeout=config.wall_clock_timeout_seconds,
            )
            return completed.returncode
        except subprocess.TimeoutExpired:
            print(
                "error: wall-clock timeout exhausted; worker terminated",
                file=sys.stderr,
            )
            return 2
    except Exception as exc:
        if args.debug:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


def worker_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        internal_timeout = os.environ.get("DEEP_RESEARCH_INTERNAL_TIMEOUT")
        if internal_timeout is not None:
            config = replace(
                config,
                wall_clock_timeout_seconds=float(internal_timeout),
            )
        model_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        search_key = os.environ.get("TAVILY_API_KEY")
        if not model_key or (not args.plan_only and not search_key):
            raise ConfigurationError("worker credentials are unavailable")
        model_name = args.model or os.environ.get(
            "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6"
        )
        region = os.environ.get("AWS_REGION") or os.environ.get(
            "AWS_DEFAULT_REGION", "us-east-1"
        )
        model = BedrockConverseModel(
            model_key,
            model_id=model_name,
            region=region,
        )
        if args.plan_only:
            if args.plan_output is None:
                raise ConfigurationError("plan output path is required")
            create_plan(
                query=args.query,
                output_path=args.plan_output,
                config=config,
                model=model,
            )
            return 0
        artifacts = run_query(
            query=args.query,
            output_root=args.output_dir,
            config=config,
            model=model,
            search=TavilySearchClient(search_key or ""),
            fetcher=HttpPageFetcher(),
            batch_extractor=TavilyExtractClient(search_key or ""),
            approved_tasks=(
                load_plan(args.approved_plan, args.query)
                if args.approved_plan is not None
                else None
            ),
        )
        sys.stdout.write(artifacts.report)
        print(f"Artifacts: {artifacts.run_dir}", file=sys.stderr)
        return 2 if artifacts.status == "failed" else 0
    except Exception as exc:
        if args.debug:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
