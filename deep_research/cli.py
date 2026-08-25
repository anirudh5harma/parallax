from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from .app import run_query
from .budget import BudgetConfig
from .providers import BedrockConverseModel, HttpPageFetcher, TavilySearchClient


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
        choices=("dev", "serious"),
        default="dev",
        help="Budget preset (default: dev)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--max-searches", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-concurrent-fetches", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None, help="Wall-clock seconds")
    parser.add_argument("--debug", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> BudgetConfig:
    config = BudgetConfig.serious() if args.profile == "serious" else BudgetConfig()
    replacements = {
        "max_searches": args.max_searches,
        "max_pages": args.max_pages,
        "max_concurrent_fetches": args.max_concurrent_fetches,
        "wall_clock_timeout_seconds": args.timeout,
    }
    return replace(
        config,
        **{key: value for key, value in replacements.items() if value is not None},
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        model_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        search_key = os.environ.get("TAVILY_API_KEY")
        if not model_key or not search_key:
            missing = [
                name
                for name, value in (
                    ("AWS_BEARER_TOKEN_BEDROCK", model_key),
                    ("TAVILY_API_KEY", search_key),
                )
                if not value
            ]
            raise ConfigurationError(
                "missing required environment variables: " + ", ".join(missing)
            )
        command_args = list(argv) if argv is not None else sys.argv[1:]
        worker_path = Path(__file__).with_name("_worker.py").resolve()
        command = [sys.executable, "-I", str(worker_path), *command_args]
        try:
            completed = subprocess.run(
                command,
                check=False,
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
        model_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        search_key = os.environ.get("TAVILY_API_KEY")
        if not model_key or not search_key:
            raise ConfigurationError("worker credentials are unavailable")
        model_name = args.model or os.environ.get(
            "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6"
        )
        region = os.environ.get("AWS_REGION") or os.environ.get(
            "AWS_DEFAULT_REGION", "us-east-1"
        )
        artifacts = run_query(
            query=args.query,
            output_root=args.output_dir,
            config=config,
            model=BedrockConverseModel(
                model_key,
                model_id=model_name,
                region=region,
            ),
            search=TavilySearchClient(search_key),
            fetcher=HttpPageFetcher(),
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
