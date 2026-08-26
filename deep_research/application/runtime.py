"""Shared process-runtime limits for CLI and HTTP workers."""

from pathlib import Path


def worker_script_path() -> Path:
    """Resolve the isolated worker entrypoint independently of caller location."""
    return Path(__file__).resolve().parents[1] / "_worker.py"


def worker_timeout(total_seconds: float) -> float:
    """Reserve a bounded cleanup window inside the public wall-clock ceiling."""
    cleanup_grace = min(5.0, max(0.05, total_seconds * 0.1))
    return max(0.001, total_seconds - cleanup_grace)
