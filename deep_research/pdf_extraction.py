from __future__ import annotations

import logging
import math
import multiprocessing
import sys
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from multiprocessing.connection import Connection

from pypdf import PdfReader
from pypdf.errors import PdfReadError


MIN_PDF_TEXT_CHARACTERS = 80
MIN_ALPHANUMERIC_RATIO = 0.25
MAX_REPLACEMENT_CHARACTER_RATIO = 0.01
PDF_WORKER_MEMORY_BYTES = 512 * 1024 * 1024


class PdfExtractionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PdfExtraction:
    text: str
    title: str
    page_count: int


def extract_pdf(
    raw: bytes,
    *,
    timeout_seconds: float,
    max_pages: int = 150,
    max_chars: int = 120_000,
) -> PdfExtraction:
    if timeout_seconds <= 0:
        raise PdfExtractionError("PDF extraction timed out")
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_pdf_worker,
        args=(send, raw, max_pages, max_chars, timeout_seconds),
        daemon=True,
    )
    try:
        process.start()
    except (OSError, RuntimeError) as exc:
        receive.close()
        send.close()
        raise PdfExtractionError("PDF extraction could not start safely") from exc
    send.close()
    try:
        try:
            if not receive.poll(timeout_seconds):
                raise PdfExtractionError("PDF extraction timed out")
            status, payload = receive.recv()
        except (EOFError, OSError) as exc:
            raise PdfExtractionError("PDF extraction failed safely") from exc
        if status != "ok":
            raise PdfExtractionError(str(payload))
        text, title, page_count = payload
        return PdfExtraction(text=text, title=title, page_count=page_count)
    finally:
        receive.close()
        process.join(timeout=0.2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
        if process.is_alive():  # pragma: no cover - defensive OS fallback
            process.kill()
            process.join(timeout=1)


def _pdf_worker(
    send: Connection,
    raw: bytes,
    max_pages: int,
    max_chars: int,
    timeout_seconds: float,
) -> None:
    try:
        _apply_worker_limits(timeout_seconds)
        result = _extract_pdf_inline(raw, max_pages=max_pages, max_chars=max_chars)
        send.send(("ok", (result.text, result.title, result.page_count)))
    except PdfExtractionError as exc:
        send.send(("error", str(exc)))
    except BaseException:
        send.send(("error", "PDF extraction failed safely"))
    finally:
        send.close()


def _apply_worker_limits(timeout_seconds: float) -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        import resource

        cpu_seconds = max(1, math.ceil(timeout_seconds))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(
            resource.RLIMIT_AS,
            (PDF_WORKER_MEMORY_BYTES, PDF_WORKER_MEMORY_BYTES),
        )
    except (ImportError, OSError, ValueError):
        pass


def _extract_pdf_inline(
    raw: bytes,
    *,
    max_pages: int,
    max_chars: int,
) -> PdfExtraction:
    if max_pages <= 0 or max_chars <= 0:
        raise ValueError("PDF extraction limits must be positive")
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        reader = PdfReader(BytesIO(raw), strict=False)
    except (PdfReadError, OSError, ValueError) as exc:
        raise PdfExtractionError("PDF is malformed") from exc
    if reader.is_encrypted:
        raise PdfExtractionError("PDF is encrypted")
    page_count = len(reader.pages)
    if page_count > max_pages:
        raise PdfExtractionError("PDF exceeds page limit")

    page_texts: list[str] = []
    remaining = max_chars
    try:
        for page in reader.pages:
            if remaining <= 0:
                break
            separator_size = 2 if page_texts else 0
            available = remaining - separator_size
            if available <= 0:
                break
            text = _normalize_text(page.extract_text() or "", max_chars=available)
            if not text:
                continue
            page_texts.append(text)
            remaining -= len(text) + separator_size
    except (PdfReadError, OSError, TypeError, ValueError) as exc:
        raise PdfExtractionError("PDF text extraction failed") from exc

    text = "\n\n".join(page_texts).strip()
    _validate_text_quality(text)
    title = ""
    try:
        if reader.metadata and reader.metadata.title:
            title = _normalize_text(str(reader.metadata.title))[:300]
    except (AttributeError, PdfReadError, TypeError, ValueError):
        pass
    return PdfExtraction(text=text, title=title, page_count=page_count)


def _normalize_text(value: str, *, max_chars: int | None = None) -> str:
    output: list[str] = []
    pending_space = False
    for character in value:
        if (
            unicodedata.category(character) in {"Cc", "Cs"}
            and character not in "\n\t"
        ):
            continue
        if character.isspace():
            pending_space = bool(output)
            continue
        if pending_space:
            output.append(" ")
            pending_space = False
            if max_chars is not None and len(output) >= max_chars:
                break
        output.append(character)
        if max_chars is not None and len(output) >= max_chars:
            break
    return "".join(output)


def _validate_text_quality(text: str) -> None:
    compact_count = 0
    alphanumeric_count = 0
    replacement_count = 0
    for character in text:
        if character.isspace():
            continue
        compact_count += 1
        alphanumeric_count += character.isalnum()
        replacement_count += character == "\ufffd"
    if compact_count < MIN_PDF_TEXT_CHARACTERS:
        raise PdfExtractionError("PDF has no extractable text")
    if alphanumeric_count / compact_count < MIN_ALPHANUMERIC_RATIO:
        raise PdfExtractionError("PDF extracted text is low-signal")
    if replacement_count / compact_count > MAX_REPLACEMENT_CHARACTER_RATIO:
        raise PdfExtractionError("PDF extracted text is low-signal")
