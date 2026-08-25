from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from .web_sessions import ResearchSessionService, SessionCapacityError, TERMINAL_STATUSES


LOCAL_ORIGINS = {"http://localhost:3000", "http://127.0.0.1:3000"}


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4_000)


class BranchRequest(BaseModel):
    observation_id: str = Field(min_length=2, max_length=80)


def create_app(service: ResearchSessionService | None = None) -> FastAPI:
    sessions = service or ResearchSessionService(
        output_root=Path("runs/web"),
    )
    app = FastAPI(title="Parallax Research API", version="0.1.0")
    app.state.sessions = sessions
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "testserver"],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(LOCAL_ORIGINS),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    @app.middleware("http")
    async def enforce_local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin is not None and origin not in LOCAL_ORIGINS:
            return PlainTextResponse("forbidden origin", status_code=403)
        return await call_next(request)

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "configured": sessions.configured(),
            "model": sessions.model_id,
        }

    @app.get("/api/sessions")
    def list_sessions() -> list[dict[str, object]]:
        return sessions.list_sessions()

    @app.post("/api/sessions", status_code=status.HTTP_202_ACCEPTED)
    def create_session(request: ResearchRequest) -> dict[str, object]:
        if not sessions.configured():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bedrock and Tavily environment variables are required",
            )
        try:
            session = sessions.create(request.query)
        except SessionCapacityError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return session.summary()

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, object]:
        try:
            return sessions.get(session_id).detail()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc

    @app.post(
        "/api/sessions/{session_id}/branches",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def branch_session(
        session_id: str,
        request: BranchRequest,
    ) -> dict[str, object]:
        try:
            session = sessions.create_branch(session_id, request.observation_id)
        except SessionCapacityError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return session.summary()

    @app.get("/api/sessions/{session_id}/events")
    async def stream_events(
        session_id: str,
        last_event_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        try:
            session = sessions.get(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        try:
            cursor = int(last_event_id) + 1 if last_event_id is not None else 0
        except ValueError:
            cursor = 0
        if not sessions.acquire_sse():
            raise HTTPException(status_code=429, detail="event stream capacity reached")

        async def events():
            nonlocal cursor
            try:
                while True:
                    pending = session.event_slice(cursor)
                    for event in pending:
                        cursor = int(event["id"]) + 1
                        payload = json.dumps(event["data"], ensure_ascii=False)
                        yield (
                            f"id: {event['id']}\n"
                            f"event: {event['event']}\n"
                            f"data: {payload}\n\n"
                        )
                    if session.status in TERMINAL_STATUSES and not session.event_slice(cursor):
                        break
                    yield ": keepalive\n\n"
                    await asyncio.sleep(0.35)
            finally:
                sessions.release_sse()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
