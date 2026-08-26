# Parallax

Evidence-aware deep research that keeps support, contradiction, confidence, and unresolved gaps visible.

Live app: [parallax-five-sepia.vercel.app](https://parallax-five-sepia.vercel.app)

## What it does

Parallax turns a research question into four focused tasks, screens a broad set of web results in parallel, deeply extracts the strongest sources, and produces a concise cited report. Users review the plan before research begins and can start a new research path from contradicting evidence.

The web app provides streaming progress, concurrent sessions, source inspection, PDF extraction, invalid-query checks, and actionable provider errors. State is in memory; every run also writes an inspectable Evidence Ledger and JSONL audit trail.

## Run locally

Requires Python 3.11+, Node.js 24, and:

```bash
export AWS_BEARER_TOKEN_BEDROCK="..."
export TAVILY_API_KEY="..."
```

Start the API and frontend in separate terminals:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
deep-research-api
```

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`.

The default region is `us-east-1`; the default model is `us.anthropic.claude-sonnet-4-6`. Override CLI runs with `BEDROCK_MODEL_ID` and web runs with `BEDROCK_WEB_MODEL_ID`.

CLI usage:

```bash
python -m deep_research "What evidence supports and challenges remote work productivity?"
python -m deep_research "Your question" --profile serious
```

## Research bounds

Exactly three roles are used:

1. **Planner** creates four focused primary tasks.
2. **Researcher** searches in parallel, deduplicates URLs, ranks sources, and compresses pages into validated evidence.
3. **Critic/Synthesizer** checks coverage, may create at most two depth-one follow-ups, then writes the report.

The serious profile enforces ceilings of 100 searches, 600 page reservations, 12 concurrent fetches, six total tasks, depth one, and 30 minutes. These are safety limits, not targets. A typical broad run screens hundreds of search results, then deeply extracts at most 30 ranked sources per task. Raw pages are never passed wholesale into synthesis.

Confidence is computed in code:

- **High:** at least 3 supporting domains and at most 1 contradicting domain
- **Moderate:** 2 supporting domains and at most 1 contradicting domain
- **Low:** 1 supporting domain or at least 2 contradicting domains
- **Insufficient:** no supporting domains

## Run artifacts

Each run writes:

- `report.md` - cited final report
- `ledger.json` - claims, observations, polarity, and confidence
- `events.jsonl` - append-only audit events
- `run.json` - tasks, critic checks, budget usage, and status

Source IDs are deterministic. Unknown or claim-mismatched citations fail validation.

## Backend layout

Core rules live in `domain/`; the three roles in `agents/`; provider and extraction adapters in `infrastructure/`; run orchestration in `application/`; and HTTP/session delivery in `api/`.

## Deployment

- Frontend: Vercel, rooted at `frontend/`
- Backend: Render, configured by `render.yaml`
- Pushes to `main` deploy the frontend through Vercel's Git integration. Backend code or configuration changes trigger Render through `.github/workflows/deploy-backend.yml`.

Production secrets stay in the hosting platforms. `NEXT_PUBLIC_RESEARCH_API_URL` points the frontend to Render; the backend uses explicit allowed origins and hosts.

The current public deployment uses a free Render instance. It can cold-start after inactivity, and restarts clear in-memory sessions and process-local anonymous quotas. V1 has no authentication or persistent database.

## Tests

```bash
python -m unittest discover -s tests -v
cd frontend && npm run lint && npm run build
```

## V1 boundaries

No agent framework, agent swarm, database, embeddings, deep recursion, multi-turn memory, or semantic claim merging. Similar claims remain separate unless their normalized text matches exactly.

See [DECISION_NOTE.md](DECISION_NOTE.md) for the architecture and scope rationale.
