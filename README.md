# Transparent Deep Research

Small CLI research system that preserves evidence strength and disagreement instead of producing only a polished answer.

## Run

Requires Python 3.11+ and two environment variables:

```bash
export AWS_BEARER_TOKEN_BEDROCK="..."
export TAVILY_API_KEY="..."
python -m deep_research "What evidence supports and challenges remote work productivity?"
```

Optional: set `AWS_REGION` and `BEDROCK_MODEL_ID`. Defaults are `us-east-1` and `us.anthropic.claude-sonnet-4-6`.

Local web workspace:

```bash
python -m pip install -e '.[dev]'
deep-research-api
cd frontend && npm install && npm run dev
```

Open `http://localhost:3000`. Web research defaults to Opus 4.6 through Bedrock; override with `BEDROCK_WEB_MODEL_ID`. Sessions stay in memory and support new research paths from contradicting citations.

Default `dev` budget: 4 primary tasks, 2 follow-ups, 24 searches, 40 pages, 8 concurrent fetches, 5-minute timeout. Larger run:

```bash
python -m deep_research "Your question" --profile serious
```

`serious` allows 80 searches, 200 pages, 10 concurrent fetches, and 15 minutes. CLI overrides remain bounded by absolute guards.

## Architecture

Exactly three roles:

1. **Planner** creates four focused, non-overlapping primary tasks.
2. **Researcher** runs tasks in parallel, deduplicates URLs, fetches pages under one shared concurrency gate, and compresses each page into validated evidence observations.
3. **Critic/Synthesizer** checks coverage, may create at most two depth-1 follow-ups total, performs one final check, then writes the report.

Researchers return immutable results. One controlled writer updates the in-memory ledger. Synthesis receives structured claims and short excerpts, never raw pages.

Confidence is computed in code:

- **High:** at least 3 supporting domains and at most 1 contradicting domain
- **Moderate:** 2 supporting domains and at most 1 contradicting domain
- **Low:** 1 supporting domain or at least 2 contradicting domains
- **Insufficient:** no supporting domains

## Outputs

Each run creates one directory under `runs/`:

- `report.md` - cited report with confidence, contradictions, weak evidence, and gaps
- `ledger.json` - complete claims and observations
- `events.jsonl` - append-only audit trail
- `run.json` - tasks, critic checks, budget usage, and status

Source IDs are assigned deterministically. Unknown or claim-mismatched citations fail validation.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover hard ceilings, depth rejection, concurrency, deduplication, literal excerpt validation, confidence rules, follow-up limits, citation validation, and full orchestration.

## V1 boundaries

No agent framework, database, embeddings, auth, multi-turn memory, deep recursion, or semantic claim merging. Similar claims stay separate unless their normalized text matches exactly. Web UI is local-only in V1.
