'use client';

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { API_BASE, EvidenceClaim, Observation, SessionDetail, SessionStatus, SessionSummary, api } from '../lib/api';

type Tab = 'report' | 'evidence' | 'activity';
type Activity = { id: string; message: string; stage?: string; tone?: 'warning' | 'done' };
const examples = [
  ['Remote work', 'What evidence supports and challenges remote work productivity?'],
  ['GLP-1 outcomes', 'How effective are GLP-1 medicines for long-term weight management, and where does evidence disagree?'],
  ['Python 3.13', 'What changed in free-threaded Python 3.13, and what limitations remain?'],
];
const terminal = new Set<SessionStatus>(['completed', 'completed_with_errors', 'failed']);

function statusLabel(status: SessionStatus) {
  return { queued: 'Queued', running: 'Researching', completed: 'Complete', completed_with_errors: 'Complete · partial', failed: 'Failed' }[status];
}

function SessionButton({ session, active, child, onSelect }: { session: SessionSummary; active: boolean; child?: boolean; onSelect: () => void }) {
  return <button className={`session-item${active ? ' active' : ''}${child ? ' child' : ''}`} onClick={onSelect} type="button">
    {child && <span className="branch-glyph" aria-hidden="true">↳</span>}<span className={`session-state status-${session.status}`} aria-hidden="true" />
    <span className="session-copy"><strong>{session.title}</strong><small>{terminal.has(session.status) ? `${session.claim_count} claims · ${session.contested_count} contested` : statusLabel(session.status)}</small></span>
  </button>;
}

function SourceLink({ observation }: { observation: Observation }) {
  return <a className="source-link" href={observation.source_url} rel="noreferrer" target="_blank"><span>{observation.source_id}</span>{observation.source_domain}<i aria-hidden="true">↗</i></a>;
}

function ClaimCard({ claim, branching, onBranch }: { claim: EvidenceClaim; branching: string | null; onBranch: (observation: Observation) => void }) {
  const contradictions = claim.observations.filter((item) => item.polarity === 'contradict');
  const support = claim.observations.filter((item) => item.polarity === 'support');
  return <article className={`evidence-card${claim.disagreement ? ' contested' : ''}`}>
    <div className="evidence-card-head"><span className={`confidence confidence-${claim.confidence.toLowerCase()}`}>{claim.confidence}</span>{claim.disagreement && <span className="disagreement-label">Contested</span>}<span className="domain-count">{claim.supporting_domain_count} supporting domains</span></div>
    <h3>{claim.text}</h3>
    {support.length > 0 && <div className="evidence-group"><p className="group-label support-label">Support · {support.length}</p>{support.map((observation) => <SourceLink key={observation.observation_id} observation={observation} />)}</div>}
    {contradictions.map((observation) => <div className="contradiction-path" key={observation.observation_id}>
      <div className="path-topline"><span>Contradicting path</span><SourceLink observation={observation} /></div><blockquote>{observation.excerpt}</blockquote>
      <button disabled={branching === observation.observation_id} onClick={() => onBranch(observation)} type="button">{branching === observation.observation_id ? 'Starting path…' : `Research this path from ${observation.source_id}`} <span aria-hidden="true">↗</span></button>
    </div>)}
  </article>;
}

export default function Home() {
  const [query, setQuery] = useState(''); const [sessions, setSessions] = useState<SessionSummary[]>([]); const [activeId, setActiveId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SessionDetail | null>(null); const [streamedReport, setStreamedReport] = useState(''); const [activity, setActivity] = useState<Activity[]>([]);
  const [tab, setTab] = useState<Tab>('report'); const [error, setError] = useState<string | null>(null); const [configured, setConfigured] = useState<boolean | null>(null);
  const [model, setModel] = useState('Bedrock · Opus'); const [submitting, setSubmitting] = useState(false); const [branching, setBranching] = useState<string | null>(null); const streamRef = useRef<EventSource | null>(null);
  const seenEventsRef = useRef(new Set<string>());
  const active = sessions.find((session) => session.id === activeId) ?? null;
  const roots = useMemo(() => sessions.filter((item) => !item.parent_session_id), [sessions]);
  const children = useCallback((id: string) => sessions.filter((item) => item.parent_session_id === id), [sessions]);
  const refreshSessions = useCallback(async () => { const items = await api.sessions(); setSessions(items); return items; }, []);
  const loadDetail = useCallback(async (id: string) => { const value = await api.session(id); setDetail(value); if (value.report) setStreamedReport(value.report); return value; }, []);
  const updateStatus = useCallback((id: string, status: SessionStatus) => setSessions((items) => items.map((item) => item.id === id ? { ...item, status } : item)), []);

  const connect = useCallback((session: SessionSummary) => {
    streamRef.current?.close();
    if (terminal.has(session.status)) { void loadDetail(session.id); return; }
    const stream = new EventSource(`${API_BASE}/api/sessions/${session.id}/events`); streamRef.current = stream;
    const consume = (event: MessageEvent) => { const key = `${session.id}:${event.lastEventId}`; if (seenEventsRef.current.has(key)) return null; seenEventsRef.current.add(key); return JSON.parse(event.data) as { message?: string; stage?: string; text?: string; status?: SessionStatus }; };
    const push = (data: { message?: string; stage?: string }, event: MessageEvent, tone?: Activity['tone']) => { if (data.message) setActivity((items) => [...items, { id: `${session.id}-${event.lastEventId}`, message: data.message!, stage: data.stage, tone }]); };
    stream.addEventListener('session.started', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; updateStatus(session.id, 'running'); push(data, event); }); stream.addEventListener('research.progress', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (data) push(data, event); });
    stream.addEventListener('report.chunk', (raw) => { const data = consume(raw as MessageEvent); if (data?.text) setStreamedReport((value) => value + data.text); });
    stream.addEventListener('session.completed', async (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; updateStatus(session.id, data.status ?? 'completed'); push(data, event, 'done'); stream.close(); await Promise.all([loadDetail(session.id), refreshSessions()]); });
    stream.addEventListener('session.failed', async (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; updateStatus(session.id, 'failed'); push(data, event, 'warning'); stream.close(); await Promise.all([loadDetail(session.id), refreshSessions()]); });
    stream.onerror = () => { if (stream.readyState !== EventSource.CLOSED) setError('Live connection interrupted. Reconnecting…'); };
  }, [loadDetail, refreshSessions, updateStatus]);

  useEffect(() => { Promise.all([api.health(), api.sessions()]).then(([health, items]) => { setConfigured(health.configured); setModel(health.model.replace('us.anthropic.claude-', 'Opus ').replace('-v1', '').replaceAll('-', '.')); setSessions(items); if (items[0]) { setActiveId(items[0].id); connect(items[0]); } }).catch((cause: Error) => setError(`Research service unavailable: ${cause.message}`)); return () => streamRef.current?.close(); }, [connect]);

  function selectSession(session: SessionSummary) { if (session.id === activeId) return; seenEventsRef.current.forEach((key) => { if (key.startsWith(`${session.id}:`)) seenEventsRef.current.delete(key); }); setDetail(null); setStreamedReport(''); setActivity([]); setError(null); setActiveId(session.id); connect(session); }

  async function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); if (!query.trim() || submitting) return; setSubmitting(true); setError(null); try { const session = await api.create(query.trim()); setSessions((items) => [session, ...items]); selectSession(session); setQuery(''); setTab('activity'); } catch (cause) { setError(cause instanceof Error ? cause.message : 'Research could not start'); } finally { setSubmitting(false); } }
  async function startBranch(observation: Observation) { if (!detail) return; setBranching(observation.observation_id); setError(null); try { const session = await api.branch(detail.id, observation.observation_id); setSessions((items) => [session, ...items]); selectSession(session); setTab('activity'); } catch (cause) { setError(cause instanceof Error ? cause.message : 'Evidence path could not start'); } finally { setBranching(null); } }
  function newResearch() { streamRef.current?.close(); setActiveId(null); setDetail(null); setStreamedReport(''); setActivity([]); setError(null); setTab('report'); }
  const running = active && !terminal.has(active.status); const contested = detail?.evidence.filter((claim) => claim.disagreement).length ?? 0;

  return <main className="app-shell"><aside className="sidebar">
    <div className="brand-row"><div className="brand-mark" aria-hidden="true"><span /><span /></div><div><p className="brand-name">PARALLAX</p><p className="brand-subtitle">Evidence research</p></div></div>
    <button className="new-research" onClick={newResearch} type="button"><span aria-hidden="true">＋</span>New research</button><div className="session-heading"><span>Research sessions</span><span>{sessions.length}</span></div>
    <nav className="session-list" aria-label="Research sessions">{roots.length === 0 && <p className="empty-sessions">No research yet. Start with a question.</p>}{roots.map((session) => <div className="session-family" key={session.id}><SessionButton active={activeId === session.id} onSelect={() => selectSession(session)} session={session} />{children(session.id).map((child) => <SessionButton active={activeId === child.id} child key={child.id} onSelect={() => selectSession(child)} session={child} />)}</div>)}</nav>
    <div className="sidebar-foot"><span className={`status-dot${configured === false ? ' offline' : ''}`} /><span>{model}</span><span className="local-pill">Local</span></div>
  </aside><section className="workspace"><header className="workspace-header"><div><p className="eyebrow">Evidence workspace</p><h1>{active?.title ?? 'Start a research thread'}</h1></div><div className="header-meta"><span className={`pulse${running ? ' running' : ''}`} aria-hidden="true" /><span>{active ? statusLabel(active.status) : configured === false ? 'Keys required' : 'Ready'}</span></div></header>
    {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError(null)} type="button">Dismiss</button></div>}
    {!active ? <div className="hero-grid"><section className="query-panel"><div className="hero-copy"><p className="hero-kicker">Deep research, without flattened certainty</p><h2>Ask the question.<br />Keep disagreement.</h2><p>Parallax researches multiple evidence paths in parallel, then shows where sources support, contradict, or leave the record unresolved.</p></div>
      <form className="query-form" onSubmit={submit}><label htmlFor="research-query">What do you want to understand?</label><textarea autoFocus id="research-query" onChange={(event) => setQuery(event.target.value)} placeholder="e.g. Does a four-day workweek improve productivity without increasing burnout?" rows={5} value={query} /><div className="composer-foot"><div className="composer-note"><span className="mini-ledger" aria-hidden="true" />Evidence ledger included</div><button disabled={!query.trim() || submitting || configured === false} type="submit">{submitting ? 'Starting…' : 'Begin research'} <span aria-hidden="true">→</span></button></div></form>
      <div className="prompts" aria-label="Example questions">{examples.map(([label, value]) => <button key={label} onClick={() => setQuery(value)} type="button">{label}</button>)}</div></section>
      <aside className="ledger-preview" aria-label="Evidence ledger preview"><div className="preview-topline"><span>HOW PARALLAX THINKS</span><span>01 / 03</span></div><h3>Evidence remains steerable</h3><p className="preview-intro">Every finding keeps source strength, contradiction, and unresolved gaps.</p><div className="claim-card support"><div className="claim-head"><span>SUPPORTED</span><strong>High</strong></div><p>Independent sources converge on primary outcome.</p><div className="source-row"><span>S1</span><span>S2</span><span>S3</span><small>3 domains</small></div></div><div className="claim-card contradict"><div className="claim-head"><span>CONTESTED</span><strong>Low</strong></div><p>Credible source reaches a different conclusion.</p><button type="button">Follow contradiction S4 <span aria-hidden="true">↗</span></button></div><div className="preview-footer"><span>Planner</span><i /><span>Researchers</span><i /><span>Critic</span></div></aside></div> :
      <div className="result-shell"><div className="research-title"><div><p className="hero-kicker">{active.branch ? `Evidence path · ${active.branch.source_id}` : 'Research question'}</p><h2>{active.branch?.claim_text ?? active.query}</h2>{active.branch && <a href={active.branch.source_url} rel="noreferrer" target="_blank">Branched from {active.branch.source_id} · inspect source ↗</a>}</div><div className="metric-row"><div><strong>{detail?.evidence.length ?? active.claim_count}</strong><span>Claims</span></div><div className={contested ? 'metric-contested' : ''}><strong>{contested || active.contested_count}</strong><span>Contested</span></div></div></div>
        <div className="tab-bar" role="tablist" aria-label="Research views">{(['report', 'evidence', 'activity'] as Tab[]).map((value) => <button aria-selected={tab === value} className={tab === value ? 'active' : ''} key={value} onClick={() => setTab(value)} role="tab" type="button">{value}{value === 'evidence' && contested > 0 ? <span>{contested}</span> : null}</button>)}</div><section className="result-view">
          {tab === 'report' && <div className="report-layout"><article className="report-paper">{streamedReport ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamedReport}</ReactMarkdown> : <RunningState activity={activity} status={active.status} />}</article><aside className="report-rail"><p className="rail-label">Evidence state</p><div><strong>{detail?.evidence.filter((item) => item.confidence === 'High').length ?? 0}</strong><span>High confidence</span></div><div><strong>{contested}</strong><span>Disagreements kept</span></div><div><strong>{new Set(detail?.evidence.flatMap((claim) => claim.observations.map((obs) => obs.source_domain))).size}</strong><span>Distinct domains</span></div>{contested > 0 && <button onClick={() => setTab('evidence')} type="button">Explore contested paths <span>→</span></button>}</aside></div>}
          {tab === 'evidence' && <div className="evidence-view"><div className="section-intro"><div><p className="hero-kicker">Transparent by design</p><h2>Evidence ledger</h2></div><p>Claims stay separate. Domain counts and contradictions remain visible; confidence comes from code-enforced rules.</p></div>{detail ? [...detail.evidence].sort((a, b) => Number(b.disagreement) - Number(a.disagreement)).map((claim) => <ClaimCard branching={branching} claim={claim} key={claim.claim_id} onBranch={startBranch} />) : <RunningState activity={activity} status={active.status} />}</div>}
          {tab === 'activity' && <ActivityView activity={activity} status={active.status} />}
        </section></div>}
  </section></main>;
}

function RunningState({ activity, status }: { activity: Activity[]; status: SessionStatus }) { if (status === 'failed') return <div className="empty-state"><span>!</span><h3>Research stopped</h3><p>Open Activity for failure details, then start a new research session.</p></div>; return <div className="empty-state researching"><span /><h3>Building evidence map</h3><p>{activity.at(-1)?.message ?? 'Preparing bounded research tasks…'}</p></div>; }
function ActivityView({ activity, status }: { activity: Activity[]; status: SessionStatus }) { return <div className="activity-view"><div className="section-intro"><div><p className="hero-kicker">Live audit trail</p><h2>Research activity</h2></div><p>Visible progress from planning through evidence compression and final critique.</p></div><div className="activity-list">{activity.length === 0 && <RunningState activity={activity} status={status} />}{activity.map((item, index) => <div className={`activity-item ${item.tone ?? ''}`} key={item.id}><span>{item.tone === 'done' ? '✓' : index + 1}</span><div><strong>{item.message}</strong>{item.stage && <small>{item.stage.replaceAll('.', ' · ')}</small>}</div></div>)}</div></div>; }
