'use client';

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { API_BASE, EvidenceClaim, Observation, SessionDetail, SessionStatus, SessionSummary, api } from '../lib/api';

type View = 'answer' | 'process';
type Activity = { id: string; message: string; stage?: string; taskId?: string; tone?: 'warning' | 'done' };
type DrawerSelection = { claim: EvidenceClaim; observation: Observation };
const examples = [
  'How effective are GLP-1 medicines for long-term weight management, and where does evidence disagree?',
  'Does a four-day workweek improve productivity without increasing burnout?',
  'What changed in free-threaded Python 3.13, and what limitations remain?',
];
const terminal = new Set<SessionStatus>(['completed', 'completed_with_errors', 'failed', 'rejected']);

function statusLabel(status: SessionStatus) {
  return { planning: 'Creating plan', ready: 'Plan ready', queued: 'Starting', running: 'Researching', synthesizing: 'Writing answer', completed: 'Complete', completed_with_errors: 'Complete · partial', failed: 'Failed', rejected: 'Needs revision' }[status];
}

function cleanReport(report: string) {
  return report
    .replace(/\n## Sources\s*[\s\S]*$/i, '')
    .trim()
    .replace(/\[(S\d+)\]/g, '$1')
    .replace(/\b(S\d+)\b/g, '[$1](#evidence-$1)');
}

function findCitation(evidence: EvidenceClaim[], sourceId: string): DrawerSelection | null {
  for (const claim of evidence) {
    const observation = claim.observations.find((item) => item.source_id === sourceId);
    if (observation) return { claim, observation };
  }
  return null;
}

function SessionItem({ session, active, child, onClick }: { session: SessionSummary; active: boolean; child?: boolean; onClick: () => void }) {
  return <button className={`history-item${active ? ' active' : ''}${child ? ' child' : ''}`} onClick={onClick} type="button">
    {child && <span className="branch-mark" aria-hidden="true">↳</span>}
    <span className="history-copy"><strong>{session.branch ? `Path from ${session.branch.source_id}` : session.title}</strong><small>{statusLabel(session.status)}</small></span>
    <span className={`history-status status-${session.status}`} aria-hidden="true" />
  </button>;
}

export default function Home() {
  const [query, setQuery] = useState('');
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [activity, setActivity] = useState<Activity[]>([]);
  const [streamedReport, setStreamedReport] = useState('');
  const [view, setView] = useState<View>('answer');
  const [drawer, setDrawer] = useState<DrawerSelection | null>(null);
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [branching, setBranching] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const streamRef = useRef<EventSource | null>(null);
  const activeIdRef = useRef<string | null>(null);
  const seenEventsRef = useRef(new Set<string>());
  const active = sessions.find((item) => item.id === activeId) ?? null;
  const roots = useMemo(() => sessions.filter((item) => !item.parent_session_id), [sessions]);
  const children = useCallback((id: string) => sessions.filter((item) => item.parent_session_id === id), [sessions]);
  const refreshSessions = useCallback(async () => { const items = await api.sessions(); setSessions(items); return items; }, []);
  const loadDetail = useCallback(async (id: string) => { const value = await api.session(id); if (activeIdRef.current === id) { setDetail(value); if (value.report) setStreamedReport(value.report); } return value; }, []);
  const updateSession = useCallback((id: string, patch: Partial<SessionSummary>) => setSessions((items) => items.map((item) => item.id === id ? { ...item, ...patch } : item)), []);
  const renderedReport = useMemo(() => cleanReport(streamedReport), [streamedReport]);

  const connect = useCallback((session: SessionSummary) => {
    streamRef.current?.close();
    const replayingTerminal = terminal.has(session.status);
    if (replayingTerminal) void loadDetail(session.id);
    const stream = new EventSource(`${API_BASE}/api/sessions/${session.id}/events`); streamRef.current = stream;
    const consume = (event: MessageEvent) => {
      const key = `${session.id}:${event.lastEventId}`;
      if (seenEventsRef.current.has(key)) return null;
      seenEventsRef.current.add(key);
      return JSON.parse(event.data) as { message?: string; stage?: string; text?: string; status?: SessionStatus; tasks?: SessionSummary['plan']; task_id?: string };
    };
    const push = (data: { message?: string; stage?: string; task_id?: string }, event: MessageEvent, tone?: Activity['tone']) => {
      if (data.message && activeIdRef.current === session.id) setActivity((items) => [...items, { id: `${session.id}-${event.lastEventId}`, message: data.message!, stage: data.stage, taskId: data.task_id, tone }]);
    };
    stream.addEventListener('session.created', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (data) push(data, event); });
    stream.addEventListener('plan.ready', async (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; push(data, event, 'done'); if (!replayingTerminal) { updateSession(session.id, { status: 'ready', plan: data.tasks ?? [] }); stream.close(); await loadDetail(session.id); } });
    stream.addEventListener('plan.rejected', async (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; push(data, event, 'warning'); if (!replayingTerminal) { updateSession(session.id, { status: 'rejected', error: data.message ?? 'Please rewrite the question.' }); await loadDetail(session.id); } stream.close(); });
    stream.addEventListener('session.started', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; if (!replayingTerminal) updateSession(session.id, { status: 'running' }); push(data, event); });
    stream.addEventListener('research.progress', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (data) push(data, event); });
    stream.addEventListener('report.started', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; if (!replayingTerminal) { updateSession(session.id, { status: 'synthesizing' }); if (activeIdRef.current === session.id) setStreamedReport(''); } push(data, event); });
    stream.addEventListener('report.chunk', (raw) => { const data = consume(raw as MessageEvent); if (!replayingTerminal && data?.text && activeIdRef.current === session.id) setStreamedReport((value) => value + data.text); });
    stream.addEventListener('session.completed', async (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; updateSession(session.id, { status: data.status ?? 'completed' }); push(data, event, 'done'); stream.close(); await Promise.all([loadDetail(session.id), refreshSessions()]); });
    stream.addEventListener('session.failed', async (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; updateSession(session.id, { status: 'failed' }); push(data, event, 'warning'); stream.close(); await Promise.all([loadDetail(session.id), refreshSessions()]); });
    stream.onerror = () => { if (!replayingTerminal && stream.readyState !== EventSource.CLOSED) setError('Live connection interrupted. Reconnecting…'); };
  }, [loadDetail, refreshSessions, updateSession]);

  useEffect(() => {
    Promise.all([api.health(), api.sessions()]).then(([health, items]) => { setConfigured(health.configured); setSessions(items); if (items[0]) { activeIdRef.current = items[0].id; setActiveId(items[0].id); connect(items[0]); } }).catch((cause: Error) => setError(`Research service unavailable: ${cause.message}`));
    return () => streamRef.current?.close();
  }, [connect]);

  function selectSession(session: SessionSummary) {
    streamRef.current?.close(); seenEventsRef.current.clear(); activeIdRef.current = session.id; setActiveId(session.id); setDetail(null); setActivity([]); setStreamedReport(''); setDrawer(null); setError(null); setSidebarOpen(false); setView('answer'); connect(session);
  }
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); if (!query.trim() || busy) return; setBusy(true); setError(null);
    try { const session = await api.create(query.trim()); setSessions((items) => [session, ...items]); setQuery(''); selectSession(session); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Plan could not be created'); }
    finally { setBusy(false); }
  }
  async function startResearch() {
    if (!active || busy) return; setBusy(true); setError(null); setActivity([]); setStreamedReport('');
    try { const session = await api.start(active.id); updateSession(active.id, { status: session.status }); connect(session); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Research could not start'); }
    finally { setBusy(false); }
  }
  async function startBranch(observation: Observation) {
    if (!detail || branching) return; setBranching(observation.observation_id); setError(null);
    try { const session = await api.branch(detail.id, observation.observation_id); setSessions((items) => [session, ...items]); setDrawer(null); selectSession(session); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Evidence path could not start'); }
    finally { setBranching(null); }
  }
  function newResearch() { streamRef.current?.close(); seenEventsRef.current.clear(); activeIdRef.current = null; setActiveId(null); setDetail(null); setActivity([]); setStreamedReport(''); setDrawer(null); setError(null); setSidebarOpen(false); setView('answer'); }
  const openCitation = useCallback((sourceId: string) => { const selection = findCitation(detail?.evidence ?? [], sourceId); if (selection) setDrawer(selection); }, [detail]);

  return <main className="shell">
    <aside className={`sidebar${sidebarOpen ? ' open' : ''}`}><div className="brand"><span className="brand-orbit" aria-hidden="true" /><strong>Parallax</strong></div><button className="new-thread" onClick={newResearch} type="button"><span>＋</span> New research</button><p className="sidebar-label">Your research</p><nav className="history" aria-label="Research history">{roots.length === 0 && <p className="history-empty">Research sessions will appear here.</p>}{roots.map((session) => <div key={session.id}><SessionItem active={activeId === session.id} onClick={() => selectSession(session)} session={session} />{children(session.id).map((child) => <SessionItem active={activeId === child.id} child key={child.id} onClick={() => selectSession(child)} session={child} />)}</div>)}</nav><div className="local-status"><span className={configured === false ? 'offline' : ''} />{configured === false ? 'API keys required' : 'Local workspace'}</div></aside>
    <button className="mobile-menu" onClick={() => setSidebarOpen((value) => !value)} type="button" aria-label="Toggle research history">☰</button>
    <section className="main-column">{error && <div className="error-banner" role="alert">{error}<button onClick={() => setError(null)} type="button">Dismiss</button></div>}{!active ? <Welcome query={query} setQuery={setQuery} submit={submit} busy={busy} configured={configured} /> : <ResearchConversation active={active} activity={activity} busy={busy} detail={detail} newResearch={newResearch} openCitation={openCitation} report={renderedReport} setView={setView} startResearch={startResearch} view={view} />}</section>
    {drawer && <EvidenceDrawer branching={branching} close={() => setDrawer(null)} selection={drawer} startBranch={startBranch} />}
  </main>;
}

function Welcome({ query, setQuery, submit, busy, configured }: { query: string; setQuery: (value: string) => void; submit: (event: FormEvent<HTMLFormElement>) => void; busy: boolean; configured: boolean | null }) {
  return <div className="welcome"><div className="welcome-mark"><span /><span /></div><h1>What should we research?</h1><p>Ask a complex question. Review the plan before any sources are searched.</p><form className="composer" onSubmit={submit}><textarea autoFocus onChange={(event) => setQuery(event.target.value)} placeholder="Ask a research question…" rows={4} value={query} /><div><span>Evidence-aware deep research</span><button disabled={!query.trim() || busy || configured === false} type="submit">{busy ? 'Planning…' : 'Create plan'} <i aria-hidden="true">↑</i></button></div></form><div className="examples">{examples.map((example) => <button key={example} onClick={() => setQuery(example)} type="button">{example}</button>)}</div></div>;
}

function ResearchConversation({ active, activity, busy, detail, newResearch, openCitation, report, setView, startResearch, view }: { active: SessionSummary; activity: Activity[]; busy: boolean; detail: SessionDetail | null; newResearch: () => void; openCitation: (id: string) => void; report: string; setView: (view: View) => void; startResearch: () => void; view: View }) {
  return <div className="conversation"><header className="conversation-header"><div><p>{active.branch ? `Branched from ${active.branch.source_id}` : 'Deep research'}</p><strong>{active.title}</strong></div><span className={`run-status status-${active.status}`}><i />{statusLabel(active.status)}</span></header><div className="thread"><div className="user-message"><p>{active.branch?.claim_text ?? active.query}</p></div>{active.status === 'planning' && <PlanningState />}{active.status === 'ready' && <PlanReview plan={active.plan} busy={busy} start={startResearch} />}{(active.status === 'queued' || active.status === 'running') && <ResearchingState activity={activity} plan={active.plan} />}{active.status === 'failed' && <div className="assistant-message failure"><span className="assistant-mark">P</span><div><h2>Research stopped</h2><p>{active.error ?? activity.at(-1)?.message ?? 'Research could not complete.'}</p></div></div>}{active.status === 'rejected' && <div className="assistant-message revision"><span className="assistant-mark">P</span><div><h2>This needs a clearer research question</h2><p>{active.error ?? 'Add a specific topic, comparison, outcome, or time range.'}</p><button onClick={newResearch} type="button">Rewrite question</button></div></div>}{(active.status === 'synthesizing' || (terminal.has(active.status) && !['failed', 'rejected'].includes(active.status))) && <div className="response-area"><div className="answer-tabs"><button className={view === 'answer' ? 'active' : ''} onClick={() => setView('answer')} type="button">Answer</button><button className={view === 'process' ? 'active' : ''} onClick={() => setView('process')} type="button">Research process</button></div>{view === 'answer' ? <Report report={report} evidence={detail?.evidence ?? []} openCitation={openCitation} streaming={active.status === 'synthesizing'} /> : <ProcessView activity={activity} detail={detail} plan={active.plan} />}</div>}</div></div>;
}

function PlanningState() { return <div className="assistant-message"><span className="assistant-mark thinking">P</span><div className="planning-copy"><h2>Building an actionable plan</h2><p>Breaking your question into focused, non-overlapping evidence paths.</p><div className="thinking-lines"><span /><span /><span /></div></div></div>; }

function PlanReview({ plan, busy, start }: { plan: SessionSummary['plan']; busy: boolean; start: () => void }) {
  return <div className="assistant-message plan-message"><span className="assistant-mark">P</span><div><p className="assistant-intro">I’ll investigate four evidence paths in parallel. Review the scope, then start research.</p><section className="plan-card"><div className="plan-card-head"><div><span>Research plan</span><strong>{plan.length} steps</strong></div><span className="budget-note">Bounded run</span></div><ol>{plan.map((task, index) => <li key={task.id} style={{ '--delay': `${index * 70}ms` } as React.CSSProperties}><span>{index + 1}</span><div><strong>{task.question}</strong><p>{task.rationale}</p></div><small>{task.priority}</small></li>)}</ol><div className="plan-actions"><p>No web research begins before approval.</p><button disabled={busy} onClick={start} type="button">{busy ? 'Starting…' : 'Start research'} <span aria-hidden="true">→</span></button></div></section></div></div>;
}

function ResearchingState({ activity, plan }: { activity: Activity[]; plan: SessionSummary['plan'] }) {
  const activeTasks = new Set(activity.map((item) => item.taskId).filter(Boolean));
  return <div className="assistant-message"><span className="assistant-mark thinking">P</span><div className="research-live"><h2>Researching across sources</h2><p className="live-message"><span />{activity.at(-1)?.message ?? 'Starting parallel evidence paths…'}</p><div className="live-plan">{plan.map((task, index) => <div className={activeTasks.has(task.id) ? 'visited' : ''} key={task.id}><span>{activeTasks.has(task.id) ? '✓' : index + 1}</span><p>{task.question}</p></div>)}</div><p className="leave-note">You can leave this session. Research continues locally.</p></div></div>;
}

function Report({ report, evidence, openCitation, streaming }: { report: string; evidence: EvidenceClaim[]; openCitation: (id: string) => void; streaming: boolean }) {
  return <div className="assistant-message report-message"><span className="assistant-mark">P</span><article className="report"><ReactMarkdown remarkPlugins={[remarkGfm]} components={{ img: () => null, a: ({ href, children }) => { if (href?.startsWith('#evidence-')) { const id = href.slice(10); const exists = evidence.some((claim) => claim.observations.some((item) => item.source_id === id)); return exists ? <button className="citation" onClick={() => openCitation(id)} onFocus={() => openCitation(id)} onMouseEnter={() => openCitation(id)} type="button">{children}</button> : <span>{children}</span>; } return <span>{children}</span>; } }}>{report}</ReactMarkdown>{streaming && <span className="stream-caret" aria-label="Answer is streaming" />}</article></div>;
}

function ProcessView({ activity, detail, plan }: { activity: Activity[]; detail: SessionDetail | null; plan: SessionSummary['plan'] }) {
  const count = (stage: string) => activity.filter((item) => item.stage === stage).length;
  const stages = [
    ['Plan approved', `${plan.length} focused evidence paths`],
    ['Sources searched', `${count('search.executed')} focused searches`],
    ['Pages compressed', `${count('page.explored')} pages processed · ${count('page.fetch_failed')} unavailable`],
    ['Evidence ledger built', `${count('observation.extracted')} atomic observations`],
    ['Critic and answer complete', `${count('critic.followup_created')} bounded follow-up paths`],
  ];
  return <div className="process-view"><div className="process-summary"><strong>{plan.length}</strong><span>planned paths</span><strong>{detail?.evidence.length ?? 0}</strong><span>ledger claims</span><strong>{detail?.evidence.filter((claim) => claim.disagreement).length ?? 0}</strong><span>contested</span></div><ol>{stages.map(([title, summary]) => <li key={title}><span className="done">✓</span><div><strong>{title}</strong><small>{summary}</small></div></li>)}</ol></div>;
}

function EvidenceDrawer({ branching, close, selection, startBranch }: { branching: string | null; close: () => void; selection: DrawerSelection; startBranch: (observation: Observation) => void }) {
  const { claim, observation } = selection;
  const support = claim.observations.filter((item) => item.polarity === 'support'); const contradictions = claim.observations.filter((item) => item.polarity === 'contradict');
  return <><div className="drawer-backdrop" aria-hidden="true" /><aside className="evidence-drawer" aria-label={`Evidence for ${observation.source_id}`}><header><div><span>{observation.source_id}</span><p>Evidence details</p></div><button onClick={close} type="button" aria-label="Close">×</button></header><div className="drawer-scroll"><a className="selected-source" href={observation.source_url} rel="noreferrer" target="_blank"><span>{observation.source_domain}</span><strong>{observation.statement}</strong><p>“{observation.excerpt}”</p><small>Open original source ↗</small></a><section className="confidence-panel"><div><span>Confidence</span><strong className={`confidence-${claim.confidence.toLowerCase()}`}>{claim.confidence}</strong></div><p>Rule-based tier · {claim.supporting_domain_count} supporting {claim.supporting_domain_count === 1 ? 'domain' : 'domains'}{claim.disagreement ? ` · ${claim.contradicting_domain_count} contradicting` : ''}</p></section><section className="drawer-claim"><span>Claim</span><h2>{claim.text}</h2></section><SourceGroup label="Supporting sources" observations={support} /><SourceGroup branching={branching} label="Contradicting sources" observations={contradictions} startBranch={startBranch} />{contradictions.length === 0 && <p className="no-contradiction">No contradicting evidence attached to this claim.</p>}</div></aside></>;
}

function SourceGroup({ branching, label, observations, startBranch }: { branching?: string | null; label: string; observations: Observation[]; startBranch?: (observation: Observation) => void }) {
  if (!observations.length) return null;
  return <section className="source-group"><h3>{label}<span>{observations.length}</span></h3>{observations.map((item) => <article key={item.observation_id}><div><span>{item.source_id}</span><a href={item.source_url} rel="noreferrer" target="_blank">{item.source_domain} ↗</a></div><p>{item.excerpt}</p>{item.polarity === 'contradict' && startBranch && <button disabled={branching === item.observation_id} onClick={() => startBranch(item)} type="button">{branching === item.observation_id ? 'Creating branch…' : 'Research this perspective'} <span>→</span></button>}</article>)}</section>;
}
