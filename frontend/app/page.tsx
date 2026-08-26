'use client';

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { EvidenceClaim, Observation, SessionDetail, SessionStatus, SessionSummary, api, eventStreamUrl } from '../lib/api';

type View = 'answer' | 'process';
type Activity = { id: string; message: string; stage?: string; sourceDomain?: string; taskId?: string; tone?: 'warning' | 'done' };
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

function sourceIcon(domain: string) {
  const label = domain.split('.').at(-2) ?? domain;
  let hue = 0;
  for (const character of domain) hue = (hue * 31 + character.charCodeAt(0)) % 360;
  return { initial: label[0]?.toUpperCase() ?? '·', hue };
}

function cleanReport(report: string, evidence: EvidenceClaim[]) {
  const claimsByText = new Map(evidence.map((claim) => [claim.text, claim.claim_id]));
  const output: string[] = [];
  let claimId: string | null = null;
  let claimSources: string[] = [];
  let pillsPlaced = false;
  for (const line of report.replace(/\n## Sources\s*[\s\S]*$/i, '').trim().split('\n')) {
    if (line.startsWith('## ')) {
      claimId = null;
      claimSources = [];
      pillsPlaced = false;
    }
    if (line.startsWith('### ')) {
      claimId = claimsByText.get(line.slice(4).trim()) ?? null;
      claimSources = [];
      pillsPlaced = false;
    }
    if (/^- (Confidence|Support|Contradiction):/i.test(line)) continue;
    if (/^- Sources:/i.test(line)) {
      claimSources = [...new Set(line.match(/S\d+/g) ?? [])];
      continue;
    }
    if (claimSources.length && line.trim() && !line.startsWith('#')) {
      const citedInline = new Set(line.match(/S\d+/g) ?? []);
      const linkedLine = claimId
        ? line.replace(/\b(S\d+)\b/g, (sourceId) => claimSources.includes(sourceId) ? `[${sourceId}](#evidence-${claimId}-${sourceId})` : sourceId)
        : line;
      const pills = (pillsPlaced ? [] : claimSources.filter((sourceId) => !citedInline.has(sourceId))).map((sourceId) => claimId
        ? `[${sourceId}](#evidence-${claimId}-${sourceId})`
        : sourceId).join(' ');
      pillsPlaced = true;
      output.push(pills ? `${linkedLine} ${pills}` : linkedLine);
      continue;
    }
    output.push(line);
  }
  return output.join('\n').trim();
}

function findCitation(evidence: EvidenceClaim[], claimId: string, sourceId: string): DrawerSelection | null {
  const claim = evidence.find((item) => item.claim_id === claimId);
  const observation = claim?.observations.find((item) => item.source_id === sourceId);
  return claim && observation ? { claim, observation } : null;
}

function sessionTree(sessions: SessionSummary[]) {
  const children = new Map<string | null, SessionSummary[]>();
  for (const session of sessions) {
    const siblings = children.get(session.parent_session_id) ?? [];
    siblings.push(session);
    children.set(session.parent_session_id, siblings);
  }
  const result: { session: SessionSummary; depth: number }[] = [];
  const visit = (parentId: string | null, depth: number) => {
    for (const session of children.get(parentId) ?? []) {
      result.push({ session, depth });
      visit(session.id, depth + 1);
    }
  };
  visit(null, 0);
  return result;
}

function SessionItem({ session, active, depth, onClick }: { session: SessionSummary; active: boolean; depth: number; onClick: () => void }) {
  return <button className={`history-item${active ? ' active' : ''}${depth ? ' child' : ''}`} onClick={onClick} style={{ '--branch-depth': depth } as React.CSSProperties} type="button">
    {depth > 0 && <span className="branch-mark" aria-hidden="true">↳</span>}
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
  const orderedSessions = useMemo(() => sessionTree(sessions), [sessions]);
  const refreshSessions = useCallback(async () => { const items = await api.sessions(); setSessions(items); return items; }, []);
  const loadDetail = useCallback(async (id: string) => { const value = await api.session(id); if (activeIdRef.current === id) { setDetail(value); if (value.report) setStreamedReport(value.report); } return value; }, []);
  const updateSession = useCallback((id: string, patch: Partial<SessionSummary>) => setSessions((items) => items.map((item) => item.id === id ? { ...item, ...patch } : item)), []);
  const renderedReport = useMemo(() => cleanReport(streamedReport, detail?.evidence ?? []), [streamedReport, detail]);

  const connect = useCallback((session: SessionSummary) => {
    streamRef.current?.close();
    const replayingTerminal = terminal.has(session.status);
    if (replayingTerminal) void loadDetail(session.id);
    const stream = new EventSource(eventStreamUrl(session.id)); streamRef.current = stream;
    let closedIntentionally = false;
    stream.onopen = () => setError((value) => value === 'Live connection interrupted. Reconnecting…' ? null : value);
    const consume = (event: MessageEvent) => {
      const key = `${session.id}:${event.lastEventId}`;
      if (seenEventsRef.current.has(key)) return null;
      seenEventsRef.current.add(key);
      return JSON.parse(event.data) as { message?: string; source_domain?: string; stage?: string; text?: string; status?: SessionStatus; tasks?: SessionSummary['plan']; task_id?: string };
    };
    const push = (data: { message?: string; source_domain?: string; stage?: string; task_id?: string }, event: MessageEvent, tone?: Activity['tone']) => {
      if (data.message && activeIdRef.current === session.id) setActivity((items) => [...items, { id: `${session.id}-${event.lastEventId}`, message: data.message!, sourceDomain: data.source_domain, stage: data.stage, taskId: data.task_id, tone }]);
    };
    stream.addEventListener('session.created', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (data) push(data, event); });
    stream.addEventListener('plan.ready', async (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; push(data, event, 'done'); if (!replayingTerminal) { updateSession(session.id, { status: 'ready', plan: data.tasks ?? [] }); closedIntentionally = true; stream.close(); await loadDetail(session.id); } });
    stream.addEventListener('plan.rejected', async (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; push(data, event, 'warning'); closedIntentionally = true; stream.close(); if (!replayingTerminal) { updateSession(session.id, { status: 'rejected', error: data.message ?? 'Please rewrite the question.' }); await loadDetail(session.id); } });
    stream.addEventListener('session.started', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; if (!replayingTerminal) updateSession(session.id, { status: 'running' }); push(data, event); });
    stream.addEventListener('research.progress', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (data) push(data, event); });
    stream.addEventListener('report.started', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; if (!replayingTerminal) { updateSession(session.id, { status: 'synthesizing' }); if (activeIdRef.current === session.id) setStreamedReport(''); } push(data, event); });
    stream.addEventListener('report.chunk', (raw) => { const data = consume(raw as MessageEvent); if (!replayingTerminal && data?.text && activeIdRef.current === session.id) setStreamedReport((value) => value + data.text); });
    stream.addEventListener('session.completed', async (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; updateSession(session.id, { status: data.status ?? 'completed' }); push(data, event, 'done'); closedIntentionally = true; stream.close(); await Promise.all([loadDetail(session.id), refreshSessions()]); });
    stream.addEventListener('session.failed', async (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; updateSession(session.id, { status: 'failed' }); push(data, event, 'warning'); closedIntentionally = true; stream.close(); await Promise.all([loadDetail(session.id), refreshSessions()]); });
    stream.onerror = () => { if (!replayingTerminal && !closedIntentionally && stream.readyState !== EventSource.CLOSED) setError('Live connection interrupted. Reconnecting…'); };
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
  const openCitation = useCallback((claimId: string, sourceId: string) => { const selection = findCitation(detail?.evidence ?? [], claimId, sourceId); if (selection) setDrawer(selection); }, [detail]);

  return <main className="shell">
    <aside className={`sidebar${sidebarOpen ? ' open' : ''}`}><div className="brand"><span className="brand-orbit" aria-hidden="true" /><strong>Parallax</strong></div><button className="new-thread" onClick={newResearch} type="button"><span>＋</span> New research</button><p className="sidebar-label">Your research</p><nav className="history" aria-label="Research history">{orderedSessions.length === 0 && <p className="history-empty">Research sessions will appear here.</p>}{orderedSessions.map(({ session, depth }) => <SessionItem active={activeId === session.id} depth={depth} key={session.id} onClick={() => selectSession(session)} session={session} />)}</nav></aside>
    <button className="mobile-menu" onClick={() => setSidebarOpen((value) => !value)} type="button" aria-label="Toggle research history">☰</button>
    <section className="main-column">{error && <div className="error-banner" role="alert">{error}<button onClick={() => setError(null)} type="button">Dismiss</button></div>}{!active ? <Welcome query={query} setQuery={setQuery} submit={submit} busy={busy} configured={configured} /> : <ResearchConversation active={active} activity={activity} busy={busy} detail={detail} newResearch={newResearch} openCitation={openCitation} report={renderedReport} setView={setView} startResearch={startResearch} view={view} />}</section>
    {drawer && <EvidenceDrawer branching={branching} close={() => setDrawer(null)} selection={drawer} startBranch={startBranch} />}
  </main>;
}

function Welcome({ query, setQuery, submit, busy, configured }: { query: string; setQuery: (value: string) => void; submit: (event: FormEvent<HTMLFormElement>) => void; busy: boolean; configured: boolean | null }) {
  return <div className="welcome"><div className="welcome-mark"><span /><span /></div><h1>What should we research?</h1><p>Ask a complex question. Review the plan before any sources are searched.</p><form className="composer" onSubmit={submit}><textarea autoFocus onChange={(event) => setQuery(event.target.value)} placeholder="Ask a research question…" rows={4} value={query} /><div><span>Evidence-aware deep research</span><button disabled={!query.trim() || busy || configured === false} type="submit">{busy ? 'Planning…' : 'Create plan'} <i aria-hidden="true">↑</i></button></div></form><div className="examples">{examples.map((example) => <button key={example} onClick={() => setQuery(example)} type="button">{example}</button>)}</div></div>;
}

function ResearchConversation({ active, activity, busy, detail, newResearch, openCitation, report, setView, startResearch, view }: { active: SessionSummary; activity: Activity[]; busy: boolean; detail: SessionDetail | null; newResearch: () => void; openCitation: (claimId: string, sourceId: string) => void; report: string; setView: (view: View) => void; startResearch: () => void; view: View }) {
  return <div className="conversation"><header className="conversation-header"><div><p>{active.branch ? `Branched from ${active.branch.source_id}` : 'Deep research'}</p><strong>{active.title}</strong></div><span className={`run-status status-${active.status}`}><i />{statusLabel(active.status)}</span></header><div className="thread"><div className="user-message"><p>{active.branch?.claim_text ?? active.query}</p></div>{active.status === 'planning' && <PlanningState />}{active.status === 'ready' && <PlanReview plan={active.plan} busy={busy} start={startResearch} />}{(active.status === 'queued' || active.status === 'running') && <ResearchingState activity={activity} plan={active.plan} />}{active.status === 'failed' && <div className="assistant-message failure"><span className="assistant-mark">P</span><div><h2>Research stopped</h2><p>{active.error ?? activity.at(-1)?.message ?? 'Research could not complete.'}</p></div></div>}{active.status === 'rejected' && <div className="assistant-message revision"><span className="assistant-mark">P</span><div><h2>This needs a clearer research question</h2><p>{active.error ?? 'Add a specific topic, comparison, outcome, or time range.'}</p><button onClick={newResearch} type="button">Rewrite question</button></div></div>}{(active.status === 'synthesizing' || (terminal.has(active.status) && !['failed', 'rejected'].includes(active.status))) && <div className="response-area"><div className="answer-tabs"><button className={view === 'answer' ? 'active' : ''} onClick={() => setView('answer')} type="button">Answer</button><button className={view === 'process' ? 'active' : ''} onClick={() => setView('process')} type="button">Research process</button></div>{view === 'answer' ? <Report report={report} evidence={detail?.evidence ?? []} openCitation={openCitation} streaming={active.status === 'synthesizing'} /> : <ProcessView activity={activity} detail={detail} plan={active.plan} />}</div>}</div></div>;
}

function PlanningState() { return <div className="assistant-message"><span className="assistant-mark thinking">P</span><div className="planning-copy"><h2>Building an actionable plan</h2><p>Breaking your question into focused, non-overlapping evidence paths.</p><div className="thinking-lines"><span /><span /><span /></div></div></div>; }

function PlanReview({ plan, busy, start }: { plan: SessionSummary['plan']; busy: boolean; start: () => void }) {
  return <div className="assistant-message plan-message"><span className="assistant-mark">P</span><div><p className="assistant-intro">I’ll investigate four evidence paths in parallel and scan broadly across the web. Review the scope, then start research.</p><section className="plan-card"><div className="plan-card-head"><div><span>Research plan</span><strong>{plan.length} steps</strong></div><span className="budget-note">Up to 600 sources</span></div><ol>{plan.map((task, index) => <li key={task.id} style={{ '--delay': `${index * 70}ms` } as React.CSSProperties}><span>{index + 1}</span><div><strong>{task.question}</strong><p>{task.rationale}</p></div><small>{task.priority}</small></li>)}</ol><div className="plan-actions"><p>No web research begins before approval.</p><button disabled={busy} onClick={start} type="button">{busy ? 'Starting…' : 'Start research'} <span aria-hidden="true">→</span></button></div></section></div></div>;
}

function ResearchingState({ activity, plan }: { activity: Activity[]; plan: SessionSummary['plan'] }) {
  const [messageIndex, setMessageIndex] = useState(0);
  const [domainIndex, setDomainIndex] = useState(0);
  const progress = useMemo(() => {
    const activeTasks = new Set<string>();
    const domains: string[] = [];
    const seenDomains = new Set<string>();
    let searches = 0; let sources = 0; let pages = 0; let observations = 0; let contradictions = 0;
    for (const item of activity) {
      if (item.taskId) activeTasks.add(item.taskId);
      if (item.sourceDomain && !seenDomains.has(item.sourceDomain)) {
        seenDomains.add(item.sourceDomain); domains.push(item.sourceDomain);
      }
      if (item.stage === 'search.executed') searches += 1;
      else if (item.stage === 'source.discovered') sources += 1;
      else if (item.stage === 'page.explored') pages += 1;
      else if (item.stage === 'observation.extracted') observations += 1;
      else if (item.stage === 'ledger.contradiction_added') { contradictions += 1; observations += 1; }
    }
    return { activeTasks, contradictions, domains, observations, pages, searches, sources };
  }, [activity]);
  const { activeTasks, contradictions, domains, observations, pages, searches, sources } = progress;
  const messages = [
    searches ? `Searching broadly across ${searches} focused queries` : 'Preparing diverse searches across four evidence paths',
    sources ? `Screening ${sources} distinct results before deeper reading` : 'Opening results and removing duplicate URLs',
    pages ? `Reading and filtering ${pages} sources for usable evidence` : 'Opening results and removing duplicate URLs',
    observations ? `Compressing ${observations} observations into the evidence ledger` : 'Checking primary sources, methods, and counter-evidence',
    contradictions ? `Preserving ${contradictions} conflicting findings for comparison` : 'Cross-checking support against contradicting perspectives',
    domains.length ? `Comparing evidence across ${domains.length} distinct domains` : 'Building a broad, auditable source base',
  ];
  const messageCount = messages.length;
  useEffect(() => { const timer = window.setInterval(() => setMessageIndex((value) => (value + 1) % messageCount), 4200); return () => window.clearInterval(timer); }, [messageCount]);
  useEffect(() => { if (domains.length < 2) return; const timer = window.setInterval(() => setDomainIndex((value) => (value + 1) % domains.length), 2600); return () => window.clearInterval(timer); }, [domains.length]);
  const currentDomain = domains.length ? domains[domainIndex % domains.length] : null;
  const visibleDomains = domains.length ? Array.from({ length: Math.min(5, domains.length) }, (_, offset) => domains[(domainIndex + offset) % domains.length]) : [];
  return <div className="assistant-message"><span className="assistant-mark thinking">P</span><div className="research-live"><h2>Researching across sources</h2><p className="live-message" key={messageIndex}><span />{messages[messageIndex % messages.length]}</p><p className="sr-only" role="status">Research is in progress.</p><div className="source-pulse"><div className="source-orbits" aria-hidden="true">{visibleDomains.length ? visibleDomains.map((domain) => { const icon = sourceIcon(domain); return <span key={domain} style={{ '--source-hue': icon.hue } as React.CSSProperties}>{icon.initial}</span>; }) : <><span>·</span><span>·</span><span>·</span></>}</div><div><small>{currentDomain ? 'Screening now' : 'Discovering sources'}</small><strong>{currentDomain ?? 'Finding reliable domains'}</strong></div></div><div className="research-stats"><span><strong>{sources}</strong> sources</span><span><strong>{pages}</strong> read</span><span><strong>{observations}</strong> evidence</span></div><div className="live-plan">{plan.map((task, index) => <div className={activeTasks.has(task.id) ? 'visited' : ''} key={task.id}><span>{activeTasks.has(task.id) ? '✓' : index + 1}</span><p>{task.question}</p></div>)}</div><p className="leave-note">While this one is under works, you&apos;re free to start a new research.</p></div></div>;
}

function Report({ report, evidence, openCitation, streaming }: { report: string; evidence: EvidenceClaim[]; openCitation: (claimId: string, sourceId: string) => void; streaming: boolean }) {
  return <div className="assistant-message report-message"><span className="assistant-mark">P</span><article className="report"><ReactMarkdown remarkPlugins={[remarkGfm]} components={{ img: () => null, a: ({ href, children }) => { if (href?.startsWith('#evidence-')) { const match = href.match(/^#evidence-(.+)-(S\d+)$/); const claimId = match?.[1] ?? ''; const sourceId = match?.[2] ?? ''; const exists = evidence.some((claim) => claim.claim_id === claimId && claim.observations.some((item) => item.source_id === sourceId)); return exists ? <button className="citation" onClick={() => openCitation(claimId, sourceId)} onFocus={() => openCitation(claimId, sourceId)} onMouseEnter={() => openCitation(claimId, sourceId)} type="button">{children}</button> : <span>{children}</span>; } return <span>{children}</span>; } }}>{report}</ReactMarkdown>{streaming && <span className="stream-caret" aria-label="Answer is streaming" />}</article></div>;
}

function ProcessView({ activity, detail, plan }: { activity: Activity[]; detail: SessionDetail | null; plan: SessionSummary['plan'] }) {
  const count = (stage: string) => activity.filter((item) => item.stage === stage).length;
  const stages = [
    ['Plan approved', `${plan.length} focused evidence paths`],
    ['Sources searched', `${count('search.executed')} focused searches`],
    ['Pages compressed', `${count('page.explored')} pages processed · ${count('page.fetch_failed')} unavailable`],
    ['Evidence ledger built', `${count('observation.extracted') + count('ledger.contradiction_added')} atomic observations`],
    ['Critic and answer complete', `${count('critic.followup_created')} bounded follow-up paths`],
  ];
  return <div className="process-view"><div className="process-summary"><strong>{plan.length}</strong><span>planned paths</span><strong>{detail?.evidence.length ?? 0}</strong><span>ledger claims</span><strong>{detail?.evidence.filter((claim) => claim.disagreement).length ?? 0}</strong><span>contested</span></div><ol>{stages.map(([title, summary]) => <li key={title}><span className="done">✓</span><div><strong>{title}</strong><small>{summary}</small></div></li>)}</ol></div>;
}

function EvidenceDrawer({ branching, close, selection, startBranch }: { branching: string | null; close: () => void; selection: DrawerSelection; startBranch: (observation: Observation) => void }) {
  const { claim, observation } = selection;
  const support = claim.observations.filter((item) => item.polarity === 'support'); const contradictions = claim.observations.filter((item) => item.polarity === 'contradict');
  return <><div className="drawer-backdrop" aria-hidden="true" /><aside className="evidence-drawer" aria-label={`Evidence for ${observation.source_id}`}><header><div><span>{observation.source_id}</span><p>{observation.source_domain}</p></div><button onClick={close} type="button" aria-label="Close">×</button></header><div className="drawer-scroll"><section className="drawer-summary"><h2>{claim.text}</h2><div><strong className={`confidence-${claim.confidence.toLowerCase()}`}>{claim.confidence}</strong><span>{claim.supporting_domain_count} support · {claim.contradicting_domain_count} contradict</span></div></section><a className="selected-source" href={observation.source_url} rel="noreferrer" target="_blank"><p>“{observation.excerpt}”</p><small>Open source ↗</small></a><SourceGroup label="Supporting" observations={support} /><SourceGroup branching={branching} label="Contradicting" observations={contradictions} startBranch={startBranch} />{contradictions.length === 0 && <p className="no-contradiction">No contradicting evidence attached.</p>}</div></aside></>;
}

function SourceGroup({ branching, label, observations, startBranch }: { branching?: string | null; label: string; observations: Observation[]; startBranch?: (observation: Observation) => void }) {
  if (!observations.length) return null;
  return <section className="source-group"><h3>{label}<span>{observations.length}</span></h3>{observations.map((item) => <article key={item.observation_id}><div><span>{item.source_id}</span><a href={item.source_url} rel="noreferrer" target="_blank">{item.source_domain} ↗</a></div><p>{item.excerpt}</p>{item.polarity === 'contradict' && startBranch && <button disabled={branching === item.observation_id} onClick={() => startBranch(item)} type="button">{branching === item.observation_id ? 'Creating branch…' : 'Research this perspective'} <span>→</span></button>}</article>)}</section>;
}
