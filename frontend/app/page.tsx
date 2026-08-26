'use client';

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ApiError, EvidenceClaim, Observation, ResearchEventStream, SessionDetail, SessionStatus, SessionSummary, api } from '../lib/api';
import { EvidenceDrawer } from '../components/evidence-drawer';
import { ErrorModal } from '../components/error-modal';
import { ErrorNotice, errorNotice } from '../lib/errors';
import { DrawerSelection, cleanReport, findCitation } from '../lib/report';

type Activity = { id: string; message: string; stage?: string; sourceDomain?: string; taskId?: string; tone?: 'warning' | 'done' };
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
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [composing, setComposing] = useState(false);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [activity, setActivity] = useState<Activity[]>([]);
  const [streamedReport, setStreamedReport] = useState('');
  const [drawer, setDrawer] = useState<DrawerSelection | null>(null);
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [error, setError] = useState<ErrorNotice | null>(null);
  const [busy, setBusy] = useState(false);
  const [branching, setBranching] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const streamRef = useRef<ResearchEventStream | null>(null);
  const activeIdRef = useRef<string | null>(null);
  const seenEventsRef = useRef(new Set<string>());
  const active = composing ? null : sessions.find((item) => item.id === activeId) ?? null;
  const orderedSessions = useMemo(() => sessionTree(sessions), [sessions]);
  const refreshSessions = useCallback(async () => { const items = await api.sessions(); setSessions(items); return items; }, []);
  const loadDetail = useCallback(async (id: string) => { const value = await api.session(id); if (activeIdRef.current === id) { setDetail(value); if (value.report) setStreamedReport(value.report); } return value; }, []);
  const updateSession = useCallback((id: string, patch: Partial<SessionSummary>) => setSessions((items) => items.map((item) => item.id === id ? { ...item, ...patch } : item)), []);
  const renderedReport = useMemo(() => cleanReport(streamedReport, detail?.evidence ?? []), [streamedReport, detail]);

  const connect = useCallback((session: SessionSummary) => {
    streamRef.current?.close();
    const replayingTerminal = terminal.has(session.status);
    const backgroundFailure = (cause: unknown) => {
      if (activeIdRef.current === session.id) {
        setError(errorNotice(cause, 'Research details could not refresh', () => connect(session)));
      }
    };
    if (replayingTerminal || session.status === 'ready') {
      void loadDetail(session.id).catch(backgroundFailure);
      return;
    }
    const stream = new ResearchEventStream(session.id); streamRef.current = stream;
    let closedIntentionally = false;
    let polling = false;
    stream.onopen = () => setError((value) => value?.code === 'connection_interrupted' ? null : value);
    const consume = (event: MessageEvent) => {
      const key = `${session.id}:${event.lastEventId}`;
      if (seenEventsRef.current.has(key)) return null;
      seenEventsRef.current.add(key);
      return JSON.parse(event.data) as { code?: string; error_code?: string; message?: string; provider?: string; retryable?: boolean; source_domain?: string; stage?: string; text?: string; status?: SessionStatus; tasks?: SessionSummary['plan']; task_id?: string };
    };
    const push = (data: { message?: string; source_domain?: string; stage?: string; task_id?: string }, event: MessageEvent, tone?: Activity['tone']) => {
      if (data.message && activeIdRef.current === session.id) setActivity((items) => [...items, { id: `${session.id}-${event.lastEventId}`, message: data.message!, sourceDomain: data.source_domain, stage: data.stage, taskId: data.task_id, tone }]);
    };
    stream.addEventListener('session.created', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (data) push(data, event); });
    stream.addEventListener('plan.ready', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data || session.status !== 'planning') return; push(data, event, 'done'); if (!replayingTerminal) { updateSession(session.id, { status: 'ready', plan: data.tasks ?? [] }); closedIntentionally = true; stream.close(); void loadDetail(session.id).catch(backgroundFailure); } });
    stream.addEventListener('plan.rejected', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; push(data, event, 'warning'); closedIntentionally = true; stream.close(); if (!replayingTerminal) { updateSession(session.id, { status: 'rejected', error: data.message ?? 'Please rewrite the question.' }); void loadDetail(session.id).catch(backgroundFailure); } });
    stream.addEventListener('session.started', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; if (!replayingTerminal) updateSession(session.id, { status: 'running' }); push(data, event); });
    stream.addEventListener('research.progress', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (data) push(data, event); });
    stream.addEventListener('report.started', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; if (!replayingTerminal) { updateSession(session.id, { status: 'synthesizing' }); if (activeIdRef.current === session.id) setStreamedReport(''); } push(data, event); });
    stream.addEventListener('report.chunk', (raw) => { const data = consume(raw as MessageEvent); if (!replayingTerminal && data?.text && activeIdRef.current === session.id) setStreamedReport((value) => value + data.text); });
    stream.addEventListener('session.completed', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; updateSession(session.id, { status: data.status ?? 'completed' }); push(data, event, 'done'); closedIntentionally = true; stream.close(); void Promise.all([loadDetail(session.id), refreshSessions()]).catch(backgroundFailure); });
    stream.addEventListener('session.failed', (raw) => { const event = raw as MessageEvent; const data = consume(event); if (!data) return; updateSession(session.id, { status: 'failed' }); push(data, event, 'warning'); setError(errorNotice(new ApiError({ code: data.code ?? data.error_code ?? null, message: data.message ?? 'Research could not complete.', provider: data.provider ?? null, retryable: data.retryable ?? false, status: 0 }), 'Research could not complete')); closedIntentionally = true; stream.close(); void Promise.all([loadDetail(session.id), refreshSessions()]).catch(backgroundFailure); });
    stream.onerror = () => {
      if (replayingTerminal || closedIntentionally || stream.isClosed || polling) return;
      polling = true;
      window.setTimeout(() => {
        void loadDetail(session.id).then((value) => {
          updateSession(session.id, { status: value.status });
          if (terminal.has(value.status) || value.status === 'ready') {
            closedIntentionally = true;
            stream.close();
            return refreshSessions();
          }
        }).catch(() => undefined).finally(() => { polling = false; });
      }, 2_500);
    };
  }, [loadDetail, refreshSessions, updateSession]);

  useEffect(() => {
    Promise.all([api.health(), api.sessions()]).then(([health, items]) => { setConfigured(health.configured); setSessions(items); const wantsComposer = new URLSearchParams(window.location.search).has('new'); setComposing(wantsComposer); if (items[0] && !wantsComposer) { activeIdRef.current = items[0].id; setActiveId(items[0].id); connect(items[0]); } }).catch((cause: unknown) => setError(errorNotice(cause, 'Research service unavailable', () => window.location.reload())));
    return () => streamRef.current?.close();
  }, [connect]);

  function selectSession(session: SessionSummary) {
    window.history.replaceState(null, '', '/'); streamRef.current?.close(); seenEventsRef.current.clear(); activeIdRef.current = session.id; setActiveId(session.id); setComposing(false); setDetail(null); setActivity([]); setStreamedReport(''); setDrawer(null); setError(null); setSidebarOpen(false); connect(session);
  }
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); await createPlan();
  }
  async function createPlan() {
    if (!query.trim() || busy) return; setBusy(true); setError(null);
    try { const session = await api.create(query.trim()); setSessions((items) => [session, ...items]); setQuery(''); selectSession(session); }
    catch (cause) { setError(errorNotice(cause, 'Plan could not be created', () => void createPlan())); }
    finally { setBusy(false); }
  }
  async function startResearch() {
    if (!active || busy) return; setBusy(true); setError(null); setActivity([]); setStreamedReport('');
    try { const session = await api.start(active.id); updateSession(active.id, { status: session.status }); connect(session); }
    catch (cause) { setError(errorNotice(cause, 'Research could not start', () => void startResearch())); }
    finally { setBusy(false); }
  }
  async function startBranch(observation: Observation) {
    if (!detail || branching) return; setBranching(observation.observation_id); setError(null);
    try { const session = await api.branch(detail.id, observation.observation_id); setSessions((items) => [session, ...items]); setDrawer(null); selectSession(session); }
    catch (cause) { setError(errorNotice(cause, 'Evidence path could not start', () => void startBranch(observation))); }
    finally { setBranching(null); }
  }
  function newResearch() { router.push('/?new=1'); streamRef.current?.close(); seenEventsRef.current.clear(); activeIdRef.current = null; setActiveId(null); setComposing(true); setDetail(null); setActivity([]); setStreamedReport(''); setDrawer(null); setError(null); setSidebarOpen(false); }
  const openCitation = useCallback((claimId: string, sourceId: string) => { const selection = findCitation(detail?.evidence ?? [], claimId, sourceId); if (selection) setDrawer(selection); }, [detail]);

  return <main className="shell">
    <aside className={`sidebar${sidebarOpen ? ' open' : ''}`}><div className="brand"><span className="brand-orbit" aria-hidden="true" /><strong>Parallax</strong></div><Link className="new-thread" href="/?new=1" onClick={(event) => { event.preventDefault(); newResearch(); }}><span>＋</span> New research</Link><p className="sidebar-label">Your research</p><nav className="history" aria-label="Research history">{orderedSessions.length === 0 && <p className="history-empty">Research sessions will appear here.</p>}{orderedSessions.map(({ session, depth }) => <SessionItem active={activeId === session.id} depth={depth} key={session.id} onClick={() => selectSession(session)} session={session} />)}</nav></aside>
    <button className="mobile-menu" onClick={() => setSidebarOpen((value) => !value)} type="button" aria-label="Toggle research history">☰</button>
    <section className="main-column">{!active ? <Welcome query={query} setQuery={setQuery} submit={submit} busy={busy} configured={configured} /> : <ResearchConversation active={active} activity={activity} busy={busy} detail={detail} newResearch={newResearch} openCitation={openCitation} report={renderedReport} startResearch={startResearch} />}</section>
    {drawer && <EvidenceDrawer branching={branching} close={() => setDrawer(null)} selection={drawer} startBranch={startBranch} />}
    {error && <ErrorModal close={() => setError(null)} notice={error} />}
  </main>;
}

function Welcome({ query, setQuery, submit, busy, configured }: { query: string; setQuery: (value: string) => void; submit: (event: FormEvent<HTMLFormElement>) => void; busy: boolean; configured: boolean | null }) {
  return <div className="welcome"><div className="welcome-mark"><span /><span /></div><h1>What should we research?</h1><p>Ask a complex question. Review the plan before any sources are searched.</p><form className="composer" onSubmit={submit}><textarea autoFocus onChange={(event) => setQuery(event.target.value)} placeholder="Ask a research question…" rows={4} value={query} /><div><span>Evidence-aware deep research</span><button disabled={!query.trim() || busy || configured === false} type="submit">{busy ? 'Planning…' : 'Create plan'} <i aria-hidden="true">↑</i></button></div></form><div className="examples">{examples.map((example) => <button key={example} onClick={() => setQuery(example)} type="button">{example}</button>)}</div></div>;
}

function ResearchConversation({ active, activity, busy, detail, newResearch, openCitation, report, startResearch }: { active: SessionSummary; activity: Activity[]; busy: boolean; detail: SessionDetail | null; newResearch: () => void; openCitation: (claimId: string, sourceId: string) => void; report: string; startResearch: () => void }) {
  return <div className="conversation"><header className="conversation-header"><div><p>{active.branch ? `Branched from ${active.branch.source_id}` : 'Deep research'}</p><strong>{active.title}</strong></div><span className={`run-status status-${active.status}`}><i />{statusLabel(active.status)}</span></header><div className="thread"><div className="user-message"><p>{active.branch?.claim_text ?? active.query}</p></div>{active.status === 'planning' && <PlanningState />}{active.status === 'ready' && <PlanReview plan={active.plan} busy={busy} start={startResearch} />}{(active.status === 'queued' || active.status === 'running') && <ResearchingState activity={activity} plan={active.plan} />}{active.status === 'failed' && <div className="assistant-message failure"><span className="assistant-mark">P</span><div><h2>Research stopped</h2><p>{active.error ?? activity.at(-1)?.message ?? 'Research could not complete.'}</p></div></div>}{active.status === 'rejected' && <div className="assistant-message revision"><span className="assistant-mark">P</span><div><h2>This needs a clearer research question</h2><p>{active.error ?? 'Add a specific topic, comparison, outcome, or time range.'}</p><button onClick={newResearch} type="button">Rewrite question</button></div></div>}{(active.status === 'synthesizing' || (terminal.has(active.status) && !['failed', 'rejected'].includes(active.status))) && <div className="response-area">{active.status === 'synthesizing' && !report ? <FinalizingState /> : <Report report={report} evidence={detail?.evidence ?? []} openCitation={openCitation} streaming={active.status === 'synthesizing'} />}</div>}</div></div>;
}

function PlanningState() { return <div className="assistant-message"><span className="assistant-mark thinking">P</span><div className="planning-copy"><h2>Building an actionable plan</h2><p>Breaking your question into focused, non-overlapping evidence paths.</p><div className="thinking-lines"><span /><span /><span /></div></div></div>; }

function FinalizingState() {
  const [step, setStep] = useState(0);
  const steps = [
    'Prioritizing the strongest evidence-backed findings',
    'Separating consensus from meaningful disagreement',
    'Binding citations to the claims they support',
    'Condensing unresolved questions into remaining gaps',
  ];
  useEffect(() => { const timer = window.setInterval(() => setStep((value) => (value + 1) % steps.length), 3600); return () => window.clearInterval(timer); }, [steps.length]);
  return <div className="assistant-message finalizing-message"><span className="assistant-mark thinking">P</span><div className="finalizing-copy"><p className="finalizing-label">Research complete</p><h2>Framing the final report</h2><p className="finalizing-step" key={step}><span aria-hidden="true" />{steps[step]}</p><div className="finalizing-progress" aria-hidden="true">{steps.map((_, index) => <span className={index === step ? 'active' : ''} key={index} />)}</div><p className="sr-only" role="status">Framing the final report. {steps[step]}</p></div></div>;
}

function PlanReview({ plan, busy, start }: { plan: SessionSummary['plan']; busy: boolean; start: () => void }) {
  return <div className="assistant-message plan-message"><span className="assistant-mark">P</span><div><p className="assistant-intro">I’ll investigate four evidence paths in parallel and scan broadly across the web. Review the scope, then start research.</p><section className="plan-card"><div className="plan-card-head"><div><span>Research plan</span><strong>{plan.length} steps</strong></div><span className="budget-note">Target 600+ screened</span></div><ol>{plan.map((task, index) => <li key={task.id} style={{ '--delay': `${index * 70}ms` } as React.CSSProperties}><span>{index + 1}</span><div><strong>{task.question}</strong><p>{task.rationale}</p></div><small>{task.priority}</small></li>)}</ol><div className="plan-actions"><p>No web research begins before approval.</p><button disabled={busy} onClick={start} type="button">{busy ? 'Starting…' : 'Start research'} <span aria-hidden="true">→</span></button></div></section></div></div>;
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
