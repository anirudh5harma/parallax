'use client';

import { Observation } from '../lib/api';
import { DrawerSelection } from '../lib/report';

export function EvidenceDrawer({ branching, close, selection, startBranch }: { branching: string | null; close: () => void; selection: DrawerSelection; startBranch: (observation: Observation) => void }) {
  const { claim, observation } = selection;
  const support = claim.observations.filter((item) => item.polarity === 'support');
  const contradictions = claim.observations.filter((item) => item.polarity === 'contradict');
  return <><button className="drawer-backdrop" aria-label="Close evidence" onClick={close} type="button" /><aside className="evidence-drawer" aria-label={`Evidence for ${observation.source_id}`} role="dialog" onKeyDown={(event) => { if (event.key === 'Escape') close(); }}>
    <header><div><span>{observation.source_id}</span><p>{observation.source_domain}</p></div><button onClick={close} type="button" aria-label="Close">×</button></header>
    <div className="drawer-scroll"><section className="drawer-summary"><h2>{claim.text}</h2><div><strong className={`confidence-${claim.confidence.toLowerCase()}`}>{claim.confidence}</strong><span>{claim.supporting_domain_count} support · {claim.contradicting_domain_count} contradict</span></div></section><a className="selected-source" href={observation.source_url} rel="noreferrer" target="_blank"><p>“{observation.excerpt}”</p><small>Open source ↗</small></a><SourceGroup label="Supporting" observations={support} /><SourceGroup branching={branching} label="Contradicting" observations={contradictions} startBranch={startBranch} />{contradictions.length === 0 && <p className="no-contradiction">No contradicting evidence attached.</p>}</div>
  </aside></>;
}

function SourceGroup({ branching, label, observations, startBranch }: { branching?: string | null; label: string; observations: Observation[]; startBranch?: (observation: Observation) => void }) {
  if (!observations.length) return null;
  return <section className="source-group"><h3>{label}<span>{observations.length}</span></h3>{observations.map((item) => <article key={item.observation_id}><div><span>{item.source_id}</span><a href={item.source_url} rel="noreferrer" target="_blank">{item.source_domain} ↗</a></div><p>{item.excerpt}</p>{item.polarity === 'contradict' && startBranch && <button disabled={branching === item.observation_id} onClick={() => startBranch(item)} type="button">{branching === item.observation_id ? 'Creating branch…' : 'Research this perspective'} <span>→</span></button>}</article>)}</section>;
}
