'use client';

import { FormEvent, useState } from 'react';

const sampleSessions = [
  { title: 'Free-threaded Python 3.13', meta: '14 claims · 1 contested', active: true },
  { title: 'Remote work productivity', meta: '22 claims · 4 contested', active: false },
];

export default function Home() {
  const [query, setQuery] = useState('');

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark" aria-hidden="true"><span /><span /></div>
          <div><p className="brand-name">PARALLAX</p><p className="brand-subtitle">Evidence research</p></div>
        </div>

        <button className="new-research" type="button"><span aria-hidden="true">＋</span>New research</button>

        <div className="session-heading"><span>Research sessions</span><span>{sampleSessions.length}</span></div>
        <nav className="session-list" aria-label="Research sessions">
          {sampleSessions.map((session) => (
            <button className={`session-item${session.active ? ' active' : ''}`} key={session.title} type="button">
              <span className="session-state" aria-hidden="true" />
              <span className="session-copy"><strong>{session.title}</strong><small>{session.meta}</small></span>
            </button>
          ))}
          <div className="branch-item">
            <span className="branch-line" aria-hidden="true" /><span className="branch-arrow" aria-hidden="true">↳</span>
            <span><strong>Adaptive interpreter path</strong><small>Branched from contradiction S4</small></span>
          </div>
        </nav>

        <div className="sidebar-foot"><span className="status-dot" /><span>Bedrock · Opus</span><span className="local-pill">Local</span></div>
      </aside>

      <section className="workspace">
        <header className="workspace-header">
          <div><p className="eyebrow">Evidence workspace</p><h1>Start a research thread</h1></div>
          <div className="header-meta"><span className="pulse" aria-hidden="true" /><span>Ready</span></div>
        </header>

        <div className="hero-grid">
          <section className="query-panel">
            <div className="hero-copy">
              <p className="hero-kicker">Deep research, without flattened certainty</p>
              <h2>Ask the question.<br />Keep the disagreement.</h2>
              <p>Parallax researches multiple evidence paths in parallel, then shows where sources support, contradict, or leave the record unresolved.</p>
            </div>

            <form className="query-form" onSubmit={submit}>
              <label htmlFor="research-query">What do you want to understand?</label>
              <textarea id="research-query" onChange={(event) => setQuery(event.target.value)} placeholder="e.g. Does a four-day workweek improve productivity without increasing burnout?" rows={5} value={query} />
              <div className="composer-foot">
                <div className="composer-note"><span className="mini-ledger" aria-hidden="true" />Evidence ledger included</div>
                <button disabled={!query.trim()} type="submit">Begin research <span aria-hidden="true">→</span></button>
              </div>
            </form>

            <div className="prompts" aria-label="Example questions">
              <button type="button" onClick={() => setQuery('What evidence supports and challenges remote work productivity?')}>Remote work</button>
              <button type="button" onClick={() => setQuery('How effective are GLP-1 medicines for long-term weight management, and where does evidence disagree?')}>GLP-1 outcomes</button>
              <button type="button" onClick={() => setQuery('What changed in free-threaded Python 3.13, and what limitations remain?')}>Python 3.13</button>
            </div>
          </section>

          <aside className="ledger-preview" aria-label="Evidence ledger preview">
            <div className="preview-topline"><span>HOW PARALLAX THINKS</span><span>01 / 03</span></div>
            <h3>Evidence remains steerable</h3>
            <p className="preview-intro">Every finding keeps its source strength, contradiction, and unresolved gaps.</p>
            <div className="claim-card support">
              <div className="claim-head"><span>SUPPORTED</span><strong>High</strong></div>
              <p>Independent sources converge on the primary outcome.</p>
              <div className="source-row"><span>S1</span><span>S2</span><span>S3</span><small>3 domains</small></div>
            </div>
            <div className="claim-card contradict">
              <div className="claim-head"><span>CONTESTED</span><strong>Low</strong></div>
              <p>A credible source reaches a different conclusion.</p>
              <button type="button">Follow contradiction S4 <span aria-hidden="true">↗</span></button>
            </div>
            <div className="preview-footer"><span>Planner</span><i /><span>Researchers</span><i /><span>Critic</span></div>
          </aside>
        </div>
      </section>
    </main>
  );
}
