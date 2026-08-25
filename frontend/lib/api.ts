export const API_BASE =
  process.env.NEXT_PUBLIC_RESEARCH_API_URL ?? 'http://127.0.0.1:8000';

export type SessionStatus =
  | 'planning'
  | 'ready'
  | 'queued'
  | 'running'
  | 'synthesizing'
  | 'completed'
  | 'completed_with_errors'
  | 'failed'
  | 'rejected';

export type Branch = {
  parent_session_id: string;
  claim_id: string;
  observation_id: string;
  source_id: string;
  source_url: string;
  claim_text: string;
};

export type SessionSummary = {
  id: string;
  query: string;
  title: string;
  created_at: string;
  status: SessionStatus;
  parent_session_id: string | null;
  branch: Branch | null;
  claim_count: number;
  contested_count: number;
  error: string | null;
  plan: ResearchPlanTask[];
};

export type ResearchPlanTask = {
  id: string;
  question: string;
  rationale: string;
  priority: 'high' | 'medium' | 'low';
  page_budget_share: number;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
};

export type Observation = {
  observation_id: string;
  source_id: string;
  source_url: string;
  source_domain: string;
  statement: string;
  polarity: 'support' | 'contradict' | 'neutral';
  excerpt: string;
  source_type?: string;
};

export type EvidenceClaim = {
  claim_id: string;
  text: string;
  confidence: 'High' | 'Moderate' | 'Low' | 'Insufficient';
  disagreement: boolean;
  supporting_domain_count: number;
  contradicting_domain_count: number;
  observations: Observation[];
};

export type SessionDetail = SessionSummary & {
  report: string | null;
  evidence: EvidenceClaim[];
  run: Record<string, unknown> | null;
};

let cachedWorkspaceKey: string | null = null;

export function workspaceKey() {
  if (cachedWorkspaceKey) return cachedWorkspaceKey;
  if (typeof window === 'undefined') return 'server-render-placeholder-key-0000';
  const stored = window.localStorage.getItem('parallax-workspace-key');
  if (stored && /^[a-z0-9_-]{32,128}$/i.test(stored)) {
    cachedWorkspaceKey = stored;
    return stored;
  }
  cachedWorkspaceKey = window.crypto.randomUUID().replaceAll('-', '');
  window.localStorage.setItem('parallax-workspace-key', cachedWorkspaceKey);
  return cachedWorkspaceKey;
}

export function eventStreamUrl(sessionId: string) {
  return `${API_BASE}/api/sessions/${sessionId}/events?workspace=${encodeURIComponent(workspaceKey())}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const method = init?.method ?? 'GET';
  const url = method === 'GET'
    ? `${API_BASE}${path}${path.includes('?') ? '&' : '?'}_=${Date.now()}`
    : `${API_BASE}${path}`;
  const response = await fetch(url, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      'X-Workspace-Key': workspaceKey(),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => ({}));
    const detail = (
      typeof payload === 'object'
      && payload !== null
      && 'detail' in payload
      && typeof payload.detail === 'string'
    ) ? payload.detail : null;
    throw new Error(detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; configured: boolean; model: string }>('/api/health'),
  sessions: () => request<SessionSummary[]>('/api/sessions'),
  session: (id: string) => request<SessionDetail>(`/api/sessions/${id}`),
  create: (query: string) =>
    request<SessionSummary>('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ query }),
    }),
  start: (id: string) =>
    request<SessionSummary>(`/api/sessions/${id}/start`, {
      method: 'POST',
    }),
  branch: (sessionId: string, observationId: string) =>
    request<SessionSummary>(`/api/sessions/${sessionId}/branches`, {
      method: 'POST',
      body: JSON.stringify({ observation_id: observationId }),
    }),
};
