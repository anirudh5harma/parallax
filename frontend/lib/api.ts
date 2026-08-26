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
  error_code: string | null;
  error_provider: string | null;
  error_retryable: boolean;
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

export type ApiErrorDetails = {
  code: string | null;
  message: string;
  provider: string | null;
  retryable: boolean;
  status: number;
};

export class ApiError extends Error {
  readonly code: string | null;
  readonly provider: string | null;
  readonly retryable: boolean;
  readonly status: number;

  constructor(details: ApiErrorDetails) {
    super(details.message);
    this.name = 'ApiError';
    this.code = details.code;
    this.provider = details.provider;
    this.retryable = details.retryable;
    this.status = details.status;
  }
}

function errorDetails(payload: unknown, status: number): ApiErrorDetails {
  const root = typeof payload === 'object' && payload !== null
    ? payload as Record<string, unknown>
    : {};
  const nested = typeof root.detail === 'object' && root.detail !== null
    ? root.detail as Record<string, unknown>
    : typeof root.error === 'object' && root.error !== null
      ? root.error as Record<string, unknown>
      : {};
  const message = typeof nested.message === 'string'
    ? nested.message
    : typeof root.detail === 'string'
      ? root.detail
      : typeof root.message === 'string'
        ? root.message
        : `Request failed (${status})`;
  const code = typeof nested.code === 'string'
    ? nested.code
    : typeof nested.error_code === 'string'
      ? nested.error_code
      : typeof root.code === 'string'
        ? root.code
        : typeof root.error_code === 'string' ? root.error_code : null;
  const provider = typeof nested.provider === 'string'
    ? nested.provider
    : typeof root.provider === 'string' ? root.provider : null;
  const explicitRetryable = typeof nested.retryable === 'boolean'
    ? nested.retryable
    : typeof root.retryable === 'boolean' ? root.retryable : null;
  return {
    code,
    message,
    provider,
    retryable: explicitRetryable ?? [408, 425, 429, 500, 502, 503, 504].includes(status),
    status,
  };
}

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
    throw new ApiError(errorDetails(payload, response.status));
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
