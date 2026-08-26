export const API_BASE =
  process.env.NEXT_PUBLIC_RESEARCH_API_URL ?? 'http://127.0.0.1:8000';
const REQUEST_TIMEOUT_MS = 45_000;

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

export class ResearchEventStream {
  private readonly controller = new AbortController();
  private readonly listeners = new Map<string, Set<(event: MessageEvent<string>) => void>>();
  private lastEventId = '';
  private stopped = false;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(private readonly sessionId: string) {
    void this.connect();
  }

  get isClosed() { return this.stopped; }

  addEventListener(name: string, listener: (event: MessageEvent<string>) => void) {
    const existing = this.listeners.get(name) ?? new Set();
    existing.add(listener);
    this.listeners.set(name, existing);
  }

  close() {
    this.stopped = true;
    this.controller.abort();
  }

  private dispatch(block: string) {
    let eventName = 'message';
    let eventId = '';
    const data: string[] = [];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) eventName = line.slice(6).trimStart();
      else if (line.startsWith('id:')) eventId = line.slice(3).trimStart();
      else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
    }
    if (!data.length) return;
    if (eventId) this.lastEventId = eventId;
    const event = new MessageEvent('message', {
      data: data.join('\n'),
      lastEventId: eventId,
    });
    for (const listener of this.listeners.get(eventName) ?? []) listener(event);
  }

  private async connect() {
    while (!this.stopped) {
      try {
        const response = await fetch(`${API_BASE}/api/sessions/${this.sessionId}/events`, {
          headers: {
            Accept: 'text/event-stream',
            'Cache-Control': 'no-cache',
            'X-Workspace-Key': workspaceKey(),
            ...(this.lastEventId ? { 'Last-Event-ID': this.lastEventId } : {}),
          },
          signal: this.controller.signal,
        });
        if (!response.ok || !response.body) {
          const payload: unknown = await response.json().catch(() => ({}));
          throw new ApiError(errorDetails(payload, response.status));
        }
        this.onopen?.();
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (!this.stopped) {
          const { done, value } = await reader.read();
          buffer += decoder.decode(value, { stream: !done }).replaceAll('\r\n', '\n');
          let boundary = buffer.indexOf('\n\n');
          while (boundary >= 0) {
            this.dispatch(buffer.slice(0, boundary));
            buffer = buffer.slice(boundary + 2);
            boundary = buffer.indexOf('\n\n');
          }
          if (done) break;
        }
      } catch (cause) {
        if (this.stopped || (cause instanceof DOMException && cause.name === 'AbortError')) return;
        this.onerror?.();
      }
      if (!this.stopped) await new Promise((resolve) => window.setTimeout(resolve, 1_200));
    }
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const method = init?.method ?? 'GET';
  const url = method === 'GET'
    ? `${API_BASE}${path}${path.includes('?') ? '&' : '?'}_=${Date.now()}`
    : `${API_BASE}${path}`;
  const controller = new AbortController();
  let timedOut = false;
  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, REQUEST_TIMEOUT_MS);
  const abortFromCaller = () => controller.abort(init?.signal?.reason);
  init?.signal?.addEventListener('abort', abortFromCaller, { once: true });
  try {
    const response = await fetch(url, {
      ...init,
      signal: controller.signal,
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
  } catch (cause) {
    if (timedOut) {
      throw new ApiError({
        code: 'request_timeout',
        message: 'The research service took too long to respond. Try again.',
        provider: null,
        retryable: true,
        status: 408,
      });
    }
    throw cause;
  } finally {
    window.clearTimeout(timeout);
    init?.signal?.removeEventListener('abort', abortFromCaller);
  }
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
