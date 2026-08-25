export const API_BASE =
  process.env.NEXT_PUBLIC_RESEARCH_API_URL ?? 'http://127.0.0.1:8000';

export type SessionStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'completed_with_errors'
  | 'failed';

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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail ?? `Request failed (${response.status})`);
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
  branch: (sessionId: string, observationId: string) =>
    request<SessionSummary>(`/api/sessions/${sessionId}/branches`, {
      method: 'POST',
      body: JSON.stringify({ observation_id: observationId }),
    }),
};
