import { ApiError } from './api';

export type ErrorNotice = {
  title: string;
  body: string;
  code?: string;
  retry?: () => void;
};

export function errorNotice(
  cause: unknown,
  fallback: string,
  retry?: () => void,
): ErrorNotice {
  const apiError = cause instanceof ApiError ? cause : null;
  const message = apiError?.message ?? (cause instanceof Error ? cause.message : fallback);
  const signature = `${apiError?.code ?? ''} ${apiError?.provider ?? ''} ${message}`.toLowerCase();
  const retryAction = apiError?.retryable === false ? undefined : retry;

  if (/tavily/.test(signature) && /(quota|credit|exhaust|limit)/.test(signature)) {
    return { title: 'Search credits exhausted', body: 'Search provider allowance has been reached. Update its plan or credits before starting more research.', code: apiError?.code ?? undefined };
  }
  if (/(bedrock|model|anthropic)/.test(signature) && /(quota|credit|exhaust|limit)/.test(signature)) {
    return { title: 'Model usage limit reached', body: 'Model provider allowance has been reached. Update its billing or limits, then try again.', code: apiError?.code ?? undefined };
  }
  if (/tavily/.test(signature) && /(access.denied|unauthori|invalid.key|authentication)/.test(signature)) {
    return { title: 'Search access unavailable', body: 'Check configured search provider credentials, then restart the research service.', code: apiError?.code ?? undefined };
  }
  if (/(bedrock|model|anthropic)/.test(signature) && /(access.denied|not.available|model.access|unsupported.model)/.test(signature)) {
    return { title: 'Model access unavailable', body: 'Configured model is not available to this account. Choose an enabled model or update model access.', code: apiError?.code ?? undefined };
  }
  if (/daily anonymous/.test(signature)) {
    return { title: 'Daily research limit reached', body: 'This service has used its anonymous research allowance for today.', code: apiError?.code ?? undefined };
  }
  if (/tavily/.test(signature) && /(api.key|credential|unauthori|invalid.key|authentication)/.test(signature)) {
    return { title: 'Search credentials rejected', body: 'Check configured search provider credentials, then restart the research service.', code: apiError?.code ?? undefined };
  }
  if (/(bedrock|model|anthropic)/.test(signature) && /(api.key|credential|unauthori|invalid.key|authentication)/.test(signature)) {
    return { title: 'Model credentials rejected', body: 'Check configured model provider credentials, then restart the research service.', code: apiError?.code ?? undefined };
  }
  if (/(api.key|credential|unauthori|invalid.key|authentication)/.test(signature)) {
    return { title: 'Provider credentials rejected', body: 'Check configured model and search provider credentials, then restart the research service.', code: apiError?.code ?? undefined };
  }
  if (/(throttl|rate.limit|too.many)/.test(signature) || apiError?.status === 429) {
    return { title: 'Provider is temporarily busy', body: 'Request limit was reached. Wait a moment, then try again.', code: apiError?.code ?? undefined, retry: retryAction };
  }
  if (!apiError || apiError.status >= 500 || apiError.status === 408) {
    return { title: 'Research service unavailable', body: message === 'Failed to fetch' ? 'Could not reach the research service. Check that it is running and try again.' : message, code: apiError?.code ?? undefined, retry: retryAction };
  }
  return { title: fallback, body: message, code: apiError.code ?? undefined, retry: retryAction };
}
