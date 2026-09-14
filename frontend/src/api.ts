import type {
  BuildStatus,
  ChatResponse,
  ChatStreamEvent,
  DocumentSummary,
  EntityDetail,
  GraphCoverage,
  GraphData,
  MergeHistoryEntry,
  Provider,
  SearchHit,
  Settings,
} from './types'

const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* body wasn't JSON */
    }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json() as Promise<T>
}

/** Parses one `event: ...\ndata: ...` SSE frame into a typed ChatStreamEvent. */
function parseSseFrame(frame: string): ChatStreamEvent | null {
  const lines = frame.split('\n')
  const eventLine = lines.find((l) => l.startsWith('event: '))
  const dataLine = lines.find((l) => l.startsWith('data: '))
  if (!eventLine || !dataLine) return null
  const event = eventLine.slice('event: '.length).trim()
  const data = JSON.parse(dataLine.slice('data: '.length))

  switch (event) {
    case 'citations':
      return { type: 'citations', citations: data.citations }
    case 'token':
      return { type: 'token', text: data.text }
    case 'done':
      return { type: 'done' }
    case 'error':
      return { type: 'error', detail: data.detail }
    default:
      return null
  }
}

async function* streamChatEvents(
  question: string,
  documentIds?: string[],
): AsyncGenerator<ChatStreamEvent> {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, document_ids: documentIds ?? null }),
  })
  if (!res.ok || !res.body) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* body wasn't JSON */
    }
    throw new Error(`${res.status}: ${detail}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let sep: number
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const parsed = parseSseFrame(frame)
      if (parsed) yield parsed
    }
  }
}

export const api = {
  listDocuments: () => request<DocumentSummary[]>('/documents/'),

  uploadDocument: (file: File, describeFigures = false) => {
    const form = new FormData()
    form.append('file', file)
    const query = describeFigures ? '?describe_figures=true' : ''
    return request<DocumentSummary>(`/documents/upload${query}`, { method: 'POST', body: form })
  },

  deleteDocument: (id: string) =>
    request<{ deleted: string }>(`/documents/${id}`, { method: 'DELETE' }),

  chat: (question: string, documentIds?: string[]) =>
    request<ChatResponse>('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, document_ids: documentIds ?? null }),
    }),

  /** Streams the answer as it's generated. Yields `citations` once (as soon as
   * retrieval finishes), then one `token` per text delta, then `done` (or
   * `error` if the provider call fails partway through). */
  streamChat: (question: string, documentIds?: string[]): AsyncGenerator<ChatStreamEvent> =>
    streamChatEvents(question, documentIds),

  getGraph: () => request<GraphData>('/graph'),

  getGraphCoverage: () => request<GraphCoverage[]>('/graph/coverage'),

  searchGraph: (q: string) =>
    request<SearchHit[]>(`/graph/search?q=${encodeURIComponent(q)}`),

  buildGraph: (opts: { documentIds?: string[]; force?: boolean } = {}) => {
    const params = new URLSearchParams()
    for (const id of opts.documentIds ?? []) params.append('document_ids', id)
    if (opts.force) params.set('force', 'true')
    const qs = params.toString()
    return request<{ status: string; total_chunks: number }>(
      `/graph/build${qs ? `?${qs}` : ''}`,
      { method: 'POST' },
    )
  },

  getBuildStatus: () => request<BuildStatus>('/graph/build/status'),

  getEntity: (id: string) => request<EntityDetail>(`/graph/entities/${id}`),

  getSettings: () => request<Settings>('/settings'),

  setProvider: (provider: Provider) =>
    request<{ provider: string }>('/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider }),
    }),

  mergeEntities: (keepId: string, mergeIds: string[]) =>
    request<{ kept: string; merged: string[]; removed_duplicate_edges: number; merge_id: string }>(
      '/graph/entities/merge',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keep_id: keepId, merge_ids: mergeIds }),
      },
    ),

  /** One LLM pass over every entity to merge acronym/synonym duplicates
   * (e.g. "RAG" + "retrieval-augmented generation") that per-chunk dedup misses. */
  consolidateGraph: () =>
    request<{
      provider: Provider
      groups: { keep_id: string; keep_name: string; merged: { id: string; name: string }[]; merge_id: string }[]
      removed_duplicate_edges: number
    }>('/graph/consolidate', { method: 'POST' }),

  listMerges: () => request<MergeHistoryEntry[]>('/graph/merges'),

  /** Reverses one merge: re-inserts the deleted entity/entities and restores
   * every relationship/mention it touched. Fails if already undone, or if the
   * entity it was merged into has itself since been merged away. */
  undoMerge: (mergeId: string) =>
    request<{ undone: string; restored_entities: number }>(
      `/graph/merges/${mergeId}/undo`,
      { method: 'POST' },
    ),
}
