import { useEffect, useState, type ReactNode } from 'react'
import { api } from '../api'
import { ENTITY_COLORS, type EntityDetail } from '../types'

interface Props {
  entityId: string
  onClose: () => void
  onSelectEntity: (id: string) => void
  /** Optional: ask the chat tab about this entity. */
  onAsk?: (name: string) => void
}

export function EntityPanel({ entityId, onClose, onSelectEntity, onAsk }: Props) {
  const [detail, setDetail] = useState<EntityDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setDetail(null)
    setError(null)
    api
      .getEntity(entityId)
      .then((d) => !cancelled && setDetail(d))
      .catch((e) => !cancelled && setError(String(e)))
    return () => {
      cancelled = true
    }
  }, [entityId])

  const outgoing = detail?.relationships.filter((r) => r.direction === 'out') ?? []
  const incoming = detail?.relationships.filter((r) => r.direction === 'in') ?? []

  return (
    <aside className="flex h-full w-96 flex-col border-l border-slate-200 bg-white">
      <div className="flex items-start justify-between border-b border-slate-200 p-4">
        <div>
          {detail ? (
            <>
              <div className="flex items-center gap-2">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ background: ENTITY_COLORS[detail.type] ?? ENTITY_COLORS.OTHER }}
                />
                <h2 className="text-base font-semibold">{detail.name}</h2>
              </div>
              <p className="mt-0.5 text-xs text-slate-500">
                {detail.type} · {detail.degree} connection{detail.degree === 1 ? '' : 's'}
              </p>
              {onAsk && (
                <button
                  onClick={() => onAsk(detail.name)}
                  className="mt-2 rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
                >
                  Ask in chat ↗
                </button>
              )}
            </>
          ) : (
            <h2 className="text-base font-semibold text-slate-400">Loading…</h2>
          )}
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-slate-700" aria-label="Close">
          ✕
        </button>
      </div>

      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-4 text-sm">
        {error && <p className="text-red-600">{error}</p>}

        {outgoing.length > 0 && (
          <Section title={`Outgoing (${outgoing.length})`}>
            {outgoing.map((r, i) => (
              <RelRow
                key={i}
                arrow="→"
                label={r.label}
                other={r.other_name}
                onOther={() => onSelectEntity(r.other_id)}
                source={`${r.filename}${r.page_number ? ` · p.${r.page_number}` : ''}`}
                excerpt={r.excerpt}
              />
            ))}
          </Section>
        )}

        {incoming.length > 0 && (
          <Section title={`Incoming (${incoming.length})`}>
            {incoming.map((r, i) => (
              <RelRow
                key={i}
                arrow="←"
                label={r.label}
                other={r.other_name}
                onOther={() => onSelectEntity(r.other_id)}
                source={`${r.filename}${r.page_number ? ` · p.${r.page_number}` : ''}`}
                excerpt={r.excerpt}
              />
            ))}
          </Section>
        )}

        {detail && detail.mentions.length > 0 && (
          <Section title={`Mentioned in ${detail.mentions.length} chunk${detail.mentions.length === 1 ? '' : 's'}`}>
            {detail.mentions.map((m, i) => (
              <div key={i} className="rounded-md border border-slate-200 p-2">
                <p className="text-xs font-medium text-slate-600">
                  {m.filename}
                  {m.page_number ? ` · p.${m.page_number}` : ''}
                </p>
                <p className="mt-1 text-xs leading-relaxed text-slate-500">{m.excerpt}</p>
              </div>
            ))}
          </Section>
        )}

        {detail &&
          outgoing.length === 0 &&
          incoming.length === 0 &&
          detail.mentions.length === 0 && (
            <p className="text-slate-400">No relationships or mentions recorded.</p>
          )}
      </div>
    </aside>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</p>
      <div className="space-y-2">{children}</div>
    </div>
  )
}

function RelRow({
  arrow,
  label,
  other,
  onOther,
  source,
  excerpt,
}: {
  arrow: string
  label: string
  other: string
  onOther: () => void
  source: string
  excerpt: string
}) {
  return (
    <div className="rounded-md border border-slate-200 p-2">
      <p className="text-slate-700">
        <span className="text-slate-400">{arrow}</span>{' '}
        <span className="font-medium">{label}</span>{' '}
        <button onClick={onOther} className="text-blue-600 hover:underline">
          {other}
        </button>
      </p>
      <p className="mt-1 text-xs text-slate-400">{source}</p>
      {excerpt && <p className="mt-1 text-xs leading-relaxed text-slate-500">{excerpt}</p>}
    </div>
  )
}
