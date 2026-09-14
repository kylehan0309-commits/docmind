import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { api } from '../api'
import type { ChatResponse } from '../types'

interface Props {
  /** A question pushed in from the graph tab ("Ask in chat" on an entity).
   * `nonce` changes each time so the same question re-fires. */
  seed: { question: string; nonce: number } | null
  /** Jump to the graph tab with this entity selected. */
  onShowEntity: (entityId: string) => void
}

// Matches a citation marker the model emits: [1], [2, 4], [1,3] ...
const CITE_RE = /\[(\d+(?:\s*,\s*\d+)*)\]/g

const RE_SPECIAL = /[.*+?^${}()|[\]\\]/g

interface EntityMatcher {
  re: RegExp
  byLower: Map<string, string>
}

/** Turn plain answer text into nodes, linking the first occurrence of each
 * known graph entity to `onEntity`. */
function linkEntities(
  seg: string,
  matcher: EntityMatcher | null,
  seen: Set<string>,
  onEntity: (id: string) => void,
  keyBase: string,
): ReactNode[] {
  if (!matcher) return [seg]
  const out: ReactNode[] = []
  let last = 0
  let k = 0
  matcher.re.lastIndex = 0
  let m: RegExpExecArray | null
  while ((m = matcher.re.exec(seg)) !== null) {
    const id = matcher.byLower.get(m[0].toLowerCase())
    if (!id || seen.has(id)) continue // unknown, or already linked earlier
    seen.add(id)
    if (m.index > last) out.push(seg.slice(last, m.index))
    out.push(
      <button
        key={`${keyBase}e${k++}`}
        type="button"
        onClick={() => onEntity(id)}
        title="Show in knowledge graph"
        className="underline decoration-slate-300 decoration-1 underline-offset-2 hover:decoration-slate-600"
      >
        {m[0]}
      </button>,
    )
    last = m.index + m[0].length
  }
  if (last < seg.length) out.push(seg.slice(last))
  return out
}

/**
 * Render the answer: `[n]` markers become buttons that jump to the source card,
 * and known entity names become buttons that jump to the graph. Out-of-range
 * markers (the model occasionally invents one) stay as plain text.
 */
function renderAnswer(
  text: string,
  citationCount: number,
  matcher: EntityMatcher | null,
  onJump: (n: number) => void,
  onEntity: (id: string) => void,
): ReactNode[] {
  const out: ReactNode[] = []
  const seen = new Set<string>()
  let last = 0
  let key = 0
  CITE_RE.lastIndex = 0
  let m: RegExpExecArray | null
  while ((m = CITE_RE.exec(text)) !== null) {
    const nums = m[1].split(',').map((s) => Number(s.trim()))
    if (!nums.some((n) => n >= 1 && n <= citationCount)) continue // e.g. "[9]" with 3 sources
    if (m.index > last) {
      out.push(...linkEntities(text.slice(last, m.index), matcher, seen, onEntity, `k${key}`))
    }
    out.push(
      <span key={`c${key++}`} className="whitespace-nowrap">
        [
        {nums.map((n, j) => {
          const ok = n >= 1 && n <= citationCount
          return (
            <span key={j}>
              {j > 0 && ', '}
              {ok ? (
                <button
                  type="button"
                  onClick={() => onJump(n)}
                  title={`Jump to source [${n}]`}
                  className="align-baseline font-medium text-slate-600 underline decoration-dotted underline-offset-2 hover:text-slate-900 hover:decoration-solid"
                >
                  {n}
                </button>
              ) : (
                n
              )}
            </span>
          )
        })}
        ]
      </span>,
    )
    last = m.index + m[0].length
  }
  if (last < text.length) {
    out.push(...linkEntities(text.slice(last), matcher, seen, onEntity, 'kt'))
  }
  return out
}

export function ChatView({ seed, onShowEntity }: Props) {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<ChatResponse | null>(null)
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [flash, setFlash] = useState<number | null>(null)
  const [revealed, setRevealed] = useState<Set<number>>(new Set())
  const [entities, setEntities] = useState<{ id: string; name: string }[]>([])
  const sourceRefs = useRef<(HTMLLIElement | null)[]>([])
  const flashTimer = useRef<number | undefined>(undefined)

  const loadEntities = useCallback(async () => {
    try {
      const g = await api.getGraph()
      setEntities(g.nodes.map((n) => ({ id: n.id, name: n.name })))
    } catch {
      /* graph is optional; no links if it isn't built */
    }
  }, [])

  useEffect(() => {
    loadEntities()
  }, [loadEntities])

  // Build one regex over every entity name >= 3 chars, longest first so the most
  // specific name wins at a given position.
  const matcher = useMemo<EntityMatcher | null>(() => {
    const usable = entities
      .filter((e) => e.name.trim().length >= 3)
      .sort((a, b) => b.name.length - a.name.length)
      .slice(0, 500)
    if (usable.length === 0) return null
    const byLower = new Map<string, string>()
    for (const e of usable) byLower.set(e.name.toLowerCase(), e.id)
    const alt = usable.map((e) => e.name.replace(RE_SPECIAL, '\\$&')).join('|')
    return {
      re: new RegExp(`(?<![A-Za-z0-9])(?:${alt})(?![A-Za-z0-9])`, 'gi'),
      byLower,
    }
  }, [entities])

  const ask = useCallback(
    async (override?: string) => {
      const q = (override ?? question).trim()
      if (!q || streaming) return
      setQuestion(q)
      setStreaming(true)
      setError(null)
      setAnswer({ answer: '', citations: [] })
      setFlash(null)
      setRevealed(new Set())
      sourceRefs.current = []
      loadEntities() // refresh in case the graph changed since mount

      try {
        for await (const ev of api.streamChat(q)) {
          if (ev.type === 'citations') {
            setAnswer((prev) => ({ answer: prev?.answer ?? '', citations: ev.citations }))
          } else if (ev.type === 'token') {
            setAnswer((prev) => ({
              answer: (prev?.answer ?? '') + ev.text,
              citations: prev?.citations ?? [],
            }))
          } else if (ev.type === 'error') {
            setError(ev.detail)
          }
        }
      } catch (e) {
        setError(String(e))
      } finally {
        setStreaming(false)
      }
    },
    [question, streaming, loadEntities],
  )

  // A question handed over from the graph tab.
  useEffect(() => {
    if (seed) ask(seed.question)
    // only re-run when a new seed arrives, not when `ask` identity changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed?.nonce])

  function jumpToSource(n: number) {
    const el = sourceRefs.current[n - 1]
    if (!el) return
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setRevealed((prev) => new Set(prev).add(n))
    setFlash(n)
    window.clearTimeout(flashTimer.current)
    flashTimer.current = window.setTimeout(() => setFlash(null), 1500)
  }

  function toggleReveal(n: number) {
    setRevealed((prev) => {
      const next = new Set(prev)
      if (next.has(n)) next.delete(n)
      else next.add(n)
      return next
    })
  }

  const hasAnswerText = (answer?.answer.length ?? 0) > 0
  const citationCount = answer?.citations.length ?? 0

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col gap-4 overflow-y-auto p-6">
      <div className="flex gap-2">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask()
          }}
          rows={2}
          placeholder="Ask something about your uploaded documents…  (⌘/Ctrl+Enter)"
          className="flex-1 resize-none rounded-md border border-slate-300 bg-white px-3 py-2 text-sm outline-none focus:border-slate-500"
        />
        <button
          onClick={() => ask()}
          disabled={streaming || !question.trim()}
          className="self-stretch rounded-md bg-slate-900 px-4 text-sm font-medium text-white disabled:opacity-40"
        >
          {streaming ? '…' : 'Ask'}
        </button>
      </div>

      {streaming && !hasAnswerText && (
        <p className="text-sm text-slate-500">
          Thinking… the first question after a server start takes ~10–20s while the model loads.
        </p>
      )}
      {error && (
        <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      )}

      {answer && (hasAnswerText || answer.citations.length > 0) && (
        <div className="space-y-4">
          <div className="whitespace-pre-wrap rounded-lg border border-slate-200 bg-white p-4 text-sm leading-relaxed">
            {renderAnswer(answer.answer, citationCount, matcher, jumpToSource, onShowEntity)}
            {streaming && hasAnswerText && (
              <span className="ml-0.5 inline-block w-1.5 animate-pulse bg-slate-400 align-middle">
                &nbsp;
              </span>
            )}
          </div>

          {answer.citations.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                Sources
              </p>
              <ol className="space-y-2">
                {answer.citations.map((c, i) => {
                  const n = i + 1
                  const isRevealed = revealed.has(n)
                  return (
                    <li
                      key={c.chunk_index}
                      ref={(el) => {
                        sourceRefs.current[i] = el
                      }}
                      onClick={() => toggleReveal(n)}
                      className={`cursor-pointer rounded-md border bg-white p-3 text-sm transition-colors ${
                        flash === n
                          ? 'border-slate-400 ring-2 ring-slate-900/50'
                          : 'border-slate-200 hover:border-slate-300'
                      }`}
                    >
                      <div className="mb-1 flex items-center gap-2 text-xs text-slate-500">
                        <span className="rounded bg-slate-100 px-1.5 py-0.5 font-medium">[{n}]</span>
                        <span className="font-medium text-slate-700">{c.filename}</span>
                        <span>· page {c.page_number}</span>
                        <span>· score {c.score.toFixed(3)}</span>
                      </div>
                      <p className={isRevealed ? 'whitespace-pre-wrap text-slate-600' : 'line-clamp-4 text-slate-600'}>
                        {c.text}
                      </p>
                    </li>
                  )
                })}
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
