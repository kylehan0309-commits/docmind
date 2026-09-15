import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { api } from '../api'
import { loadJSON, saveJSON } from '../storage'
import {
  ENTITY_COLORS,
  type BuildStatus,
  type GraphCoverage,
  type GraphData,
  type MergeHistoryEntry,
  type SearchHit,
} from '../types'
import { EntityPanel } from './EntityPanel'

interface FGNode {
  id: string
  name: string
  type: string
  x?: number
  y?: number
  fx?: number
  fy?: number
}
interface FGLink {
  id: string
  source: string
  target: string
  label: string
  weight: number
}

const MATCH_LABEL: Record<SearchHit['match_field'], string> = {
  name: 'name',
  type: 'type',
  relationship: 'link',
  mention: 'text',
}

// Node layout persists across reloads/restarts (browser-side, not server
// state) - keyed by entity id so it survives a fresh /graph fetch.
const POSITIONS_KEY = 'docmind:graph-positions'
type Positions = Record<string, { x: number; y: number }>

function saveNodePosition(id: string, x: number, y: number) {
  const positions = loadJSON<Positions>(POSITIONS_KEY, {})
  positions[id] = { x, y }
  saveJSON(POSITIONS_KEY, positions)
}

interface Props {
  /** An entity to focus, handed over from a link in a chat answer. */
  focus: { id: string; nonce: number } | null
  /** Jump to the chat tab and ask this question. */
  onAskInChat: (question: string) => void
}

export function GraphView({ focus, onAskInChat }: Props) {
  const [graph, setGraph] = useState<GraphData>({ nodes: [], edges: [] })
  const [selected, setSelected] = useState<string | null>(null)
  const [status, setStatus] = useState<BuildStatus | null>(null)
  const [building, setBuilding] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [consolidating, setConsolidating] = useState(false)
  const [consolidateMsg, setConsolidateMsg] = useState<string | null>(null)
  const [consolidateWarn, setConsolidateWarn] = useState(false)

  const [mergeHistory, setMergeHistory] = useState<MergeHistoryEntry[]>([])
  const [historyOpen, setHistoryOpen] = useState(false)
  const [undoingId, setUndoingId] = useState<string | null>(null)

  const [coverage, setCoverage] = useState<GraphCoverage[]>([])
  const [pickedDocs, setPickedDocs] = useState<Set<string>>(new Set())
  const [forceRemap, setForceRemap] = useState(false)
  const [pickerOpen, setPickerOpen] = useState(false)

  const [searchInput, setSearchInput] = useState('')
  const [search, setSearch] = useState('')
  const [searchResults, setSearchResults] = useState<SearchHit[]>([])
  const [searchOpen, setSearchOpen] = useState(false)

  const wrapRef = useRef<HTMLDivElement>(null)
  const fgRef = useRef<any>(null)
  const pinnedRef = useRef(false)
  const [size, setSize] = useState({ w: 0, h: 0 })

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      // Ignore 0x0 (the tab is hidden) so the graph keeps its last good size
      // and its laid-out node positions instead of collapsing.
      if (el.clientWidth > 0 && el.clientHeight > 0) {
        setSize({ w: el.clientWidth, h: el.clientHeight })
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const loadGraph = useCallback(async () => {
    try {
      setGraph(await api.getGraph())
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  const loadCoverage = useCallback(async () => {
    try {
      setCoverage(await api.getGraphCoverage())
    } catch {
      /* non-fatal */
    }
  }, [])

  const loadMergeHistory = useCallback(async () => {
    try {
      setMergeHistory(await api.listMerges())
    } catch {
      /* non-fatal */
    }
  }, [])

  useEffect(() => {
    loadGraph()
    loadCoverage()
    loadMergeHistory()
  }, [loadGraph, loadCoverage, loadMergeHistory])

  useEffect(() => {
    if (!building) return
    const t = setInterval(async () => {
      try {
        const s = await api.getBuildStatus()
        setStatus(s)
        if (!s.running) {
          setBuilding(false)
          await loadGraph()
          await loadCoverage()
        }
      } catch (e) {
        setError(String(e))
        setBuilding(false)
      }
    }, 2000)
    return () => clearInterval(t)
  }, [building, loadGraph, loadCoverage])

  async function startBuild() {
    setError(null)
    const documentIds = pickedDocs.size ? [...pickedDocs] : undefined
    if (forceRemap && !documentIds) {
      setError('Pick one or more documents to re-map.')
      return
    }
    try {
      const r = await api.buildGraph({ documentIds, force: forceRemap })
      if (r.status === 'nothing_to_do') {
        setError(
          documentIds
            ? 'Those documents are already fully mapped. Tick “re-map” to redo them.'
            : 'Every chunk is already mapped. Upload a document first.',
        )
        return
      }
      setStatus(null)
      setBuilding(true)
    } catch (e) {
      setError(String(e))
    }
  }

  async function runConsolidate() {
    setError(null)
    setConsolidateMsg(null)
    setConsolidateWarn(false)
    setConsolidating(true)
    try {
      const r = await api.consolidateGraph()
      if (r.groups.length === 0) {
        setConsolidateMsg('No duplicate entities found.')
      } else {
        const pairs = r.groups
          .map((g) => `${g.merged.map((m) => m.name).join(', ')} → ${g.keep_name}`)
          .join('; ')
        const isLocal = r.provider === 'local'
        setConsolidateWarn(isLocal)
        setConsolidateMsg(
          `Merged: ${pairs}` +
            (isLocal ? ' — local model; check below and undo if wrong.' : '. See below to undo.'),
        )
        setHistoryOpen(true)
      }
      await loadGraph()
      await loadCoverage()
      await loadMergeHistory()
    } catch (e) {
      setError(String(e))
    } finally {
      setConsolidating(false)
    }
  }

  async function undo(mergeId: string) {
    setError(null)
    setUndoingId(mergeId)
    try {
      await api.undoMerge(mergeId)
      await loadGraph()
      await loadCoverage()
      await loadMergeHistory()
    } catch (e) {
      setError(String(e))
    } finally {
      setUndoingId(null)
    }
  }

  function toggleDoc(id: string) {
    setPickedDocs((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const pendingTotal = coverage.reduce(
    (n, c) => n + (c.total_chunks - c.extracted_chunks),
    0,
  )

  const data = useMemo(() => {
    const positions = loadJSON<Positions>(POSITIONS_KEY, {})
    return {
      // A node with a saved position starts pinned there (fx/fy fixes it for
      // d3-force); a genuinely new node has none yet and lays out normally,
      // flowing in around the ones already placed.
      nodes: graph.nodes.map((n) => {
        const p = positions[n.id]
        return p ? ({ ...n, x: p.x, y: p.y, fx: p.x, fy: p.y } as FGNode) : ({ ...n } as FGNode)
      }),
      links: graph.edges.map((e) => ({ ...e })) as FGLink[],
    }
  }, [graph])

  // New data -> let the layout run again, then re-pin on the next engine stop.
  useEffect(() => {
    pinnedRef.current = false
  }, [data])

  // Nodes returned by the server-side content search (name / type / relationship
  // label / connected entity / mention text).
  const searchMatches = useMemo(
    () => (search ? new Set(searchResults.map((h) => h.id)) : null),
    [search, searchResults],
  )

  // What to emphasise: the selected node and/or the search matches, plus their
  // immediate neighbourhood. `primary` = the actual hits (ringed); `nodes` =
  // hits + neighbours (not dimmed); `links` = the edges touching a hit.
  const highlight = useMemo(() => {
    const primary = new Set<string>()
    if (selected) primary.add(selected)
    if (searchMatches) for (const id of searchMatches) primary.add(id)
    if (primary.size === 0) return null

    const nodes = new Set(primary)
    const links = new Set<string>()
    for (const e of graph.edges) {
      if (primary.has(e.source) || primary.has(e.target)) {
        nodes.add(e.source)
        nodes.add(e.target)
        links.add(e.id)
      }
    }
    return { primary, nodes, links }
  }, [selected, searchMatches, graph.edges])

  async function runSearch() {
    const q = searchInput.trim()
    if (!q) return clearSearch()
    try {
      const hits = await api.searchGraph(q)
      setSearchResults(hits)
      setSearch(q)
      setSearchOpen(true)
      if (hits.length === 1) focusNode(hits[0].id)
    } catch (e) {
      setError(String(e))
    }
  }
  function clearSearch() {
    setSearchInput('')
    setSearch('')
    setSearchResults([])
    setSearchOpen(false)
  }

  function focusNode(id: string) {
    setSelected(id)
    const n = data.nodes.find((x) => x.id === id)
    if (n && n.x != null && n.y != null && fgRef.current) {
      fgRef.current.centerAt?.(n.x, n.y, 600)
      fgRef.current.zoom?.(2.2, 600)
    }
  }

  function relayout() {
    for (const n of data.nodes) {
      n.fx = undefined
      n.fy = undefined
    }
    pinnedRef.current = false
    fgRef.current?.d3ReheatSimulation?.()
  }

  // An entity clicked in a chat answer: select it, and once the layout has
  // given it coordinates, centre the view on it.
  useEffect(() => {
    if (!focus) return
    clearSearch() // a stale search would dim the node we're focusing
    setSelected(focus.id)
    let tries = 0
    const timer = setInterval(() => {
      const n = data.nodes.find((x) => x.id === focus.id)
      if (n && n.x != null && n.y != null && fgRef.current) {
        fgRef.current.centerAt?.(n.x, n.y, 600)
        fgRef.current.zoom?.(2.2, 600)
        clearInterval(timer)
      } else if (++tries > 25) {
        clearInterval(timer)
      }
    }, 150)
    return () => clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.nonce])

  return (
    <div className="flex h-full">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex flex-wrap items-center gap-3 border-b border-slate-200 bg-white px-4 py-2 text-sm">
          <button
            onClick={() => setPickerOpen((o) => !o)}
            className="rounded-md border border-slate-300 px-2.5 py-1.5 font-medium text-slate-700 hover:bg-slate-50"
          >
            Documents{' '}
            {pickedDocs.size > 0 ? `(${pickedDocs.size})` : `· ${coverage.length}`} {pickerOpen ? '▴' : '▾'}
          </button>

          <label className="flex items-center gap-1.5 text-xs text-slate-500">
            <input
              type="checkbox"
              checked={forceRemap}
              onChange={(e) => setForceRemap(e.target.checked)}
              disabled={pickedDocs.size === 0}
            />
            re-map selected
          </label>

          <button
            onClick={startBuild}
            disabled={building}
            className="rounded-md bg-slate-900 px-3 py-1.5 font-medium text-white disabled:opacity-40"
          >
            {building
              ? 'Building…'
              : pickedDocs.size > 0
                ? `Build ${pickedDocs.size} selected`
                : `Build ${pendingTotal || ''} pending`.trim()}
          </button>

          {/* search */}
          <div className="flex items-center overflow-hidden rounded-md border border-slate-300">
            <input
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') runSearch()
                if (e.key === 'Escape') clearSearch()
              }}
              placeholder="Search entities…"
              className="w-44 px-2 py-1 text-sm outline-none"
            />
            {search && (
              <button
                onClick={clearSearch}
                className="px-1.5 text-slate-400 hover:text-slate-700"
                title="Clear"
              >
                ✕
              </button>
            )}
            <button
              onClick={runSearch}
              className="border-l border-slate-300 bg-slate-50 px-2.5 py-1 font-medium text-slate-600 hover:bg-slate-100"
            >
              Search
            </button>
          </div>
          {search && (
            <button
              onClick={() => setSearchOpen((o) => !o)}
              className="text-xs text-slate-500 hover:text-slate-800"
            >
              {searchResults.length} result{searchResults.length === 1 ? '' : 's'} for “{search}”{' '}
              {searchOpen ? '▴' : '▾'}
            </button>
          )}

          {building && status && (
            <div className="flex items-center gap-2 text-slate-500">
              <div className="h-1.5 w-40 overflow-hidden rounded-full bg-slate-200">
                <div
                  className="h-full bg-slate-700 transition-all"
                  style={{ width: `${status.total ? (status.processed / status.total) * 100 : 0}%` }}
                />
              </div>
              <span>
                {status.processed}/{status.total} chunks · {status.new_entities} entities ·{' '}
                {status.new_relationships} edges
                {status.failed > 0 && (
                  <span className="text-amber-600"> · {status.failed} failed</span>
                )}
              </span>
            </div>
          )}
          {building && !status && <span className="text-slate-400">starting…</span>}

          {!building && status && status.failed > 0 && (
            <span className="text-amber-600">
              {status.failed} chunk{status.failed === 1 ? '' : 's'} failed (transient provider
              error) — click Build again to retry
            </span>
          )}

          {!building && (
            <span className="text-slate-400">
              {graph.nodes.length} entities · {graph.edges.length} relationships
            </span>
          )}

          {graph.nodes.length > 0 && (
            <button
              onClick={relayout}
              className="text-xs text-slate-400 hover:text-slate-700"
              title="Re-run the force layout"
            >
              re-layout
            </button>
          )}

          <button
            onClick={runConsolidate}
            disabled={consolidating}
            title="One LLM pass over every entity to merge acronym/synonym duplicates (e.g. RAG + retrieval-augmented generation) that per-chunk dedup misses"
            className="rounded-md border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40"
          >
            {consolidating ? 'Consolidating…' : 'Consolidate synonyms'}
          </button>
          {consolidateMsg && (
            <span className={`text-xs ${consolidateWarn ? 'text-amber-600' : 'text-slate-500'}`}>
              {consolidateMsg}
            </span>
          )}

          {mergeHistory.length > 0 && (
            <button
              onClick={() => setHistoryOpen((o) => !o)}
              className="text-xs text-slate-400 hover:text-slate-700"
            >
              History ({mergeHistory.length}) {historyOpen ? '▴' : '▾'}
            </button>
          )}

          <div className="ml-auto flex flex-wrap gap-2 text-xs">
            {Object.entries(ENTITY_COLORS).map(([k, c]) => (
              <span key={k} className="flex items-center gap-1 text-slate-500">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: c }} />
                {k}
              </span>
            ))}
          </div>
        </div>

        {search && searchOpen && (
          <div className="max-h-64 overflow-y-auto border-b border-slate-200 bg-white px-4 py-2 text-sm">
            {searchResults.length === 0 ? (
              <p className="text-slate-400">No entities contain “{search}”.</p>
            ) : (
              <>
                <p className="mb-1 text-xs text-slate-400">
                  Click a result to focus it and open its panel
                </p>
                {searchResults.map((h) => (
                  <button
                    key={h.id}
                    onClick={() => focusNode(h.id)}
                    className={`flex w-full items-center gap-2 rounded px-1 py-1 text-left hover:bg-slate-100 ${
                      selected === h.id ? 'bg-slate-100' : ''
                    }`}
                  >
                    <span
                      className="h-2.5 w-2.5 shrink-0 rounded-full"
                      style={{ background: ENTITY_COLORS[h.type] ?? ENTITY_COLORS.OTHER }}
                    />
                    <span className="font-medium text-slate-800">{h.name}</span>
                    <span className="rounded bg-slate-100 px-1.5 text-xs text-slate-500">
                      {MATCH_LABEL[h.match_field]}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-xs text-slate-500">
                      {h.snippet}
                    </span>
                  </button>
                ))}
              </>
            )}
          </div>
        )}

        {historyOpen && (
          <div className="max-h-64 overflow-y-auto border-b border-slate-200 bg-white px-4 py-2 text-sm">
            {mergeHistory.length === 0 ? (
              <p className="text-slate-400">No merges yet.</p>
            ) : (
              mergeHistory.map((h) => (
                <div
                  key={h.id}
                  className={`flex items-center gap-2 py-1 ${h.undone ? 'opacity-50' : ''}`}
                >
                  <span className="min-w-0 flex-1 truncate">
                    <span className="text-slate-500">
                      {h.merged.map((m) => m.name).join(', ')}
                    </span>{' '}
                    <span className="text-slate-400">→</span>{' '}
                    <span className="font-medium text-slate-800">{h.keep_name}</span>
                  </span>
                  <span className="shrink-0 text-xs text-slate-400">
                    {new Date(h.created_at).toLocaleTimeString()}
                  </span>
                  {h.undone ? (
                    <span className="shrink-0 text-xs text-slate-400">undone</span>
                  ) : (
                    <button
                      onClick={() => undo(h.id)}
                      disabled={undoingId === h.id}
                      className="shrink-0 rounded border border-slate-300 px-2 py-0.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40"
                    >
                      {undoingId === h.id ? 'Undoing…' : 'Undo'}
                    </button>
                  )}
                </div>
              ))
            )}
          </div>
        )}

        {pickerOpen && (
          <div className="max-h-56 overflow-y-auto border-b border-slate-200 bg-white px-4 py-2 text-sm">
            {coverage.length === 0 ? (
              <p className="text-slate-400">No documents uploaded.</p>
            ) : (
              <>
                <div className="mb-1 flex gap-3 text-xs text-slate-400">
                  <button
                    className="hover:text-slate-700"
                    onClick={() => setPickedDocs(new Set(coverage.map((c) => c.document_id)))}
                  >
                    select all
                  </button>
                  <button className="hover:text-slate-700" onClick={() => setPickedDocs(new Set())}>
                    clear
                  </button>
                </div>
                {coverage.map((c) => {
                  const done = c.total_chunks > 0 && c.extracted_chunks === c.total_chunks
                  return (
                    <label
                      key={c.document_id}
                      className="flex items-center gap-2 py-0.5 text-slate-700"
                    >
                      <input
                        type="checkbox"
                        checked={pickedDocs.has(c.document_id)}
                        onChange={() => toggleDoc(c.document_id)}
                      />
                      <span className="flex-1 truncate">{c.filename}</span>
                      <span
                        className={`rounded-full px-2 py-0.5 text-xs ${
                          done ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'
                        }`}
                      >
                        {c.extracted_chunks}/{c.total_chunks} mapped
                      </span>
                    </label>
                  )
                })}
              </>
            )}
          </div>
        )}

        {status?.error && (
          <div className="border-b border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
            Build failed: {status.error}
          </div>
        )}
        {error && (
          <div className="border-b border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        <div ref={wrapRef} className="relative min-h-0 flex-1 bg-slate-50">
          {graph.nodes.length === 0 ? (
            <div className="absolute inset-0 grid place-items-center text-sm text-slate-400">
              No graph yet — upload documents, then “Build graph”.
            </div>
          ) : (
            <ForceGraph2D
              ref={fgRef}
              width={size.w}
              height={size.h}
              graphData={data}
              cooldownTime={4000}
              nodeRelSize={4}
              nodeLabel={(n) => (n as FGNode).name}
              linkLabel={(l) => (l as FGLink).label}
              linkColor={(l) =>
                highlight && !highlight.links.has((l as FGLink).id) ? '#e2e8f0' : '#94a3b8'
              }
              linkWidth={(l) => (highlight && highlight.links.has((l as FGLink).id) ? 2 : 1)}
              linkDirectionalArrowLength={3}
              linkDirectionalArrowRelPos={1}
              onBackgroundClick={() => setSelected(null)}
              onNodeClick={(n) => setSelected((n as FGNode).id)}
              onEngineStop={() => {
                if (pinnedRef.current) return
                pinnedRef.current = true
                const positions = loadJSON<Positions>(POSITIONS_KEY, {})
                for (const n of data.nodes) {
                  n.fx = n.x
                  n.fy = n.y
                  if (n.x != null && n.y != null) positions[n.id] = { x: n.x, y: n.y }
                }
                saveJSON(POSITIONS_KEY, positions)
              }}
              onNodeDragEnd={(n) => {
                const fn = n as FGNode
                fn.fx = fn.x
                fn.fy = fn.y
                if (fn.x != null && fn.y != null) saveNodePosition(fn.id, fn.x, fn.y)
              }}
              nodeCanvasObjectMode={() => 'replace'}
              nodeCanvasObject={(node, ctx, scale) => {
                const n = node as FGNode
                const dim = highlight && !highlight.nodes.has(n.id)
                const isHit = highlight?.primary.has(n.id) ?? false
                const r = 4
                ctx.globalAlpha = dim ? 0.12 : 1
                ctx.beginPath()
                ctx.arc(n.x!, n.y!, isHit ? r + 1.5 : r, 0, 2 * Math.PI)
                ctx.fillStyle = ENTITY_COLORS[n.type] ?? ENTITY_COLORS.OTHER
                ctx.fill()
                if (isHit) {
                  ctx.lineWidth = 2 / scale
                  ctx.strokeStyle = '#0f172a'
                  ctx.stroke()
                }
                const fontSize = Math.max(10 / scale, 2)
                ctx.font = `${fontSize}px system-ui, sans-serif`
                ctx.fillStyle = isHit ? '#0f172a' : '#334155'
                ctx.textBaseline = 'middle'
                ctx.fillText(n.name, n.x! + r + 2, n.y!)
                ctx.globalAlpha = 1
              }}
            />
          )}
        </div>
      </div>

      {selected && (
        <EntityPanel
          entityId={selected}
          onClose={() => setSelected(null)}
          onSelectEntity={(id) => setSelected(id)}
          onAsk={(name) => onAskInChat(`What does the knowledge base say about ${name}?`)}
        />
      )}
    </div>
  )
}
