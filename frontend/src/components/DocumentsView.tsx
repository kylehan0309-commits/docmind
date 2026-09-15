import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { DocumentSummary } from '../types'

const STATUS_STYLE: Record<string, string> = {
  ready: 'bg-green-100 text-green-700',
  processing: 'bg-amber-100 text-amber-700',
  failed: 'bg-red-100 text-red-700',
}

export function DocumentsView() {
  const [docs, setDocs] = useState<DocumentSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [describeFigures, setDescribeFigures] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [filter, setFilter] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)
  const dragDepth = useRef(0)

  async function refresh() {
    try {
      setDocs(await api.listDocuments())
      setError(null)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  // Parsing/chunking/embedding runs in the background now (POST /upload
  // returns as soon as the file is saved), so poll for the processing ->
  // ready/failed transition instead of waiting on the request.
  const anyProcessing = docs.some((d) => d.status === 'processing')
  useEffect(() => {
    if (!anyProcessing) return
    const t = setInterval(refresh, 1500)
    return () => clearInterval(t)
  }, [anyProcessing])

  async function onUploadFiles(files: FileList | File[]) {
    const all = Array.from(files)
    const pdfs = all.filter((f) => f.name.toLowerCase().endsWith('.pdf'))
    if (pdfs.length === 0) {
      setError('Only PDF files are supported.')
      return
    }
    setUploading(true)
    setError(null)
    const failures: string[] = []
    for (const file of pdfs) {
      try {
        await api.uploadDocument(file, describeFigures)
      } catch (e) {
        failures.push(`${file.name}: ${String(e)}`)
      }
    }
    await refresh()
    setUploading(false)
    if (fileInput.current) fileInput.current.value = ''
    if (failures.length) setError(failures.join('; '))
    else if (all.length > pdfs.length) setError('Skipped non-PDF file(s) - only PDFs are supported.')
  }

  async function onDelete(id: string) {
    try {
      await api.deleteDocument(id)
      await refresh()
    } catch (e) {
      setError(String(e))
    }
  }

  // Drag events fire on children too (bubbling), which naively causes the
  // highlight to flicker on/off as the pointer crosses child elements - a
  // depth counter (enter increments, leave decrements) keeps it stable.
  function onDragEnter(e: React.DragEvent) {
    e.preventDefault()
    dragDepth.current += 1
    setDragging(true)
  }
  function onDragLeave(e: React.DragEvent) {
    e.preventDefault()
    dragDepth.current -= 1
    if (dragDepth.current <= 0) setDragging(false)
  }
  function onDragOver(e: React.DragEvent) {
    e.preventDefault() // required for onDrop to fire at all
  }
  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    if (!uploading && e.dataTransfer.files.length) onUploadFiles(e.dataTransfer.files)
  }

  const filteredDocs = filter.trim()
    ? docs.filter((d) => d.filename.toLowerCase().includes(filter.trim().toLowerCase()))
    : docs

  return (
    <div className="mx-auto max-w-3xl space-y-6 overflow-y-auto p-6">
      <div
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDragOver={onDragOver}
        onDrop={onDrop}
        className={`rounded-lg border border-dashed p-6 text-center transition-colors ${
          dragging ? 'border-slate-500 bg-slate-50' : 'border-slate-300 bg-white'
        }`}
      >
        <p className="mb-3 text-sm text-slate-600">
          Drag &amp; drop PDFs here, or choose one to parse, chunk and embed it.
        </p>
        <input
          ref={fileInput}
          type="file"
          accept="application/pdf"
          multiple
          disabled={uploading}
          onChange={(e) => e.target.files && onUploadFiles(e.target.files)}
          className="mx-auto block text-sm file:mr-3 file:rounded-md file:border-0 file:bg-slate-900 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-white hover:file:bg-slate-700"
        />
        <label className="mt-3 flex items-center justify-center gap-2 text-xs text-slate-500">
          <input
            type="checkbox"
            checked={describeFigures}
            disabled={uploading}
            onChange={(e) => setDescribeFigures(e.target.checked)}
          />
          Describe charts &amp; diagrams with the vision model (slow — one call per figure page)
        </label>
        {uploading && <p className="mt-3 text-sm text-slate-500">Uploading…</p>}
      </div>

      {error && (
        <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      )}

      {loading ? (
        <p className="text-sm text-slate-500">Loading…</p>
      ) : docs.length === 0 ? (
        <p className="text-sm text-slate-500">No documents yet.</p>
      ) : (
        <>
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder={`Filter ${docs.length} document${docs.length === 1 ? '' : 's'} by filename…`}
            className="w-full rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm outline-none focus:border-slate-500"
          />
          {filteredDocs.length === 0 ? (
            <p className="text-sm text-slate-400">No documents match “{filter}”.</p>
          ) : (
            <table className="w-full border-separate border-spacing-y-1 text-sm">
              <thead className="text-left text-xs uppercase tracking-wide text-slate-400">
                <tr>
                  <th className="px-3 py-1">File</th>
                  <th className="px-3 py-1">Pages</th>
                  <th className="px-3 py-1">Status</th>
                  <th className="px-3 py-1"></th>
                </tr>
              </thead>
              <tbody>
                {filteredDocs.map((d) => (
                  <tr key={d.id} className="bg-white">
                    <td className="rounded-l-md px-3 py-2 font-medium">{d.filename}</td>
                    <td className="px-3 py-2 text-slate-500">{d.page_count || '—'}</td>
                    <td className="px-3 py-2">
                      <span
                        title={d.error ?? undefined}
                        className={`rounded-full px-2 py-0.5 text-xs ${
                          STATUS_STYLE[d.status] ?? 'bg-slate-100 text-slate-600'
                        } ${d.status === 'processing' ? 'animate-pulse' : ''}`}
                      >
                        {d.status}
                      </span>
                      {d.status === 'failed' && d.error && (
                        <span className="ml-2 text-xs text-red-600">{d.error}</span>
                      )}
                    </td>
                    <td className="rounded-r-md px-3 py-2 text-right">
                      <button
                        onClick={() => onDelete(d.id)}
                        className="text-xs font-medium text-red-600 hover:underline"
                      >
                        delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  )
}
