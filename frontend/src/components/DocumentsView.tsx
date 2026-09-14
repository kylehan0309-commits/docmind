import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { DocumentSummary } from '../types'

export function DocumentsView() {
  const [docs, setDocs] = useState<DocumentSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [describeFigures, setDescribeFigures] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

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

  async function onUpload(file: File) {
    setUploading(true)
    setError(null)
    try {
      await api.uploadDocument(file, describeFigures)
      await refresh()
    } catch (e) {
      setError(String(e))
    } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  async function onDelete(id: string) {
    try {
      await api.deleteDocument(id)
      await refresh()
    } catch (e) {
      setError(String(e))
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 overflow-y-auto p-6">
      <div className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-center">
        <p className="mb-3 text-sm text-slate-600">Upload a PDF to parse, chunk and embed it.</p>
        <input
          ref={fileInput}
          type="file"
          accept="application/pdf"
          disabled={uploading}
          onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])}
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
        {uploading && (
          <p className="mt-3 text-sm text-slate-500">
            {describeFigures
              ? 'Uploading, describing figures, embedding… this can take a few minutes.'
              : 'Uploading and processing…'}
          </p>
        )}
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
            {docs.map((d) => (
              <tr key={d.id} className="bg-white">
                <td className="rounded-l-md px-3 py-2 font-medium">{d.filename}</td>
                <td className="px-3 py-2 text-slate-500">{d.page_count}</td>
                <td className="px-3 py-2">
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
                    {d.status}
                  </span>
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
    </div>
  )
}
