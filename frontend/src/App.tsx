import { useEffect, useRef, useState } from 'react'
import { api } from './api'
import { ChatView } from './components/ChatView'
import { DocumentsView } from './components/DocumentsView'
import { GraphView } from './components/GraphView'
import type { Provider, Settings } from './types'

type Tab = 'documents' | 'chat' | 'graph'

const TABS: { id: Tab; label: string }[] = [
  { id: 'documents', label: 'Documents' },
  { id: 'chat', label: 'Chat' },
  { id: 'graph', label: 'Knowledge graph' },
]

const PROVIDERS: { id: Provider; label: string }[] = [
  { id: 'local', label: 'Local' },
  { id: 'anthropic', label: 'Claude' },
  { id: 'gemini', label: 'Gemini' },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('documents')
  const [settings, setSettings] = useState<Settings | null>(null)

  // Cross-tab handoffs. Each carries a bumping `nonce` so repeating the same
  // value still re-triggers the target tab's effect.
  const [chatSeed, setChatSeed] = useState<{ question: string; nonce: number } | null>(null)
  const [graphFocus, setGraphFocus] = useState<{ id: string; nonce: number } | null>(null)
  const nonce = useRef(0)

  function askInChat(question: string) {
    nonce.current += 1
    setChatSeed({ question, nonce: nonce.current })
    setTab('chat')
  }

  function showInGraph(entityId: string) {
    nonce.current += 1
    setGraphFocus({ id: entityId, nonce: nonce.current })
    setTab('graph')
  }

  useEffect(() => {
    api.getSettings().then(setSettings).catch(() => setSettings(null))
  }, [])

  async function switchProvider(provider: Provider) {
    if (!settings || settings.provider === provider || !settings.available[provider]) return
    try {
      await api.setProvider(provider)
      setSettings({ ...settings, provider })
    } catch {
      /* keep current */
    }
  }

  function providerTitle(p: Provider): string {
    if (!settings) return ''
    if (settings.available[p]) {
      if (p === 'anthropic') return `Claude (${settings.models.anthropic})`
      if (p === 'gemini') return `Gemini (${settings.models.gemini})`
      return `Local models via Ollama`
    }
    if (p === 'anthropic') return 'Set ANTHROPIC_API_KEY in backend/.env to enable'
    if (p === 'gemini') return 'Set GEMINI_API_KEY in backend/.env to enable (free tier)'
    return ''
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-6 border-b border-slate-200 bg-white px-6 py-3">
        <span className="text-lg font-semibold tracking-tight">DocMind</span>
        <nav className="flex gap-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
                tab === t.id ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {settings && (
          <div className="ml-auto flex items-center gap-2 text-xs">
            <span className="text-slate-400">Model</span>
            <div className="flex overflow-hidden rounded-md border border-slate-300">
              {PROVIDERS.map((p, i) => {
                const enabled = settings.available[p.id]
                const active = settings.provider === p.id
                return (
                  <button
                    key={p.id}
                    onClick={() => switchProvider(p.id)}
                    disabled={!enabled}
                    title={providerTitle(p.id)}
                    className={`px-2.5 py-1 font-medium ${i > 0 ? 'border-l border-slate-300' : ''} ${
                      active ? 'bg-slate-900 text-white' : 'bg-white text-slate-600 hover:bg-slate-100'
                    } ${enabled ? '' : 'cursor-not-allowed opacity-40'}`}
                  >
                    {p.label}
                  </button>
                )
              })}
            </div>
          </div>
        )}
      </header>

      {/* All three stay mounted so state (e.g. dragged graph node positions,
          zoom, an in-progress chat) survives switching tabs. */}
      <main className="relative min-h-0 flex-1 overflow-hidden">
        <div className={tab === 'documents' ? 'h-full' : 'hidden'}>
          <DocumentsView />
        </div>
        <div className={tab === 'chat' ? 'h-full' : 'hidden'}>
          <ChatView seed={chatSeed} onShowEntity={showInGraph} />
        </div>
        <div className={tab === 'graph' ? 'h-full' : 'hidden'}>
          <GraphView focus={graphFocus} onAskInChat={askInChat} />
        </div>
      </main>
    </div>
  )
}
