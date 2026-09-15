/**
 * Thin localStorage wrapper. Every call is wrapped in try/catch - localStorage
 * can throw (private browsing, blocked site data, quota exceeded) - so a
 * failure here degrades to "nothing persisted this time", never crashes
 * the app or loses in-memory state.
 */
export function loadJSON<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch {
    return fallback
  }
}

export function saveJSON(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    /* storage full or blocked - just don't persist this time */
  }
}

export function removeKey(key: string): void {
  try {
    localStorage.removeItem(key)
  } catch {
    /* ignore */
  }
}
