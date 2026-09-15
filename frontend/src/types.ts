export interface DocumentSummary {
  id: string
  filename: string
  status: string
  page_count: number
  error: string | null
}

export interface Citation {
  document_id: string
  filename: string
  page_number: number
  chunk_index: number
  score: number
  text: string
}

export interface ChatResponse {
  answer: string
  citations: Citation[]
}

export type ChatStreamEvent =
  | { type: 'citations'; citations: Citation[] }
  | { type: 'token'; text: string }
  | { type: 'done' }
  | { type: 'error'; detail: string }

export interface GraphNode {
  id: string
  name: string
  type: string
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  label: string
  weight: number
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface BuildStatus {
  running: boolean
  processed: number
  failed: number
  total: number
  new_entities: number
  new_relationships: number
  remaining_chunks: number | null
  error: string | null
  last_error: string | null
  started_at: string | null
  finished_at: string | null
}

export interface EntityRelationship {
  direction: 'in' | 'out'
  label: string
  other_id: string
  other_name: string
  document_id: string
  filename: string
  page_number: number | null
  excerpt: string
}

export interface EntityMention {
  document_id: string
  filename: string
  page_number: number | null
  surface_form: string
  excerpt: string
}

export interface EntityDetail {
  id: string
  name: string
  type: string
  degree: number
  relationships: EntityRelationship[]
  mentions: EntityMention[]
}

export type Provider = 'local' | 'anthropic' | 'gemini'

export interface Settings {
  provider: Provider
  available: Record<Provider, boolean>
  models: {
    anthropic: string
    gemini: string
    local: {
      chat_model: string
      extraction_model: string
      vision_model: string
      embedding_model: string
    }
  }
}

export interface GraphCoverage {
  document_id: string
  filename: string
  total_chunks: number
  extracted_chunks: number
}

export interface SearchHit {
  id: string
  name: string
  type: string
  match_field: 'name' | 'type' | 'relationship' | 'mention'
  snippet: string
}

export interface MergedEntityRef {
  id: string
  name: string
}

export interface MergeHistoryEntry {
  id: string
  keep_id: string
  keep_name: string
  merged: MergedEntityRef[]
  created_at: string
  undone: boolean
}

export const ENTITY_COLORS: Record<string, string> = {
  PERSON: '#e15759',
  ORGANIZATION: '#4e79a7',
  CONCEPT: '#59a14f',
  TECHNOLOGY: '#f28e2b',
  LOCATION: '#b07aa1',
  EVENT: '#edc948',
  OTHER: '#9c9c9c',
}
