/**
 * API Service — wires React frontend to FastAPI backend
 * Base URL configurable via VITE_API_URL env var
 */

export const BASE_URL =
  (import.meta as any).env?.VITE_API_URL ??
  "https://adikaprojects--jf-rag-backend-fastapi-app.modal.run";

// ── Image query detection ────────────────────────────────────────────────────
// When ANY of these keywords appear in the user's message, image result cards
// are shown above the assistant answer (Feature 1).
export const IMAGE_KEYWORDS = [
  "image", "images",
  "pic", "pics",
  "picture", "pictures",
  "photo", "photos",
  "photograph", "photographs",
  "show me",
  "visual", "visuals",
  "portrait", "portraits",
  "snapshot", "snapshots",
  "screenshot", "screenshots",
  "figure", "figures",
  "illustration", "illustrations",
  "diagram", "diagrams",
  "look like",
  "what does",
];

/**
 * Returns true when the query contains at least one visual-intent keyword.
 * Used by Index.tsx to decide whether to pass imageQueryDetected=true
 * and by ChatView.tsx to decide whether to render <ImageResultCards />.
 */
export function isImageQuery(query: string): boolean {
  const lower = query.toLowerCase();
  return IMAGE_KEYWORDS.some((kw) => lower.includes(kw));
}

export interface QueryRequest {
  query: string;
  top_k?: number;
  rerank_top_n?: number;
  force_provider?: string | null;
  use_cache?: boolean;
  use_hyde?: boolean;
  use_query_rewrite?: boolean;
  use_parent_child?: boolean;
  use_image_search?: boolean;
}

export interface RetrievalResult {
  chunk_id: string;
  text: string;
  source_file: string;
  page: number;
  doc_type: string;
  confidence_score: number;
  image_path?: string;
  caption?: string;
  geo_city?: string;
  similarity?: number;
}

export interface QueryResponse {
  answer: string;
  provider: string;
  model: string;
  sources: RetrievalResult[];
  image_sources: RetrievalResult[];
  error?: string;
  cached: boolean;
  rewritten_query?: string;
  hyde_used: boolean;
  guardrail_triggered: boolean;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  features: Record<string, boolean | string>;
}

export interface StatusResponse {
  system_health: string;
  timestamp: string;
  vector_store: {
    faiss_vectors: number;
    bm25_chunks: number;
    parent_chunks: number;
    embedding_model: string;
    image_vectors: number;
    image_records: number;
  };
  performance_metrics: {
    total_queries: number;
    avg_execution_time: number;
  };
  components: Record<string, string>;
}

export interface CacheStats {
  cache_dir: string;
  size: number;
}


async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

async function requestMultipart<T>(path: string, formData: FormData): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  query: (body: QueryRequest) =>
    request<QueryResponse>("/query", { method: "POST", body: JSON.stringify(body) }),

  queryWithFile: (formData: FormData) =>
    requestMultipart<QueryResponse>("/query-with-file", formData),

  health: () => request<HealthResponse>("/health"),

  status: () => request<StatusResponse>("/status"),

  cacheStats: () => request<CacheStats>("/cache/stats"),

  cacheClear: () => request<{ status: string }>("/cache/clear", { method: "POST" }),

  cacheDeleteEntry: (query: string) =>
    request<{ status: string }>("/cache/delete-entry", {
      method: "POST",
      body: JSON.stringify({ query }),
    }),
};
