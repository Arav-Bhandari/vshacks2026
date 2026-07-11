import type {
  AnalyzeResponse,
  HealthResponse,
  NlSearchResponse,
  SearchResponse,
  Session,
  SessionsResponse,
  UploadResponse,
} from "./types";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(
      "Could not reach the TrialScope API. Is the backend running?",
      0,
    );
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new ApiError(text || `Request failed (${res.status})`, res.status);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),

  searchTrials: (q: string, limit = 20) =>
    request<SearchResponse>(
      `/api/trials/search?q=${encodeURIComponent(q)}&limit=${limit}`,
    ),

  searchNl: (query: string) =>
    request<NlSearchResponse>("/api/search/nl", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    }),

  uploadProtocol: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<UploadResponse>("/api/protocol/upload", {
      method: "POST",
      body: form,
    });
  },

  analyzeProtocol: (sessionId: string) =>
    request<AnalyzeResponse>(`/api/protocol/${sessionId}/analyze`, {
      method: "POST",
    }),

  getSession: (sessionId: string) =>
    request<Session>(`/api/protocol/${sessionId}`),

  listSessions: () => request<SessionsResponse>("/api/sessions"),

  wsUrl: (sessionId: string) =>
    `${API_URL.replace(/^http/, "ws")}/ws/${sessionId}`,

  exportUrl: (sessionId: string, format: "usdm" | "xml") =>
    `${API_URL}/api/protocol/${sessionId}/export/${format}`,
};
