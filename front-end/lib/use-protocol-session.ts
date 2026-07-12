"use client";

import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { ProgressEvent, Session } from "@/lib/types";

export function useProtocolSession(sessionId: string) {
  const [session, setSession] = useState<Session | null>(null);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let cancelled = false;
    let ws: WebSocket | null = null;

    const stopPolling = () => {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
    };

    const applySession = (s: Session) => {
      if (cancelled) return;
      setSession(s);
      if (s.progress) setProgress(s.progress);
      if (s.status === "complete" || s.status === "error") stopPolling();
    };

    const startPolling = () => {
      if (pollRef.current) return;
      pollRef.current = setInterval(async () => {
        try {
          const s = await api.getSession(sessionId);
          applySession(s);
        } catch {
          // transient poll failure; next tick retries
        }
      }, 3000);
    };

    async function init() {
      try {
        const s = await api.getSession(sessionId);
        applySession(s);
        if (s.status === "complete" || s.status === "error") return;
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof ApiError ? err.message : "Could not load session.",
          );
        }
        return;
      }

      try {
        ws = new WebSocket(api.wsUrl(sessionId));
        ws.onmessage = (evt) => {
          try {
            const data = JSON.parse(evt.data) as ProgressEvent;
            if (cancelled) return;
            setProgress(data);
            if (data.status === "done" || data.status === "error") {
              api.getSession(sessionId).then(applySession).catch(() => {});
            }
          } catch {
            // ignore malformed frame
          }
        };
        ws.onerror = () => startPolling();
        ws.onclose = () => startPolling();
      } catch {
        startPolling();
      }
    }

    init();

    return () => {
      cancelled = true;
      ws?.close();
      stopPolling();
    };
  }, [sessionId]);

  return { session, progress, error };
}
