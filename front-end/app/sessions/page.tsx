"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AlertCircle, FileText, ChevronRight } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { SessionSummary } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

const STATUS_TONE = {
  created: "neutral",
  processing: "amber",
  complete: "accent",
  error: "red",
} as const;

export default function SessionsPage() {
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listSessions()
      .then((res) => setSessions(res.sessions))
      .catch((err) =>
        setError(
          err instanceof ApiError ? err.message : "Could not load sessions.",
        ),
      );
  }, []);

  return (
    <div className="mx-auto max-w-4xl px-6 py-16">
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent">
        History
      </p>
      <h1 className="mt-2 font-display text-3xl tracking-tight">Sessions</h1>
      <p className="mt-2 text-ink-muted">
        Every protocol you&apos;ve uploaded for analysis.
      </p>

      <div className="mt-8 flex flex-col gap-3">
        {error && (
          <div className="flex items-center gap-2 rounded-md border border-red/30 bg-red-soft px-4 py-3 text-sm text-red">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </div>
        )}

        {!sessions && !error &&
          Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full" />
          ))}

        {sessions && sessions.length === 0 && (
          <p className="text-sm text-ink-muted">
            Nothing here. The platform can only review what you upload.
          </p>
        )}

        {sessions && sessions.length > 0 && (
          <div className="stagger flex flex-col gap-3">
            {sessions.map((s) => (
              <Link key={s.session_id} href={`/protocol/${s.session_id}/analysis`}>
                <Card className="pressable flex items-center justify-between gap-4 p-4 transition-colors duration-150 hover:border-accent/40">
                  <div className="flex items-center gap-3 min-w-0">
                    <FileText className="h-5 w-5 shrink-0 text-ink-muted" />
                    <div className="min-w-0">
                      <p className="truncate font-medium text-ink">{s.filename}</p>
                      <p className="font-mono text-xs text-ink-muted tabular-nums">
                        {new Date(s.created_at).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-3">
                    <Badge tone={STATUS_TONE[s.status]}>{s.status}</Badge>
                    <ChevronRight className="h-4 w-4 text-ink-muted" />
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
