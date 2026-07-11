import ReactMarkdown from "react-markdown";
import { Download, FileCode2 } from "lucide-react";
import type { OptimizedProtocol } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

export function OptimizedTab({
  optimized,
  sessionId,
}: {
  optimized: OptimizedProtocol | null;
  sessionId: string;
}) {
  if (!optimized) {
    return (
      <p className="py-10 text-sm text-ink-muted">
        The optimized draft has not been generated yet.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-4 py-6">
      <Card>
        <CardHeader className="flex-row items-start justify-between gap-4">
          <div>
            <CardTitle>Summary</CardTitle>
            <p className="mt-2 text-sm leading-relaxed text-ink-muted">
              {optimized.summary}
            </p>
          </div>
          <div className="flex shrink-0 gap-2">
            <a href={api.exportUrl(sessionId, "usdm")} download>
              <Button variant="secondary" size="sm">
                <Download className="h-3.5 w-3.5" /> USDM JSON
              </Button>
            </a>
            <a href={api.exportUrl(sessionId, "xml")} download>
              <Button variant="secondary" size="sm">
                <FileCode2 className="h-3.5 w-3.5" /> XML
              </Button>
            </a>
          </div>
        </CardHeader>
      </Card>

      {optimized.changes.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recommended changes</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {optimized.changes.map((c, i) => (
              <div key={i} className="rounded-md border border-border p-3">
                <p className="font-mono text-xs uppercase tracking-wide text-accent">
                  {c.section}
                </p>
                <p className="mt-1 font-medium text-ink">{c.change}</p>
                <p className="mt-1 text-sm text-ink-muted">{c.rationale}</p>
                <p className="mt-1.5 font-mono text-xs text-ink-muted/80">
                  {c.citation}
                </p>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Optimized draft</CardTitle>
        </CardHeader>
        <CardContent className="prose-protocol">
          <ReactMarkdown>{optimized.markdown}</ReactMarkdown>
        </CardContent>
      </Card>
    </div>
  );
}
