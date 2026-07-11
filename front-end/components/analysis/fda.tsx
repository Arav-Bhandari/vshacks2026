import { CheckCircle2, FileStack, AlertTriangle } from "lucide-react";
import type { FdaAnalysis, FdaGap } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RadialGauge } from "@/components/analysis/radial-gauge";
import { Badge } from "@/components/ui/badge";

const SEVERITY_ORDER = ["high", "medium", "low"] as const;
const SEVERITY_TONE = { high: "red", medium: "amber", low: "blue" } as const;

export function FdaTab({ fda }: { fda: FdaAnalysis | null }) {
  if (!fda) {
    return (
      <p className="py-10 text-sm text-ink-muted">
        FDA compliance review is not yet available for this protocol.
      </p>
    );
  }

  const grouped = SEVERITY_ORDER.map((sev) => ({
    sev,
    gaps: fda.gaps.filter((g) => g.severity === sev),
  })).filter((g) => g.gaps.length > 0);

  return (
    <div className="flex flex-col gap-4 py-6">
      <Card>
        <CardContent className="flex flex-col items-center gap-4 pt-5 sm:flex-row sm:items-start">
          <RadialGauge value={fda.compliance_score} label="Compliance score" tone="accent" />
          <p className="text-sm leading-relaxed text-ink-muted">{fda.summary}</p>
        </CardContent>
      </Card>

      {grouped.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-amber" /> Gaps
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-5">
            {grouped.map(({ sev, gaps }) => (
              <div key={sev}>
                <p className="mb-2 font-mono text-xs uppercase tracking-wide text-ink-muted">
                  {sev} severity
                </p>
                <div className="flex flex-col gap-2">
                  {gaps.map((g) => (
                    <GapRow key={g.element} gap={g} />
                  ))}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {fda.strengths.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-accent" /> Strengths
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="flex flex-col gap-2">
              {fda.strengths.map((s) => (
                <li key={s} className="flex gap-2 text-sm text-ink">
                  <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent" />
                  {s}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {fda.documents_used.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileStack className="h-4 w-4 text-blue" /> Documents used
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-2">
            {fda.documents_used.map((d) => (
              <div
                key={d.filename}
                className="flex items-center justify-between rounded-md border border-border px-3 py-2 text-sm"
              >
                <div>
                  <p className="text-ink">{d.title}</p>
                  <p className="font-mono text-xs text-ink-muted">{d.filename}</p>
                </div>
                <Badge tone="blue">{d.category}</Badge>
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function GapRow({ gap }: { gap: FdaGap }) {
  return (
    <div className="rounded-md border border-border p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="font-medium text-ink">{gap.element}</p>
        <Badge tone={SEVERITY_TONE[gap.severity as keyof typeof SEVERITY_TONE] ?? "neutral"}>
          {gap.severity}
        </Badge>
      </div>
      <p className="mt-1.5 text-sm text-ink-muted">{gap.recommendation}</p>
      <p className="mt-1.5 font-mono text-xs text-ink-muted/80">source: {gap.source}</p>
    </div>
  );
}
