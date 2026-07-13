import { ExternalLink } from "lucide-react";
import type { Trial } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

const COMPONENTS: { key: keyof NonNullable<Trial["similarity"]>; label: string }[] = [
  { key: "condition", label: "Condition" },
  { key: "phase", label: "Phase" },
  { key: "endpoints", label: "Endpoints" },
  { key: "design", label: "Design" },
];

const asPercent = (score: number) => Math.round(score * 100);

export function SimilarTrialsTab({ trials }: { trials: Trial[] | null }) {
  if (!trials || trials.length === 0) {
    return (
      <p className="py-10 text-sm text-ink-muted">
        No similar trials were found for this protocol.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-4 py-6">
      {trials.map((t) => (
        <Card key={t.nct_id} className="p-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h3 className="font-display text-base leading-snug">{t.title}</h3>
              <a
                href={`https://clinicaltrials.gov/study/${t.nct_id}`}
                target="_blank"
                rel="noreferrer"
                className="mt-1 inline-flex items-center gap-1 font-mono text-xs text-accent hover:underline"
              >
                {t.nct_id}
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>
            <div className="flex items-center gap-2">
              <Badge tone="accent">{t.phase}</Badge>
              {t.similarity && (
                <span className="font-mono text-lg text-ink tabular-nums">
                  {asPercent(t.similarity.total)}%
                </span>
              )}
            </div>
          </div>

          {t.similarity && (
            <div className="mt-4 grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4">
              {COMPONENTS.map(({ key, label }) => (
                <div key={key}>
                  <div className="mb-1 flex items-center justify-between text-xs text-ink-muted">
                    <span>{label}</span>
                    <span className="font-mono tabular-nums">
                      {asPercent(t.similarity![key])}%
                    </span>
                  </div>
                  <Progress value={asPercent(t.similarity![key])} />
                </div>
              ))}
            </div>
          )}
        </Card>
      ))}
    </div>
  );
}
