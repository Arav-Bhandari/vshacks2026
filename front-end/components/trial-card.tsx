import { ExternalLink, Users, Clock3 } from "lucide-react";
import type { Trial } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

export function TrialCard({ trial }: { trial: Trial }) {
  return (
    <Card className="p-5 flex flex-col gap-3 transition-[border-color,box-shadow] duration-150 hover:border-accent/40 hover:shadow-md">
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-display text-base leading-snug text-ink">
          {trial.title}
        </h3>
        <Badge tone="accent" className="shrink-0">
          {trial.phase || "N/A"}
        </Badge>
      </div>

      <a
        href={`https://clinicaltrials.gov/study/${trial.nct_id}`}
        target="_blank"
        rel="noreferrer"
        className="inline-flex w-fit items-center gap-1 font-mono text-xs text-accent hover:underline"
      >
        {trial.nct_id}
        <ExternalLink className="h-3 w-3" />
      </a>

      {trial.conditions?.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {trial.conditions.slice(0, 4).map((c) => (
            <Badge key={c} tone="neutral" className="normal-case">
              {c}
            </Badge>
          ))}
        </div>
      )}

      <div className="mt-1 flex items-center gap-4 text-xs text-ink-muted font-mono tabular-nums">
        <span className="flex items-center gap-1">
          <Users className="h-3.5 w-3.5" /> {trial.enrollment ?? "?"} pts
        </span>
        <span className="flex items-center gap-1">
          <Clock3 className="h-3.5 w-3.5" /> {trial.duration_months ?? "?"} mo
        </span>
        <span className="ml-auto text-ink-muted/80">{trial.sponsor}</span>
      </div>
    </Card>
  );
}
