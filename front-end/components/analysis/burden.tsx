import type { Burden } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RadialGauge } from "@/components/analysis/radial-gauge";
import { Progress } from "@/components/ui/progress";

export function BurdenTab({ burden }: { burden: Burden | null }) {
  if (!burden) {
    return (
      <p className="py-10 text-sm text-ink-muted">
        Burden scoring is not yet available for this protocol.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-4 py-6">
      <Card>
        <CardContent className="flex flex-wrap justify-around gap-6 pt-5">
          <RadialGauge value={burden.complexity_score} label="Complexity" tone="amber" />
          <RadialGauge value={burden.recruitment_difficulty} label="Recruitment difficulty" tone="red" />
          <RadialGauge value={burden.patient_burden} label="Patient burden" tone="blue" />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Contributing factors</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {burden.factors.map((f) => (
            <div key={f.name}>
              <div className="mb-1 flex items-center justify-between text-sm">
                <span className="font-medium text-ink">{f.name}</span>
                <span className="font-mono text-xs text-ink-muted">
                  {Math.round(f.score)} / 100
                </span>
              </div>
              <Progress
                value={f.score}
                tone={f.score >= 66 ? "red" : f.score >= 33 ? "amber" : "accent"}
              />
              <p className="mt-1.5 text-xs text-ink-muted">{f.detail}</p>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
