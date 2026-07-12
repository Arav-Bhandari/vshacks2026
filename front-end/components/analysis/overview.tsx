import { Clock3, TrendingUp, ShieldCheck, Gauge } from "lucide-react";
import type { Session } from "@/lib/types";
import { StatCard } from "@/components/analysis/stat-card";
import { RadialGauge } from "@/components/analysis/radial-gauge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function OverviewTab({ session }: { session: Session }) {
  const { ml_prediction, fda_analysis, burden } = session;

  return (
    <div className="grid gap-4 py-6 sm:grid-cols-2 lg:grid-cols-4">
      <StatCard
        label="Predicted duration"
        value={
          ml_prediction ? `${ml_prediction.predicted_duration_months.toFixed(1)} mo` : "—"
        }
        sub={
          ml_prediction?.baseline_duration_months != null
            ? `baseline ${ml_prediction.baseline_duration_months.toFixed(1)} mo`
            : "awaiting ML prediction"
        }
        icon={Clock3}
        tone="blue"
      />
      <StatCard
        label="Overrun risk"
        value={ml_prediction ? `${Math.round(ml_prediction.overrun_risk_pct)}%` : "—"}
        sub="probability of exceeding baseline"
        icon={TrendingUp}
        tone={
          ml_prediction && ml_prediction.overrun_risk_pct >= 60
            ? "red"
            : ml_prediction && ml_prediction.overrun_risk_pct >= 30
              ? "amber"
              : "accent"
        }
      />
      <StatCard
        label="FDA compliance"
        value={fda_analysis ? `${Math.round(fda_analysis.compliance_score)}` : "—"}
        sub={fda_analysis ? `${fda_analysis.gaps.length} gaps found` : "awaiting review"}
        icon={ShieldCheck}
        tone="accent"
      />
      <StatCard
        label="Complexity score"
        value={burden ? `${Math.round(burden.complexity_score)}` : "—"}
        sub="protocol complexity index"
        icon={Gauge}
        tone="amber"
      />

      {burden && (
        <Card className="sm:col-span-2 lg:col-span-4">
          <CardHeader>
            <CardTitle>Burden summary</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap justify-around gap-6">
            <RadialGauge value={burden.complexity_score} label="Complexity" tone="amber" />
            <RadialGauge value={burden.recruitment_difficulty} label="Recruitment difficulty" tone="red" />
            <RadialGauge value={burden.patient_burden} label="Patient burden" tone="blue" />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
