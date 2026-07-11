"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MlPrediction } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { RadialGauge } from "@/components/analysis/radial-gauge";
import { StatCard } from "@/components/analysis/stat-card";
import { Clock3 } from "lucide-react";

export function RiskTab({ ml }: { ml: MlPrediction | null }) {
  if (!ml) {
    return (
      <p className="py-10 text-sm text-ink-muted">
        ML risk prediction is not yet available for this protocol.
      </p>
    );
  }

  const data = [...ml.shap_top5]
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
    .map((f) => ({ ...f, name: f.feature }));

  const gaugeTone = ml.overrun_risk_pct >= 60 ? "red" : ml.overrun_risk_pct >= 30 ? "amber" : "accent";

  return (
    <div className="flex flex-col gap-4 py-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard
          label="Predicted duration"
          value={`${ml.predicted_duration_months.toFixed(1)} mo`}
          sub={`vs baseline ${ml.baseline_duration_months.toFixed(1)} mo`}
          icon={Clock3}
          tone="blue"
        />
        <Card className="flex items-center justify-center p-5 sm:col-span-2">
          <RadialGauge value={ml.overrun_risk_pct} label="Overrun risk" tone={gaugeTone} size={120} />
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Top drivers (SHAP)</CardTitle>
          <CardDescription>
            Signed contribution to predicted duration &mdash; amber pushes
            duration up, teal pulls it down.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={Math.max(200, data.length * 56)}>
            <BarChart data={data} layout="vertical" margin={{ left: 12 }}>
              <CartesianGrid horizontal={false} stroke="var(--grid-line)" />
              <XAxis
                type="number"
                tick={{ fill: "var(--ink-muted)", fontSize: 12 }}
                axisLine={{ stroke: "var(--border)" }}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={140}
                tick={{ fill: "var(--ink)", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <ReferenceLine x={0} stroke="var(--border)" />
              <Tooltip
                cursor={{ fill: "var(--accent-soft)" }}
                contentStyle={{
                  background: "var(--surface-raised)",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                  fontSize: 12,
                  maxWidth: 260,
                }}
                formatter={(value, _key, item) => [
                  `${(value as number) > 0 ? "+" : ""}${(value as number).toFixed(2)} mo — ${item.payload.explanation}`,
                  "Impact",
                ]}
              />
              <Bar dataKey="impact" radius={[4, 4, 4, 4]} barSize={20}>
                {data.map((f) => (
                  <Cell
                    key={f.name}
                    fill={f.direction === "increase" ? "var(--amber)" : "var(--accent)"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Explanations</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {data.map((f) => (
            <div key={f.name} className="flex gap-3 text-sm">
              <span
                className="mt-1 h-2 w-2 shrink-0 rounded-full"
                style={{
                  background: f.direction === "increase" ? "var(--amber)" : "var(--accent)",
                }}
              />
              <p>
                <span className="font-medium text-ink">{f.feature}</span>{" "}
                <span className="text-ink-muted">{f.explanation}</span>
              </p>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
