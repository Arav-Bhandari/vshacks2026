"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Baseline } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { StatCard } from "@/components/analysis/stat-card";
import { Users, CalendarRange, Database } from "lucide-react";

export function BenchmarksTab({ baseline }: { baseline: Baseline | null }) {
  if (!baseline) {
    return (
      <p className="py-10 text-sm text-ink-muted">
        Benchmark data is not yet available for this protocol.
      </p>
    );
  }

  const data = [
    {
      name: "Duration",
      base: baseline.ci_low,
      range: baseline.ci_high - baseline.ci_low,
    },
  ];

  return (
    <div className="flex flex-col gap-4 py-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard
          label="Expected duration"
          value={`${baseline.expected_duration_months.toFixed(1)} mo`}
          sub={`95% CI ${baseline.ci_low.toFixed(1)}–${baseline.ci_high.toFixed(1)} mo`}
          icon={CalendarRange}
          tone="blue"
        />
        <StatCard
          label="Median enrollment"
          value={`${baseline.median_enrollment}`}
          sub="patients across comparable trials"
          icon={Users}
          tone="accent"
        />
        <StatCard
          label="Trials in cohort"
          value={`${baseline.n_trials}`}
          sub="used to compute this baseline"
          icon={Database}
          tone="amber"
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Duration confidence interval</CardTitle>
          <CardDescription>
            95% CI across comparable trials, with the expected duration marked.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={140}>
            <BarChart data={data} layout="vertical" barCategoryGap={40}>
              <CartesianGrid horizontal={false} stroke="var(--grid-line)" />
              <XAxis
                type="number"
                domain={[0, Math.ceil(baseline.ci_high + 4)]}
                tick={{ fill: "var(--ink-muted)", fontSize: 12 }}
                axisLine={{ stroke: "var(--border)" }}
                tickLine={false}
                unit=" mo"
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fill: "var(--ink-muted)", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: "var(--accent-soft)" }}
                contentStyle={{
                  background: "var(--surface-raised)",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={(value, key) =>
                  key === "range"
                    ? [`${(value as number).toFixed(1)} mo range`, "CI width"]
                    : [`${(value as number).toFixed(1)} mo`, "CI low"]
                }
              />
              <Bar dataKey="base" stackId="ci" fill="transparent" />
              <Bar
                dataKey="range"
                stackId="ci"
                fill="var(--accent)"
                radius={[4, 4, 4, 4]}
                barSize={28}
              />
              <ReferenceLine
                x={baseline.expected_duration_months}
                stroke="var(--ink)"
                strokeWidth={2}
                label={{
                  value: "expected",
                  position: "top",
                  fill: "var(--ink)",
                  fontSize: 11,
                }}
              />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
