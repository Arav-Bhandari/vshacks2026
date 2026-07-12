"use client";

import { RadialBar, RadialBarChart, PolarAngleAxis } from "recharts";

const toneColors = {
  accent: "var(--accent)",
  amber: "var(--amber)",
  red: "var(--red)",
  blue: "var(--blue)",
};

export function RadialGauge({
  value,
  label,
  tone = "accent",
  size = 140,
}: {
  value: number;
  label: string;
  tone?: keyof typeof toneColors;
  size?: number;
}) {
  const pct = Math.max(0, Math.min(100, value));
  const data = [{ value: pct, fill: toneColors[tone] }];
  return (
    <div className="flex flex-col items-center">
      <RadialBarChart
        width={size}
        height={size}
        cx="50%"
        cy="50%"
        innerRadius="72%"
        outerRadius="100%"
        barSize={10}
        data={data}
        startAngle={90}
        endAngle={-270}
      >
        <PolarAngleAxis
          type="number"
          domain={[0, 100]}
          angleAxisId={0}
          tick={false}
        />
        <RadialBar
          background={{ fill: "var(--border)" }}
          dataKey="value"
          cornerRadius={6}
          isAnimationActive
          animationDuration={700}
          animationEasing="ease-out"
        />
        <text
          x="50%"
          y="47%"
          textAnchor="middle"
          dominantBaseline="middle"
          className="font-display tabular-nums"
          style={{ fill: "var(--ink)", fontSize: size * 0.19 }}
        >
          {Math.round(pct)}
        </text>
        <text
          x="50%"
          y="63%"
          textAnchor="middle"
          dominantBaseline="middle"
          style={{ fill: "var(--ink-muted)", fontSize: size * 0.08 }}
        >
          / 100
        </text>
      </RadialBarChart>
      <p className="mt-1 text-center text-xs font-medium text-ink-muted">
        {label}
      </p>
    </div>
  );
}
