import type { LucideIcon } from "lucide-react";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export function StatCard({
  label,
  value,
  sub,
  icon: Icon,
  tone = "accent",
}: {
  label: string;
  value: string;
  sub?: string;
  icon: LucideIcon;
  tone?: "accent" | "amber" | "red" | "blue";
}) {
  const toneText = {
    accent: "text-accent",
    amber: "text-amber",
    red: "text-red",
    blue: "text-blue",
  }[tone];
  const toneBg = {
    accent: "bg-accent-soft",
    amber: "bg-amber-soft",
    red: "bg-red-soft",
    blue: "bg-blue-soft",
  }[tone];

  return (
    <Card className="p-5">
      <div className="flex items-start justify-between">
        <p className="text-xs font-medium uppercase tracking-wide text-ink-muted">
          {label}
        </p>
        <span className={cn("flex h-7 w-7 items-center justify-center rounded-md", toneBg)}>
          <Icon className={cn("h-4 w-4", toneText)} strokeWidth={2} />
        </span>
      </div>
      <p className="mt-3 font-display text-3xl tracking-tight text-ink">
        {value}
      </p>
      {sub && <p className="mt-1 text-xs text-ink-muted">{sub}</p>}
    </Card>
  );
}
