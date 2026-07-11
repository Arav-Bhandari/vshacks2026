import { cn } from "@/lib/utils";

export function Progress({
  value,
  className,
  tone = "accent",
}: {
  value: number;
  className?: string;
  tone?: "accent" | "amber" | "red";
}) {
  const pct = Math.max(0, Math.min(100, value));
  const bar =
    tone === "amber" ? "bg-amber" : tone === "red" ? "bg-red" : "bg-accent";
  return (
    <div
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      className={cn(
        "h-1.5 w-full overflow-hidden rounded-full bg-border/60",
        className,
      )}
    >
      <div
        className={cn("h-full rounded-full transition-all duration-500 ease-out", bar)}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
