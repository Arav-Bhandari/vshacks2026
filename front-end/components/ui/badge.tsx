import { cn } from "@/lib/utils";
import type { HTMLAttributes } from "react";

type Tone = "accent" | "amber" | "red" | "blue" | "neutral";

const tones: Record<Tone, string> = {
  accent: "bg-accent-soft text-accent border-accent/20",
  amber: "bg-amber-soft text-amber border-amber/20",
  red: "bg-red-soft text-red border-red/20",
  blue: "bg-blue-soft text-blue border-blue/20",
  neutral: "bg-surface-raised text-ink-muted border-border",
};

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: Tone;
}

export function Badge({ className, tone = "neutral", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium font-mono uppercase tracking-wide",
        tones[tone],
        className,
      )}
      {...props}
    />
  );
}
