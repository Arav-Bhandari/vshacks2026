import { Check, Loader2, X } from "lucide-react";
import { PIPELINE_STEPS } from "@/lib/types";
import type { ProgressEvent } from "@/lib/types";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

export function PipelineStepper({ progress }: { progress: ProgressEvent | null }) {
  const currentIndex = progress
    ? PIPELINE_STEPS.findIndex((s) => s.key === progress.step)
    : -1;
  const errored = progress?.status === "error";

  return (
    <div className="mx-auto max-w-2xl px-6 py-24">
      <p className="text-center font-mono text-xs uppercase tracking-[0.2em] text-accent">
        Analyzing protocol
      </p>
      <h1 className="mt-2 text-center font-display text-2xl tracking-tight">
        {progress?.detail ?? "Warming up the pipeline…"}
      </h1>

      <div className="mt-8">
        <Progress
          value={progress?.pct ?? 0}
          tone={errored ? "red" : "accent"}
        />
        <p className="mt-2 text-right font-mono text-xs text-ink-muted">
          {Math.round(progress?.pct ?? 0)}%
        </p>
      </div>

      <ol className="mt-8 flex flex-col gap-1">
        {PIPELINE_STEPS.map((step, i) => {
          const done = currentIndex > i || (currentIndex === i && !errored && progress?.status === "done");
          const active = currentIndex === i && !done;
          const failed = active && errored;
          return (
            <li
              key={step.key}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2.5 text-sm transition-colors duration-200",
                active && !failed && "bg-accent-soft",
                failed && "bg-red-soft",
              )}
            >
              <span
                className={cn(
                  "flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-[10px]",
                  done && "border-accent bg-accent text-accent-ink",
                  active && !failed && "border-accent text-accent",
                  failed && "border-red text-red",
                  !done && !active && "border-border text-ink-muted",
                )}
              >
                {done ? (
                  <Check className="h-3 w-3" />
                ) : failed ? (
                  <X className="h-3 w-3" />
                ) : active ? (
                  <Loader2 className="h-3 w-3 animate-spin [animation-duration:600ms]" />
                ) : (
                  i + 1
                )}
              </span>
              <span
                className={cn(
                  "font-medium",
                  done && "text-ink",
                  active && !failed && "text-accent",
                  failed && "text-red",
                  !done && !active && "text-ink-muted",
                )}
              >
                {step.label}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
