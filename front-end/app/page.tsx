"use client";

import { useState, type FormEvent } from "react";
import { Search, Sparkles, AlertCircle } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { Trial } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { TrialCard } from "@/components/trial-card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

type Mode = "keyword" | "ai";

export default function HomePage() {
  const [mode, setMode] = useState<Mode>("keyword");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trials, setTrials] = useState<Trial[] | null>(null);
  const [answer, setAnswer] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setAnswer(null);
    setSearched(true);
    try {
      if (mode === "keyword") {
        const res = await api.searchTrials(query);
        setTrials(res.trials);
      } else {
        const res = await api.searchNl(query);
        setAnswer(res.answer);
        setTrials(res.trials);
      }
    } catch (err) {
      setTrials(null);
      setError(
        err instanceof ApiError
          ? err.message
          : "Something went wrong while searching.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <section className="border-b border-border">
        <div className="mx-auto max-w-6xl px-6 py-20 sm:py-28">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent">
            Trial intelligence, instantly
          </p>
          <h1 className="mt-4 max-w-2xl font-display text-4xl leading-[1.1] tracking-tight sm:text-5xl">
            Search the trial landscape and{" "}
            <span className="italic text-accent">de-risk your protocol</span>{" "}
            before you write it.
          </h1>
          <p className="mt-5 max-w-xl text-ink-muted">
            Query thousands of registered studies in plain English, or upload
            a draft protocol for automated benchmarking, burden scoring, and
            FDA compliance review.
          </p>

          <form onSubmit={onSubmit} className="mt-10 max-w-2xl">
            <div className="mb-3 flex gap-1 rounded-md border border-border bg-surface p-1 w-fit">
              <ModeButton
                active={mode === "keyword"}
                onClick={() => setMode("keyword")}
                icon={Search}
                label="Keyword"
              />
              <ModeButton
                active={mode === "ai"}
                onClick={() => setMode("ai")}
                icon={Sparkles}
                label="AI search"
              />
            </div>
            <div className="flex gap-2">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={
                  mode === "keyword"
                    ? "e.g. metastatic NSCLC, phase 3"
                    : "e.g. what phase 2 trials target relapsed AML with fewer than 100 patients?"
                }
                aria-label="Search clinical trials"
                className="h-12 flex-1 rounded-md border border-border bg-surface px-4 text-sm text-ink placeholder:text-ink-muted/70 focus:border-accent transition-colors duration-200"
              />
              <Button type="submit" size="lg" disabled={loading}>
                {loading ? "Searching…" : "Search"}
              </Button>
            </div>
          </form>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-12">
        {error && (
          <div className="mb-6 flex items-center gap-2 rounded-md border border-red/30 bg-red-soft px-4 py-3 text-sm text-red">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </div>
        )}

        {loading && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-44 w-full" />
            ))}
          </div>
        )}

        {!loading && answer && (
          <div className="mb-8 animate-fade-up rounded-lg border border-accent/25 bg-accent-soft px-5 py-4">
            <p className="mb-1 flex items-center gap-1.5 font-mono text-xs uppercase tracking-wide text-accent">
              <Sparkles className="h-3.5 w-3.5" /> AI answer
            </p>
            <p className="text-sm leading-relaxed text-ink">{answer}</p>
          </div>
        )}

        {!loading && trials && trials.length === 0 && (
          <p className="text-sm text-ink-muted">
            No trials matched that query. Try broadening your search.
          </p>
        )}

        {!loading && trials && trials.length > 0 && (
          <div className="grid animate-fade-up gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {trials.map((t) => (
              <TrialCard key={t.nct_id} trial={t} />
            ))}
          </div>
        )}

        {!loading && !searched && !error && (
          <p className="text-sm text-ink-muted">
            Results will appear here once you search.
          </p>
        )}
      </section>
    </div>
  );
}

function ModeButton({
  active,
  onClick,
  icon: Icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: typeof Search;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex items-center gap-1.5 rounded px-3 py-1.5 text-xs font-medium transition-colors duration-200",
        active ? "bg-accent text-accent-ink" : "text-ink-muted hover:text-ink",
      )}
    >
      <Icon className="h-3.5 w-3.5" />
      {label}
    </button>
  );
}
