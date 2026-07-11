"use client";

import { use } from "react";
import { AlertCircle } from "lucide-react";
import { useProtocolSession } from "@/lib/use-protocol-session";
import { PipelineStepper } from "@/components/analysis/stepper";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { OverviewTab } from "@/components/analysis/overview";
import { SimilarTrialsTab } from "@/components/analysis/similar-trials";
import { BenchmarksTab } from "@/components/analysis/benchmarks";
import { BurdenTab } from "@/components/analysis/burden";
import { RiskTab } from "@/components/analysis/risk";
import { FdaTab } from "@/components/analysis/fda";
import { OptimizedTab } from "@/components/analysis/optimized";

export default function AnalysisPage({
  params,
}: {
  params: Promise<{ sessionId: string }>;
}) {
  const { sessionId } = use(params);
  const { session, progress, error } = useProtocolSession(sessionId);

  if (error) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-24 text-center">
        <AlertCircle className="mx-auto h-8 w-8 text-red" />
        <p className="mt-4 text-sm text-red">{error}</p>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-24 text-center text-sm text-ink-muted">
        Loading session…
      </div>
    );
  }

  if (session.status !== "complete") {
    if (session.status === "error") {
      return (
        <div className="mx-auto max-w-2xl px-6 py-24 text-center">
          <AlertCircle className="mx-auto h-8 w-8 text-red" />
          <p className="mt-4 text-sm text-red">
            {progress?.detail ?? "Analysis failed."}
          </p>
        </div>
      );
    }
    return <PipelineStepper progress={progress ?? session.progress} />;
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent">
        Protocol analysis
      </p>
      <h1 className="mt-2 font-display text-3xl tracking-tight">
        {session.filename}
      </h1>

      <Tabs defaultValue="overview" className="mt-6">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="similar">Similar trials</TabsTrigger>
          <TabsTrigger value="benchmarks">Benchmarks</TabsTrigger>
          <TabsTrigger value="burden">Burden</TabsTrigger>
          <TabsTrigger value="risk">Risk</TabsTrigger>
          <TabsTrigger value="fda">FDA</TabsTrigger>
          <TabsTrigger value="optimized">Optimized protocol</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <OverviewTab session={session} />
        </TabsContent>
        <TabsContent value="similar">
          <SimilarTrialsTab trials={session.similar_trials} />
        </TabsContent>
        <TabsContent value="benchmarks">
          <BenchmarksTab baseline={session.baseline} />
        </TabsContent>
        <TabsContent value="burden">
          <BurdenTab burden={session.burden} />
        </TabsContent>
        <TabsContent value="risk">
          <RiskTab ml={session.ml_prediction} />
        </TabsContent>
        <TabsContent value="fda">
          <FdaTab fda={session.fda_analysis} />
        </TabsContent>
        <TabsContent value="optimized">
          <OptimizedTab optimized={session.optimized_protocol} sessionId={sessionId} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
