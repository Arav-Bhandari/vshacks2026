export function SiteFooter() {
  return (
    <footer className="border-t border-border">
      <div className="mx-auto flex max-w-6xl flex-col gap-2 px-6 py-8 text-xs text-ink-muted sm:flex-row sm:items-center sm:justify-between">
        <p className="font-mono">
          TrialScope AI &mdash; clinical trial intelligence
        </p>
        <p>Not for clinical decision-making. Data sourced from ClinicalTrials.gov.</p>
      </div>
    </footer>
  );
}
