export type Similarity = {
  total: number;
  condition: number;
  phase: number;
  endpoints: number;
  design: number;
};

export type Trial = {
  nct_id: string;
  title: string;
  phase: string;
  conditions: string;
  interventions: string;
  primary_outcomes: string;
  enrollment: number | null;
  duration_months: number | null;
  sponsor: string;
  status: string;
  similarity?: Similarity;
};

export type SearchResponse = {
  trials: Trial[];
};

export type NlSearchResponse = {
  answer: string;
  trials: Trial[];
};

export type UploadResponse = {
  session_id: string;
};

export type AnalyzeResponse = {
  status: "started";
};

export type ProgressEvent = {
  step: string;
  status: "running" | "done" | "error";
  detail: string;
  pct: number;
};

export type BurdenFactor = {
  name: string;
  score: number;
  detail: string;
};

export type Burden = {
  complexity_score: number;
  recruitment_difficulty: number;
  patient_burden: number;
  factors: BurdenFactor[];
};

export type ShapFeature = {
  feature: string;
  impact: number;
  direction: "increases" | "decreases" | string;
  explanation: string;
};

export type MlPrediction = {
  predicted_duration_months: number;
  overrun_risk_pct: number;
  baseline_duration_months: number | null;
  shap_top5: ShapFeature[];
};

export type FdaGap = {
  element: string;
  severity: "high" | "medium" | "low" | string;
  recommendation: string;
  source: string;
};

export type FdaDocument = {
  filename: string;
  title: string;
  category: string;
};

export type FdaAnalysis = {
  compliance_score: number;
  summary: string;
  gaps: FdaGap[];
  strengths: string[];
  documents_used: FdaDocument[];
};

export type ProtocolChange = {
  section: string;
  change: string;
  rationale: string;
  citation: string;
};

export type OptimizedProtocol = {
  summary: string;
  changes: ProtocolChange[];
  markdown: string;
};

export type Baseline = {
  expected_duration_months: number | null;
  ci_low: number | null;
  ci_high: number | null;
  median_enrollment: number | null;
  n_trials: number;
};

export type SessionStatus = "created" | "processing" | "complete" | "error";

export type Session = {
  session_id: string;
  filename: string;
  status: SessionStatus;
  progress: ProgressEvent | null;
  usdm: object | null;
  similar_trials: Trial[] | null;
  baseline: Baseline | null;
  burden: Burden | null;
  ml_prediction: MlPrediction | null;
  fda_analysis: FdaAnalysis | null;
  optimized_protocol: OptimizedProtocol | null;
};

export type SessionSummary = {
  session_id: string;
  created_at: string;
  filename: string;
  status: SessionStatus;
};

export type SessionsResponse = {
  sessions: SessionSummary[];
};

export type HealthResponse = {
  status: string;
  trial_count: number;
};

export const PIPELINE_STEPS = [
  { key: "parse", label: "Parse PDF" },
  { key: "usdm", label: "USDM conversion" },
  { key: "similar", label: "Similar trials" },
  { key: "baseline", label: "Benchmarks" },
  { key: "burden", label: "Burden" },
  { key: "ml", label: "ML prediction" },
  { key: "fda", label: "FDA compliance" },
  { key: "optimize", label: "Optimized draft" },
] as const;
