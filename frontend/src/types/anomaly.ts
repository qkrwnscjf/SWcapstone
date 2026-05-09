export type PredictResponse = {
  gate_score: number;
  decision: "normal" | "anomaly";
  issueType?: string;
  heatmap_score: number;
  heatmap_overlay: string | null;
  normalized_score_heatmap?: string | null;
  heatmap_called?: boolean;
  ensemble_active: boolean;
  latency: {
    total_latency_ms: number;
  };
};
