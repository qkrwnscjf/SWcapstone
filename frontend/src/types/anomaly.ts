export type PredictResponse = {
  gate_score: number;
  decision: "normal" | "anomaly";
  issueType: string;
  heatmap_score: number;
  heatmap_overlay: string | null;
  ensemble_active: boolean;
  latency: {
    total_latency_ms: number;
  };
};
