export type PredictResponse = {
  gate_score: number;
  decision: string;
  heatmap_called: boolean;
  heatmap_score: number | null;
  override_active: boolean;
  latency: {
    gate_latency_ms: number;
    heatmap_latency_ms: number;
    total_latency_ms: number;
  };
  cascade?: {
    gate_architecture: string | null;
    heatmap_architecture: string;
    t_low: number;
    t_high: number;
  };
  heatmap_overlay?: string | null;
  normalized_score_heatmap?: string | null;
};
