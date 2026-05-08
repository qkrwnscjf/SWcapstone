export type DatasetSample = {
  id: string;
  file_url: string;
  file_name: string;
  label: string;
  source_type: string;
  feedback_type?: string | null;
  comment?: string;
  line?: string;
  operator?: string;
  predicted_label?: string;
  created_at: string;
};

export type DatasetVersion = {
  id: string;
  name: string;
  status: string;
  created_at: string;
  updated_at: string;
  source_dataset_id: string | null;
  sample_count: number;
  feedback_count: number;
  notes?: string;
  samples: DatasetSample[];
};

export type Architecture = {
  id: string;
  name: string;
  kind: "gate" | "heatmap";
  source: string;
  created_at: string;
  interface: {
    input: string;
    output: string;
  };
  file_url: string | null;
};

export type TrainingRun = {
  id: string;
  status: string;
  created_at: string;
  dataset_version_id: string;
  gate_architecture_id: string;
  heatmap_architecture_id: string;
  train_strategy: string;
  notes: string;
  lineage: string;
  sample_count: number;
};

export type ModelVersion = {
  id: string;
  name: string;
  status: string;
  dataset_version_id: string | null;
  gate_architecture_id: string;
  heatmap_architecture_id: string;
  created_at: string;
  metrics: {
    f1: number | null;
    latency_ms: number | null;
  };
  lineage: string;
  training_run_id?: string;
};

export type FeedbackItem = {
  id: string;
  sample_id: string;
  dataset_version_id: string;
  feedback_type: string;
  label: string;
  operator: string;
  comment: string;
  line: string;
  predicted_label: string;
  image_url: string;
  created_at: string;
};

export type LogItem = {
  id: string;
  level: string;
  time: string;
  message: string;
};

export type DashboardResponse = {
  active_dataset_id: string;
  dataset_versions: DatasetVersion[];
  architectures: Architecture[];
  training_runs: any[];
  model_versions: ModelVersion[];
  feedback_items: any[];
  logs: LogItem[];
  interfaces: Record<string, string>;
  available_model_files: string[];
  runtime_config: {
    ensemble_enabled: boolean;
    current_model_id: string | null;
    gate_file: string | null;
    heatmap_file: string | null;
  };
};
