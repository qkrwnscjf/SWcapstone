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
  created_at?: string;
  updated_at?: string;
  source_dataset_id?: string | null;
  sample_count: number;
  feedback_count: number;
  notes?: string;
  samples?: DatasetSample[];
};

export type Architecture = {
  id: string;
  name: string;
  kind: "gate" | "heatmap" | string;
  source?: string;
  created_at?: string;
  interface?: {
    input: string;
    output: string;
  };
  file_url?: string | null;
};

export type TrainingRecipe = {
  id: string;
  name: string;
  description: string;
  gate_model: string;
  batch_size: number;
  learning_rate: number;
  optimizer: string;
  epochs?: number;
  weight_decay: number;
  scheduler: string;
  early_stopping_patience: number;
  default_epochs: number;
  source?: string;
  file_path?: string;
};

export type TrainingRun = {
  id: string;
  name?: string;
  status?: string;
  created_at?: string;
  dataset_version_id?: string;
  base_model_version_id?: string | null;
  gate_architecture_id?: string;
  heatmap_architecture_id?: string;
  architecture?: string;
  params?: Record<string, unknown>;
  final_metrics?: Record<string, number>;
  recipe_id?: string;
  recipe?: TrainingRecipe;
  target_line?: string | null;
  epochs?: number;
  train_strategy?: string;
  notes?: string;
  lineage?: string;
  sample_count?: number;
  progress?: number;
  current_step?: string;
  started_at?: string | null;
  completed_at?: string | null;
  logs?: Array<{ time: string; message: string }>;
  device?: string;
  stop_requested?: boolean;
};

export type ModelVersion = {
  id: string;
  name: string;
  status: string;
  dataset_version_id?: string | null;
  gate_architecture_id?: string;
  heatmap_architecture_id?: string;
  created_at?: string;
  updated_at?: string;
  metrics: {
    f1?: number | null;
    latency_ms?: number | null;
    loss?: number | null;
    acc?: number | null;
  };
  lineage?: string;
  training_run_id?: string;
  base_model_version_id?: string | null;
  recipe_id?: string;
  gate_model_path?: string | null;
  heatmap_model_path?: string | null;
  calibrator_path?: string | null;
  target_line?: string | null;
};

export type FeedbackItem = {
  id: string;
  sample_id?: string;
  dataset_version_id?: string;
  feedback_type: string;
  label: string;
  operator: string;
  comment: string;
  line?: string;
  predicted_label?: string;
  model_prediction?: {
    gate_score?: number;
    heatmap_score?: number;
    predicted_label?: string;
  };
  image_url: string;
  created_at: string;
};

export type LogItem = {
  id: string;
  level: string;
  time: string;
  message: string;
};

export type DeploymentState = {
  production_model_id: string | null;
  staging_model_id: string | null;
  canary_model_id: string | null;
  canary_line: string | null;
  previous_production_model_id?: string | null;
  last_action?: string | null;
  last_action_at?: string | null;
};

export type DashboardResponse = {
  active_dataset_id: string;
  dataset_versions: DatasetVersion[];
  architectures: Architecture[];
  training_recipes?: TrainingRecipe[];
  training_runs: TrainingRun[];
  model_versions: ModelVersion[];
  feedback_items: FeedbackItem[];
  logs: LogItem[];
  deployment?: DeploymentState;
  interfaces?: Record<string, string>;
  available_model_files?: string[];
  runtime_config?: {
    ensemble_enabled?: boolean;
    current_model_id?: string | null;
  };
};
