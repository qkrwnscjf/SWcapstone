import type { DashboardResponse, ModelVersion, TrainingRecipe } from "../types/mlops";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function parseResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }

  return res.json() as Promise<T>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${path}`, init);
    return await parseResponse<T>(res);
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error("백엔드 연결에 실패했습니다. API 서버가 실행 중인지 확인해 주세요.");
    }
    throw error;
  }
}

export function fetchDashboard() {
  return request<DashboardResponse>("/mlops/dashboard").then((dashboard) => ({
    ...dashboard,
    training_recipes: dashboard.training_recipes ?? [],
    deployment: dashboard.deployment ?? {
      production_model_id:
        dashboard.runtime_config?.current_model_id ??
        dashboard.model_versions.find((model) => model.status === "production")?.id ??
        null,
      staging_model_id:
        dashboard.model_versions.find((model) => model.status === "candidate" || model.status === "staging")?.id ??
        null,
      canary_model_id: dashboard.model_versions.find((model) => model.status === "canary")?.id ?? null,
      canary_line: null,
      previous_production_model_id: null,
      last_action: null,
      last_action_at: null,
    },
    interfaces: dashboard.interfaces ?? {},
  }));
}

export function uploadFeedback(payload: {
  file: File;
  feedbackType: string;
  label: string;
  operator: string;
  comment: string;
  line: string;
  predictedLabel: string;
  gateScore?: number;
  heatmapScore?: number;
}) {
  const form = new FormData();
  form.append("file", payload.file);
  form.append("feedback_type", payload.feedbackType);
  form.append("label", payload.label);
  form.append("operator", payload.operator);
  form.append("comment", payload.comment);
  form.append("line", payload.line);
  form.append("predicted_label", payload.predictedLabel);
  form.append("gate_score", String(payload.gateScore ?? 0));
  form.append("heatmap_score", String(payload.heatmapScore ?? 0));
  return request("/mlops/feedback", { method: "POST", body: form });
}

export function deleteFeedback(feedbackId: string) {
  return request(`/mlops/feedback/${feedbackId}`, { method: "DELETE" });
}

export function uploadDatasetFiles(payload: {
  files: File[];
  label: string;
  sourceType: string;
  line: string;
  comment: string;
  datasetMode?: "append" | "new";
  datasetVersionId?: string;
  datasetName?: string;
}) {
  const form = new FormData();
  payload.files.forEach((file) =>
    form.append("files", file, file.name)
  );
  return request("/mlops/datasets/upload", { method: "POST", body: form });
}

export function materializeFeedbackDataset(payload: {
  mode: "append" | "new";
  targetDatasetId?: string;
  datasetName?: string;
  feedbackItemIds?: string[];
}) {
  return request("/mlops/datasets/from-feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mode: payload.mode,
      target_dataset_id: payload.targetDatasetId || null,
      dataset_name: payload.datasetName || null,
      feedback_item_ids: payload.feedbackItemIds || [],
    }),
  });
}

export function uploadArchitecture(payload: {
  file: File;
  kind: string;
  name: string;
}) {
  const form = new FormData();
  form.append("file", payload.file);
  form.append("kind", payload.kind);
  form.append("name", payload.name);
  return request("/mlops/architectures/upload", { method: "POST", body: form });
}

export function createTrainingRun(payload: {
  architecture?: string;
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  optimizer?: string;
  augmentation?: boolean;
  datasetVersionId?: string;
  baseModelVersionId?: string | null;
  recipeId?: string;
  targetLine?: string | null;
  gateArchitectureId?: string;
  heatmapArchitectureId?: string;
  trainStrategy?: string;
  notes?: string;
}) {
  return request("/mlops/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      architecture: payload.architecture || payload.gateArchitectureId || "ARCH-GATE-EFF",
      epochs: payload.epochs || 10,
      batch_size: payload.batchSize || 32,
      learning_rate: payload.learningRate || 0.001,
      optimizer: payload.optimizer || "Adam",
      augmentation: payload.augmentation !== undefined ? payload.augmentation : true,
    }),
  });
}

export function fetchTrainingStatus() {
  return request("/mlops/training/status");
}

export function deployModel(payload: {
  model_id: string;
  gate_file?: string;
  heatmap_file?: string;
  ensemble_enabled: boolean;
}) {
  return request("/mlops/deploy", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

// promoteModel은 UI 호환성을 위해 유지하되 deployModel을 호출하도록 래핑
export function promoteModel(modelVersionId: string, targetStatus: string, gateFile?: string, heatmapFile?: string) {
  return deployModel({
    model_id: modelVersionId,
    gate_file: gateFile,
    heatmap_file: heatmapFile,
    ensemble_enabled: true
  });
}

export function startCanary(modelVersionId: string) {
  return deployModel({
    model_id: modelVersionId,
    ensemble_enabled: true,
  });
}

export function rollbackDeployment(modelVersionId?: string) {
  return deployModel({
    model_id: modelVersionId || "MODEL-R3-FINAL",
    ensemble_enabled: true,
  });
}

export function stopTrainingRun() {
  return request("/mlops/training/status");
}

export function saveTrainingRecipe(payload: TrainingRecipe) {
  return Promise.resolve({ recipe: payload });
}

export function deleteTrainingRun(runId: string) {
  return request(`/mlops/training/runs/${runId}`, { method: "DELETE" });
}
