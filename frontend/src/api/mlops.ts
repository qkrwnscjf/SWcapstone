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
  return request<DashboardResponse>("/mlops/dashboard");
}

export function uploadFeedback(payload: {
  file: File;
  feedbackType: string;
  label: string;
  operator: string;
  comment: string;
  line: string;
  predictedLabel: string;
}) {
  const form = new FormData();
  form.append("file", payload.file);
  form.append("feedback_type", payload.feedbackType);
  form.append("label", payload.label);
  form.append("operator", payload.operator);
  form.append("comment", payload.comment);
  form.append("line", payload.line);
  form.append("predicted_label", payload.predictedLabel);
  return request("/mlops/feedback", { method: "POST", body: form });
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
    form.append(
      "files",
      file,
      (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name
    )
  );
  form.append("label", payload.label);
  form.append("source_type", payload.sourceType);
  form.append("line", payload.line);
  form.append("comment", payload.comment);
  form.append("dataset_mode", payload.datasetMode || "append");
  form.append("dataset_version_id", payload.datasetVersionId || "");
  form.append("dataset_name", payload.datasetName || "");
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
  kind: "gate" | "heatmap";
  name: string;
}) {
  const form = new FormData();
  form.append("file", payload.file);
  form.append("kind", payload.kind);
  form.append("name", payload.name);
  return request("/mlops/architectures/upload", { method: "POST", body: form });
}

export function createTrainingRun(payload: {
  datasetVersionId?: string;
  baseModelVersionId?: string | null;
  recipeId?: string;
  epochs?: number;
  targetLine?: string | null;
  gateArchitectureId: string;
  heatmapArchitectureId: string;
  trainStrategy: string;
  notes: string;
}) {
  return request("/mlops/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_version_id: payload.datasetVersionId || null,
      base_model_version_id: payload.baseModelVersionId || null,
      recipe_id: payload.recipeId || "balanced-finetune-v1",
      epochs: payload.epochs || 3,
      gate_architecture_id: payload.gateArchitectureId,
      heatmap_architecture_id: payload.heatmapArchitectureId,
      train_strategy: payload.trainStrategy,
      notes: payload.notes,
    }),
  });
}

export function promoteModel(modelVersionId: string, targetStatus: "staging" | "production") {
  return request<{ model_version: ModelVersion }>("/mlops/models/promote", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_version_id: modelVersionId,
      target_status: targetStatus,
    }),
  });
}

export function stopTrainingRun(runId?: string) {
  return request("/mlops/train/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      run_id: runId || null,
    }),
  });
}

export function saveTrainingRecipe(payload: Omit<TrainingRecipe, "source" | "file_path">) {
  return request<{ recipe: TrainingRecipe }>("/mlops/recipes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function startCanary(modelVersionId: string, line: string) {
  return request<{ model_version: ModelVersion }>("/mlops/deployments/canary", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_version_id: modelVersionId,
      line,
    }),
  });
}

export function rollbackDeployment(modelVersionId?: string) {
  return request<{ model_version: ModelVersion }>("/mlops/deployments/rollback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_version_id: modelVersionId || null,
    }),
  });
}
