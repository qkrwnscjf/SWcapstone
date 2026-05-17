import type { PredictResponse } from "../types/anomaly";
import type { DashboardResponse, ModelVersion, TrainingRecipe } from "../types/mlops";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export function cls(...names: Array<string | false | null | undefined>) {
  return names.filter(Boolean).join(" ");
}

export function clamp01(value: number | undefined | null) {
  if (value == null || Number.isNaN(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

export function formatDate(value?: string | null) {
  if (!value) return "-";
  return new Date(value).toLocaleString("ko-KR", { timeZone: "Asia/Seoul" });
}

export function formatPercent(value: number | undefined | null) {
  return `${Math.round(clamp01(value) * 100)}%`;
}

export function formatMetric(value?: number | null, digits = 3) {
  if (value == null || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

export function formatKoreaTimestamp(date = new Date()) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).formatToParts(date);
  const get = (type: string) => parts.find((part) => part.type === type)?.value ?? "00";
  return `${get("year")}-${get("month")}-${get("day")} ${get("hour")}:${get("minute")}:${get("second")} KST`;
}

export function asset(path: string) {
  if (!path) return "";
  return path.startsWith("http") || path.startsWith("data:") ? path : `${API_BASE}${path}`;
}

export function toDisplayDecision(decision?: string | null) {
  if (decision === "anomaly") return "이상";
  if (decision === "normal" || decision === "normal (heatmap)") return "정상";
  return decision ?? "-";
}

export function toIssueType(result: PredictResponse | null) {
  if (!result) return "-";
  if (result.decision === "anomaly") return "이상 패턴";
  if (result.heatmap_called) return "히트맵 확인";
  return "정상 통과";
}

export function pickProductionModel(dashboard: DashboardResponse | null) {
  if (!dashboard) return null;
  return (
    dashboard.model_versions.find((model) => model.id === dashboard.deployment.production_model_id) ??
    dashboard.model_versions.find((model) => model.status === "production") ??
    null
  );
}

export function pickStagingModel(dashboard: DashboardResponse | null) {
  if (!dashboard) return null;
  return (
    dashboard.model_versions.find((model) => model.id === dashboard.deployment.staging_model_id) ??
    dashboard.model_versions.find((model) => model.status === "staging") ??
    null
  );
}

export function pickCanaryModel(dashboard: DashboardResponse | null) {
  if (!dashboard) return null;
  return (
    dashboard.model_versions.find((model) => model.id === dashboard.deployment.canary_model_id) ??
    dashboard.model_versions.find((model) => model.status === "canary") ??
    null
  );
}

export function modelTone(model: ModelVersion | null | undefined) {
  if (!model) return "slate" as const;
  if (model.status === "production") return "green" as const;
  if (model.status === "canary") return "amber" as const;
  if (model.status === "failed" || model.status === "stopped") return "red" as const;
  return "blue" as const;
}

export function recipeDraftFrom(recipe?: TrainingRecipe): TrainingRecipe {
  return {
    id: recipe?.id ?? "custom-cpu-recipe",
    name: recipe?.name ?? "Custom CPU Recipe",
    description: recipe?.description ?? "",
    gate_model: recipe?.gate_model ?? "effnetb0",
    batch_size: recipe?.batch_size ?? 8,
    learning_rate: recipe?.learning_rate ?? 0.0003,
    optimizer: recipe?.optimizer ?? "AdamW",
    weight_decay: recipe?.weight_decay ?? 0,
    scheduler: recipe?.scheduler ?? "cosine",
    early_stopping_patience: recipe?.early_stopping_patience ?? 5,
    default_epochs: recipe?.default_epochs ?? 3,
  };
}

