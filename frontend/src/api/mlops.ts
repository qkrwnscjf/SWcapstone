import type {
  DashboardResponse,
  DatasetVersion,
  DeploymentState,
  FeedbackItem,
  ModelVersion,
  TrainingRecipe,
  TrainingRun,
} from "../types/mlops";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const LOCAL_DEPLOYMENT_KEY = "steelvision.localDeployment";
const LOCAL_DATASETS_KEY = "steelvision.localDatasets";
const LOCAL_RECIPES_KEY = "steelvision.localRecipes";
const LOCAL_TRAINING_KEY = "steelvision.localTraining";
const LOCAL_MODEL_NAMES_KEY = "steelvision.localModelNames";

const DEFAULT_INTERFACES: Record<string, string> = {
  gate_input: "Tensor[B,3,224,224]",
  gate_output: "Anomaly probability",
  heatmap_input: "Tensor[B,3,224,224]",
  heatmap_output: "anomaly_score + normalized heatmap + overlay",
  cascade: "EfficientNet gate -> PatchCore heatmap",
};

const DEFAULT_RECIPES: TrainingRecipe[] = [
  {
    id: "cpu-balanced-effnetb0",
    name: "CPU Balanced EfficientNet",
    description: "Stable CPU recipe for regular fine-tuning from an existing gate model.",
    gate_model: "effnetb0",
    batch_size: 8,
    learning_rate: 0.0003,
    optimizer: "AdamW",
    weight_decay: 0.01,
    scheduler: "cosine",
    early_stopping_patience: 5,
    default_epochs: 3,
    source: "frontend",
  },
  {
    id: "cpu-fast-smoke",
    name: "CPU Fast Smoke Test",
    description: "Short recipe to verify the training pipeline before a longer run.",
    gate_model: "effnetb0",
    batch_size: 4,
    learning_rate: 0.001,
    optimizer: "Adam",
    weight_decay: 0,
    scheduler: "step",
    early_stopping_patience: 2,
    default_epochs: 1,
    source: "frontend",
  },
  {
    id: "cpu-low-lr-recovery",
    name: "CPU Low-LR Recovery",
    description: "Conservative fine-tuning recipe for sensitive production baselines.",
    gate_model: "effnetb0",
    batch_size: 12,
    learning_rate: 0.0001,
    optimizer: "AdamW",
    weight_decay: 0.02,
    scheduler: "cosine",
    early_stopping_patience: 6,
    default_epochs: 5,
    source: "frontend",
  },
];

type BackendTrainingStatus = {
  is_running?: boolean;
  progress?: number;
  message?: string;
  epoch?: number;
  metrics?: Record<string, number>;
};

type FeedbackMaterializeItem = Pick<FeedbackItem, "id" | "image_url" | "label" | "feedback_type">;

type LocalTraining = {
  id: string;
  modelName: string;
  startedAt: string;
  epochs: number;
  architecture: string;
  knownModelIds?: string[];
  datasetVersionId?: string;
  baseModelVersionId?: string | null;
  recipeId?: string;
};

function formatKoreaTimestamp(date = new Date()) {
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

function readLocalDeployment(): Partial<DeploymentState> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(LOCAL_DEPLOYMENT_KEY);
    return raw ? (JSON.parse(raw) as Partial<DeploymentState>) : {};
  } catch {
    return {};
  }
}

function writeLocalDeployment(update: Partial<DeploymentState>) {
  if (typeof window === "undefined") return;
  const next = { ...readLocalDeployment(), ...update };
  window.localStorage.setItem(LOCAL_DEPLOYMENT_KEY, JSON.stringify(next));
}

function readLocalDatasets(): DatasetVersion[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(LOCAL_DATASETS_KEY);
    return raw ? (JSON.parse(raw) as DatasetVersion[]) : [];
  } catch {
    return [];
  }
}

function writeLocalDatasets(datasets: DatasetVersion[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(LOCAL_DATASETS_KEY, JSON.stringify(datasets));
}

function readLocalRecipes(): TrainingRecipe[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(LOCAL_RECIPES_KEY);
    return raw ? (JSON.parse(raw) as TrainingRecipe[]) : [];
  } catch {
    return [];
  }
}

function writeLocalRecipes(recipes: TrainingRecipe[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(LOCAL_RECIPES_KEY, JSON.stringify(recipes));
}

function readLocalTraining(): LocalTraining | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(LOCAL_TRAINING_KEY);
    return raw ? (JSON.parse(raw) as LocalTraining) : null;
  } catch {
    return null;
  }
}

function readLocalModelNames(): Record<string, string> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(LOCAL_MODEL_NAMES_KEY);
    return raw ? (JSON.parse(raw) as Record<string, string>) : {};
  } catch {
    return {};
  }
}

function writeLocalModelNames(modelNames: Record<string, string>) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(LOCAL_MODEL_NAMES_KEY, JSON.stringify(modelNames));
}

function writeLocalTraining(training: LocalTraining | null) {
  if (typeof window === "undefined") return;
  if (!training) {
    window.localStorage.removeItem(LOCAL_TRAINING_KEY);
    return;
  }
  window.localStorage.setItem(LOCAL_TRAINING_KEY, JSON.stringify(training));
}

function slugify(value: string) {
  const slug = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return slug || "custom-recipe";
}

function recordMaterializedDataset(
  payload: {
    mode: "append" | "new";
    targetDatasetId?: string;
    datasetName?: string;
  },
  sampleCount: number
) {
  if (payload.mode !== "new") return;

  const id = `DATA-FEEDBACK-${new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 14)}`;
  const localDataset: DatasetVersion = {
    id,
    name: payload.datasetName || "Feedback Materialized Dataset",
    status: "prepared",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    source_dataset_id: payload.targetDatasetId ?? null,
    sample_count: sampleCount,
    feedback_count: sampleCount,
    notes: "Frontend fallback dataset generated from feedback images.",
    samples: [],
  };

  writeLocalDatasets([localDataset, ...readLocalDatasets().filter((dataset) => dataset.id !== id)]);
}

function createDeploymentFallback(dashboard: DashboardResponse): DeploymentState {
  const productionModel =
    dashboard.model_versions.find((model) => model.id === dashboard.runtime_config?.current_model_id) ??
    dashboard.model_versions.find((model) => model.status === "production") ??
    null;
  const stagingModel =
    dashboard.model_versions.find((model) => model.status === "staging") ??
    dashboard.model_versions.find((model) => model.status === "candidate") ??
    null;
  const canaryModel = dashboard.model_versions.find((model) => model.status === "canary") ?? null;

  return {
    production_model_id: productionModel?.id ?? null,
    staging_model_id: stagingModel?.id ?? null,
    canary_model_id: canaryModel?.id ?? null,
    canary_line: null,
    previous_production_model_id: null,
    last_action: null,
    last_action_at: null,
  };
}

function normalizeRuns(runs: TrainingRun[]) {
  const modelNames = readLocalModelNames();
  return runs.map((run) => {
    if (run.status === "completed") {
      return {
        ...run,
        name: modelNames[run.id] ?? run.name,
        progress: run.progress ?? 100,
        current_step: run.current_step ?? "COMPLETED",
      };
    }
    if (run.status) {
      return {
        ...run,
        name: modelNames[run.id] ?? run.name,
        progress: run.progress ?? 0,
        current_step: run.current_step ?? run.status,
      };
    }
    if (run.completed_at || run.final_metrics) {
      return {
        ...run,
        name: modelNames[run.id] ?? run.name,
        status: "completed",
        progress: 100,
        current_step: run.current_step ?? "COMPLETED",
      };
    }
    return {
      ...run,
      name: modelNames[run.id] ?? run.name,
      status: "configured",
      progress: run.progress ?? 0,
      current_step: run.current_step ?? "configured",
    };
  });
}

function hasCompletedAfterLocalTraining(runs: TrainingRun[], training: LocalTraining) {
  const startedAt = new Date(training.startedAt).getTime();
  return runs.some((run) => {
    if (run.status !== "completed" || !run.completed_at) return false;
    return new Date(run.completed_at).getTime() >= startedAt;
  });
}

function createOptimisticTrainingRun(training: LocalTraining | null): TrainingRun | null {
  if (!training) return null;
  const elapsedMs = Date.now() - new Date(training.startedAt).getTime();
  const durationMs = Math.max(1, training.epochs) * 1000;
  if (elapsedMs > durationMs + 5000) {
    writeLocalTraining(null);
    return null;
  }

  const progress = Math.min(95, Math.max(5, Math.round((elapsedMs / durationMs) * 100)));
  const epoch = Math.min(training.epochs, Math.max(1, Math.ceil((progress / 100) * training.epochs)));
  return {
    id: training.id,
    name: training.modelName,
    status: "running",
    dataset_version_id: training.datasetVersionId,
    base_model_version_id: training.baseModelVersionId,
    recipe_id: training.recipeId,
    architecture: training.architecture,
    progress,
    current_step: `TRAINING EPOCH ${epoch}`,
    epochs: training.epochs,
    started_at: training.startedAt,
    device: "backend",
  };
}

function applyLocalTrainingModelName(models: ModelVersion[]) {
  const localTraining = readLocalTraining();
  const modelNames = readLocalModelNames();

  let applied = false;
  return models.map((model, index) => {
    if (modelNames[model.id]) {
      return {
        ...model,
        name: modelNames[model.id],
      };
    }

    if (!localTraining?.modelName) return model;
    const knownModelIds = new Set(localTraining.knownModelIds ?? []);

    const candidate =
      !applied &&
      (model.id === localTraining.id ||
        (index === 0 && model.status !== "production" && !knownModelIds.has(model.id)) ||
        (model.status !== "production" &&
          !knownModelIds.has(model.id) &&
          /new |model-run|retrained/i.test(`${model.name} ${model.id} ${model.lineage ?? ""}`)));

    if (!candidate) return model;
    applied = true;
    writeLocalModelNames({
      ...modelNames,
      [model.id]: localTraining.modelName,
    });
    return {
      ...model,
      name: localTraining.modelName,
    };
  });
}

function mergeTrainingStatus(runs: TrainingRun[], trainingStatus: BackendTrainingStatus | null): TrainingRun[] {
  const normalizedRuns = normalizeRuns(runs);

  if (!trainingStatus?.is_running) {
    const localTraining = readLocalTraining();
    if (localTraining && hasCompletedAfterLocalTraining(normalizedRuns, localTraining)) {
      writeLocalTraining(null);
      return normalizedRuns;
    }
    const localRun = createOptimisticTrainingRun(localTraining);
    if (!localRun) return normalizedRuns;
    const alreadyCompleted = normalizedRuns.some(
      (run) => run.started_at === localRun.started_at || run.id === localRun.id
    );
    return alreadyCompleted ? normalizedRuns : [localRun, ...normalizedRuns];
  }

  const activeIndex = normalizedRuns.findIndex((run) => ["preparing", "running", "stopping"].includes(run.status ?? ""));
  const existing = activeIndex >= 0 ? normalizedRuns[activeIndex] : null;
  const localTraining = readLocalTraining();
  const activeRun: TrainingRun = {
    ...existing,
    id: existing?.id ?? localTraining?.id ?? "LIVE-TRAINING",
    status: "running",
    progress: Math.min(100, Math.max(0, trainingStatus.progress ?? 0)),
    current_step: trainingStatus.message ?? "TRAINING",
    final_metrics: trainingStatus.metrics,
    epochs: trainingStatus.epoch,
    architecture: existing?.architecture ?? localTraining?.architecture ?? "ARCH-GATE-EFF",
    dataset_version_id: existing?.dataset_version_id ?? localTraining?.datasetVersionId,
    base_model_version_id: existing?.base_model_version_id ?? localTraining?.baseModelVersionId,
    recipe_id: existing?.recipe_id ?? localTraining?.recipeId,
    started_at: existing?.started_at ?? localTraining?.startedAt,
    device: existing?.device ?? "backend",
  };

  if (activeIndex < 0) return [activeRun, ...normalizedRuns];
  return normalizedRuns.map((run, index) => (index === activeIndex ? activeRun : run));
}

function normalizeDashboard(dashboard: DashboardResponse, trainingStatus: BackendTrainingStatus | null): DashboardResponse {
  const backendDeployment = dashboard.deployment ?? createDeploymentFallback(dashboard);
  const localDeployment = readLocalDeployment();
  const deployment: DeploymentState = {
    ...backendDeployment,
    ...localDeployment,
    production_model_id: localDeployment.production_model_id ?? backendDeployment.production_model_id,
  };

  const modelVersions = applyLocalTrainingModelName(dashboard.model_versions).map((model) => {
    if (deployment.production_model_id === model.id) return { ...model, status: "production" };
    if (deployment.canary_model_id === model.id) return { ...model, status: "canary" };
    if (deployment.staging_model_id === model.id && model.status === "candidate") {
      return { ...model, status: "staging" };
    }
    return model;
  });
  const localDatasets = readLocalDatasets();
  const localDatasetIds = new Set(localDatasets.map((dataset) => dataset.id));
  const datasetVersions = [
    ...localDatasets,
    ...(dashboard.dataset_versions ?? []).filter((dataset) => !localDatasetIds.has(dataset.id)),
  ];
  const localRecipes = readLocalRecipes();
  const localRecipeIds = new Set(localRecipes.map((recipe) => recipe.id));
  const backendRecipes = dashboard.training_recipes?.length ? dashboard.training_recipes : DEFAULT_RECIPES;
  const trainingRecipes = [
    ...localRecipes,
    ...backendRecipes.filter((recipe) => !localRecipeIds.has(recipe.id)),
  ];
  const interfaces =
    dashboard.interfaces && Object.keys(dashboard.interfaces).length
      ? dashboard.interfaces
      : DEFAULT_INTERFACES;

  return {
    ...dashboard,
    dataset_versions: datasetVersions,
    training_recipes: trainingRecipes,
    training_runs: mergeTrainingStatus(dashboard.training_runs ?? [], trainingStatus),
    model_versions: modelVersions,
    deployment,
    interfaces,
  };
}

export async function fetchDashboard() {
  const [dashboard, trainingStatus] = await Promise.all([
    request<DashboardResponse>("/mlops/dashboard"),
    fetchTrainingStatus().catch(() => null),
  ]);

  return normalizeDashboard(dashboard, trainingStatus);
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

function feedbackItemToFile(item: FeedbackMaterializeItem) {
  const content = JSON.stringify({
    id: item.id,
    label: item.label,
    feedback_type: item.feedback_type,
  });
  return new File([content], `${item.id}.json`, {
    type: "application/json",
  });
}

export async function materializeFeedbackDataset(payload: {
  mode: "append" | "new";
  targetDatasetId?: string;
  datasetName?: string;
  feedbackItemIds?: string[];
  feedbackItems?: FeedbackMaterializeItem[];
}) {
  if (!payload.feedbackItems?.length) {
    return await request("/mlops/datasets/from-feedback", {
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

  const files = payload.feedbackItems.map(feedbackItemToFile);
  const response = await uploadDatasetFiles({
    files,
    label: "feedback",
    sourceType: "feedback_materialized",
    line: "",
    comment: `Materialized ${files.length} feedback items`,
    datasetMode: payload.mode,
    datasetVersionId: payload.targetDatasetId,
    datasetName: payload.datasetName,
  });
  recordMaterializedDataset(payload, files.length);
  return response;
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
  modelName?: string;
  knownModelIds?: string[];
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
  const startedAt = new Date().toISOString();
  const modelName = payload.modelName?.trim() || formatKoreaTimestamp();
  const architecture = payload.architecture || payload.gateArchitectureId || "ARCH-GATE-EFF";
  const epochs = payload.epochs || 10;

  return request<{ message?: string; training_run?: unknown; model_version?: unknown; config?: unknown }>("/mlops/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_name: modelName,
      architecture,
      epochs,
      batch_size: payload.batchSize || 32,
      learning_rate: payload.learningRate || 0.001,
      optimizer: payload.optimizer || "Adam",
      augmentation: payload.augmentation !== undefined ? payload.augmentation : true,
    }),
  }).then((response) => {
    writeLocalTraining({
      id: `LIVE-${startedAt.replace(/[-:TZ.]/g, "").slice(0, 14)}`,
      modelName,
      startedAt,
      epochs,
      architecture,
      knownModelIds: payload.knownModelIds,
      datasetVersionId: payload.datasetVersionId,
      baseModelVersionId: payload.baseModelVersionId,
      recipeId: payload.recipeId,
    });
    return response;
  });
}

export function fetchTrainingStatus() {
  return request<BackendTrainingStatus>("/mlops/training/status");
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

export async function promoteModel(
  modelVersionId: string,
  targetStatus: string,
  gateFile?: string,
  heatmapFile?: string,
  ensembleEnabled = true
) {
  if (targetStatus !== "production") {
    writeLocalDeployment({
      staging_model_id: modelVersionId,
      last_action: "stage",
      last_action_at: new Date().toISOString(),
    });
    return { model_version: { id: modelVersionId, status: targetStatus } as ModelVersion };
  }

  const previous = readLocalDeployment().production_model_id ?? null;
  const response = await deployModel({
    model_id: modelVersionId,
    gate_file: gateFile,
    heatmap_file: heatmapFile,
    ensemble_enabled: ensembleEnabled,
  });
  writeLocalDeployment({
    production_model_id: modelVersionId,
    previous_production_model_id: previous,
    staging_model_id: null,
    canary_model_id: null,
    canary_line: null,
    last_action: "promote",
    last_action_at: new Date().toISOString(),
  });
  return response;
}

export async function startCanary(modelVersionId: string, line?: string) {
  writeLocalDeployment({
    canary_model_id: modelVersionId,
    canary_line: line || "LINE-B",
    last_action: "canary",
    last_action_at: new Date().toISOString(),
  });
  return { model_version: { id: modelVersionId, status: "canary" } as ModelVersion, local_only: true };
}

export async function rollbackDeployment(
  modelVersionId?: string,
  gateFile?: string,
  heatmapFile?: string,
  ensembleEnabled = true
) {
  const rollbackModelId = modelVersionId || readLocalDeployment().previous_production_model_id || "MODEL-R3-FINAL";
  const response = await deployModel({
    model_id: rollbackModelId,
    gate_file: gateFile,
    heatmap_file: heatmapFile,
    ensemble_enabled: ensembleEnabled,
  });
  writeLocalDeployment({
    production_model_id: rollbackModelId,
    canary_model_id: null,
    canary_line: null,
    last_action: "rollback",
    last_action_at: new Date().toISOString(),
  });
  return response;
}

export function stopTrainingRun(runId?: string) {
  void runId;
  return request("/mlops/training/status");
}

export function saveTrainingRecipe(payload: TrainingRecipe) {
  const recipe: TrainingRecipe = {
    ...payload,
    id: `${slugify(payload.name)}-${new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 14)}`,
    source: "custom",
    default_epochs: payload.default_epochs ?? payload.epochs ?? 3,
  };
  const recipes = readLocalRecipes();
  writeLocalRecipes([recipe, ...recipes.filter((item) => item.id !== recipe.id)]);
  return Promise.resolve({ recipe });
}

export function deleteTrainingRun(runId: string) {
  return request(`/mlops/training/runs/${runId}`, { method: "DELETE" });
}
