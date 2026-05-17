import { Cpu, Database, Flame, Rocket } from "lucide-react";
import { useEffect, useState } from "react";
import {
  createTrainingRun,
  promoteModel,
  rollbackDeployment,
  saveTrainingRecipe,
  startCanary,
  stopTrainingRun,
  uploadArchitecture,
} from "../api/mlops";
import { LINES } from "../app/constants";
import type { BusyAction, LineId } from "../app/types";
import { cls, formatKoreaTimestamp, pickCanaryModel, pickProductionModel, pickStagingModel, recipeDraftFrom } from "../app/utils";
import type { DashboardResponse, ModelVersion, TrainingRecipe } from "../types/mlops";
import { Badge, Card, MessageBanner, Stat } from "../components/ui";

function RequirementTag({ required }: { required: boolean }) {
  return (
    <span className={cls("rounded-full px-2 py-1 text-sm font-black", required ? "bg-blue-100 text-blue-700" : "bg-slate-200 text-slate-700")}>
      {required ? "필수" : "선택"}
    </span>
  );
}

function DeploymentCard({
  title,
  model,
  subtitle,
}: {
  title: string;
  model: ModelVersion | null;
  subtitle: string;
}) {
  return (
    <div className="min-w-0 rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
      <div className="text-sm font-black text-slate-500">{title}</div>
      <div className="mt-1 truncate text-base font-black text-slate-950">{model?.name ?? model?.id ?? "-"}</div>
      <div className="mt-1 text-sm font-semibold text-slate-600">{subtitle}</div>
      <div className="mt-2 text-sm font-bold text-slate-700">
        F1 {model?.metrics.f1 ?? "-"} / {model?.metrics.latency_ms ?? "-"}ms
      </div>
    </div>
  );
}

export function TrainDeployView({
  dashboard,
  onRefresh,
}: {
  dashboard: DashboardResponse;
  onRefresh: () => Promise<void>;
}) {
  const productionModel = pickProductionModel(dashboard);
  const stagingModel = pickStagingModel(dashboard);
  const canaryModel = pickCanaryModel(dashboard);
  const candidateModel = canaryModel ?? stagingModel ?? dashboard.model_versions[0] ?? null;
  const recipes = dashboard.training_recipes ?? [];
  const firstRecipe = recipes[0];
  const activeRun = dashboard.training_runs.find((run) => ["preparing", "running", "stopping"].includes(run.status ?? ""));
  const latestRun = activeRun ?? dashboard.training_runs[0] ?? null;

  const [busyAction, setBusyAction] = useState<BusyAction>(null);
  const [message, setMessage] = useState("");
  const [selectedDatasetId, setSelectedDatasetId] = useState(dashboard.active_dataset_id);
  const [selectedBaseModelId, setSelectedBaseModelId] = useState(productionModel?.id ?? dashboard.model_versions[0]?.id ?? "");
  const [selectedRecipeId, setSelectedRecipeId] = useState(firstRecipe?.id ?? "");
  const [selectedDeployModelId, setSelectedDeployModelId] = useState(candidateModel?.id ?? "");
  const [selectedCanaryLine, setSelectedCanaryLine] = useState<LineId>("LINE-B");
  const [modelName, setModelName] = useState(() => formatKoreaTimestamp());
  const [epochCount, setEpochCount] = useState(firstRecipe?.default_epochs ?? 3);
  const [recipeDraft, setRecipeDraft] = useState<TrainingRecipe>(() => recipeDraftFrom(firstRecipe));
  const [architectureKind, setArchitectureKind] = useState<"gate" | "heatmap">("gate");
  const [architectureName, setArchitectureName] = useState("");
  const [architectureFile, setArchitectureFile] = useState<File | null>(null);

  useEffect(() => {
    if (!dashboard.dataset_versions.some((dataset) => dataset.id === selectedDatasetId)) {
      setSelectedDatasetId(dashboard.active_dataset_id);
    }
  }, [dashboard.active_dataset_id, dashboard.dataset_versions, selectedDatasetId]);

  const selectedRecipe = recipes.find((recipe) => recipe.id === selectedRecipeId) ?? firstRecipe;
  const selectedDeployModel = dashboard.model_versions.find((model) => model.id === selectedDeployModelId) ?? candidateModel;
  const runProgress = Math.min(100, Math.max(0, latestRun?.progress ?? 0));
  const runTone = latestRun?.status === "completed" ? "green" : latestRun?.status === "failed" ? "red" : activeRun ? "amber" : "blue";

  function selectRecipe(recipe: TrainingRecipe) {
    setSelectedRecipeId(recipe.id);
    setRecipeDraft(recipeDraftFrom(recipe));
    setEpochCount(recipe.default_epochs ?? recipe.epochs ?? 3);
  }

  async function handleSaveRecipe() {
    try {
      setBusyAction("train");
      setMessage("");
      const saved = await saveTrainingRecipe({ ...recipeDraft, default_epochs: epochCount, epochs: epochCount });
      const recipe = (saved as { recipe?: TrainingRecipe }).recipe;
      if (recipe?.id) setSelectedRecipeId(recipe.id);
      setMessage("레시피를 저장했습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "레시피 저장에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleStartTraining() {
    if (!selectedDatasetId || !selectedBaseModelId || !selectedRecipeId) {
      setMessage("데이터셋, 기준 모델, 레시피를 선택하세요.");
      return;
    }

    try {
      setBusyAction("train");
      setMessage("");
      await createTrainingRun({
        modelName: modelName.trim() || formatKoreaTimestamp(),
        knownModelIds: dashboard.model_versions.map((model) => model.id),
        datasetVersionId: selectedDatasetId,
        baseModelVersionId: selectedBaseModelId,
        recipeId: selectedRecipeId,
        targetLine: selectedCanaryLine,
        epochs: epochCount,
        batchSize: recipeDraft.batch_size,
        learningRate: recipeDraft.learning_rate,
        optimizer: recipeDraft.optimizer,
      });
      setModelName(formatKoreaTimestamp());
      setMessage("학습을 시작했습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "학습 시작에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleStopTraining() {
    try {
      setBusyAction("stop");
      setMessage("");
      await stopTrainingRun(activeRun?.id);
      setMessage("중지 요청을 보냈습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "학습 중지에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleUploadArchitecture() {
    if (!architectureFile || !architectureName.trim()) return;
    try {
      setBusyAction("train");
      setMessage("");
      await uploadArchitecture({ file: architectureFile, kind: architectureKind, name: architectureName.trim() });
      setArchitectureName("");
      setArchitectureFile(null);
      setMessage("아키텍처를 등록했습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "아키텍처 등록에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleStartCanary() {
    if (!selectedDeployModel) {
      setMessage("Canary 모델을 선택하세요.");
      return;
    }
    try {
      setBusyAction("canary");
      setMessage("");
      await startCanary(selectedDeployModel.id, selectedCanaryLine);
      setMessage(`${selectedCanaryLine}에서 Canary를 시작했습니다.`);
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Canary 시작에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleApprove() {
    if (!selectedDeployModel) return;
    try {
      setBusyAction("approve");
      setMessage("");
      await promoteModel(selectedDeployModel.id, "production");
      setMessage("Production 배포가 완료되었습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "배포 승인에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleRollback() {
    try {
      setBusyAction("rollback");
      setMessage("");
      await rollbackDeployment();
      setMessage("이전 Production으로 롤백했습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "롤백에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  return (
    <div className="grid h-full min-h-0 grid-rows-[auto_1fr] gap-4">
      <div className="grid grid-cols-4 gap-4">
        <Stat label="학습 실행" value={`${dashboard.training_runs.length}건`} sub="전체 이력" icon={Flame} tone="blue" />
        <Stat label="배포 후보" value={`${dashboard.model_versions.filter((model) => model.status !== "production").length}건`} sub="staging / canary" icon={Rocket} tone="green" />
        <Stat label="선택 데이터" value={dashboard.active_dataset_id} sub="현재 기준" icon={Database} tone="amber" />
        <Stat label="Production" value={productionModel?.id ?? "-"} sub="운영 모델" icon={Cpu} />
      </div>

      <div className="grid min-h-0 grid-cols-[1.08fr_.92fr] gap-4">
        <Card className="flex min-h-0 flex-col p-5">
          <div className="mb-4 flex items-center justify-between">
            <div className="text-xl font-black text-slate-950">CPU 학습</div>
            <Badge tone={runTone}>{latestRun?.current_step ?? "ready"}</Badge>
          </div>

          {message ? <div className="mb-4"><MessageBanner message={message} tone="blue" /></div> : null}

          <div className="min-h-0 overflow-auto pr-1">
            <div className="grid grid-cols-4 gap-3">
              <label className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="flex items-center justify-between gap-2 text-sm font-black text-slate-500">
                  <span>Dataset</span><RequirementTag required />
                </div>
                <select value={selectedDatasetId} onChange={(event) => setSelectedDatasetId(event.target.value)} className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-bold">
                  {dashboard.dataset_versions.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name} · {dataset.sample_count}</option>)}
                </select>
              </label>
              <label className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="flex items-center justify-between gap-2 text-sm font-black text-slate-500">
                  <span>Base Model</span><RequirementTag required />
                </div>
                <select value={selectedBaseModelId} onChange={(event) => setSelectedBaseModelId(event.target.value)} className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-bold">
                  {dashboard.model_versions.map((model) => <option key={model.id} value={model.id}>{model.id} · {model.status}</option>)}
                </select>
              </label>
              <label className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="flex items-center justify-between gap-2 text-sm font-black text-slate-500">
                  <span>Model Name</span><RequirementTag required />
                </div>
                <input value={modelName} onChange={(event) => setModelName(event.target.value)} className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-bold" />
              </label>
              <label className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="flex items-center justify-between gap-2 text-sm font-black text-slate-500">
                  <span>Epoch</span><RequirementTag required />
                </div>
                <input type="number" min={1} max={200} value={epochCount} onChange={(event) => setEpochCount(Math.max(1, Number(event.target.value)))} className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-bold" />
              </label>
            </div>

            <div className="mt-4 grid grid-cols-[.9fr_1.1fr] gap-4">
              <div className="max-h-72 overflow-auto rounded-2xl border-2 border-slate-200 bg-white p-3">
                <div className="mb-2 text-base font-black text-slate-950">레시피</div>
                <div className="space-y-2">
                  {recipes.map((recipe) => (
                    <button key={recipe.id} type="button" onClick={() => selectRecipe(recipe)} className={cls("w-full rounded-2xl border-2 p-3 text-left", recipe.id === selectedRecipeId ? "border-blue-600 bg-blue-50" : "border-slate-200")}>
                      <div className="text-base font-black text-slate-950">{recipe.name}</div>
                      <div className="mt-2 grid grid-cols-2 gap-1 text-sm font-semibold text-slate-600">
                        <span>batch {recipe.batch_size}</span><span>lr {recipe.learning_rate}</span><span>{recipe.optimizer}</span><span>{recipe.scheduler}</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="flex items-center justify-between gap-2 text-base font-black text-slate-950">
                  <span>레시피 수정</span><RequirementTag required={false} />
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <input className="rounded-xl border-2 border-slate-200 px-3 py-2 text-sm font-semibold" value={recipeDraft.name} onChange={(event) => setRecipeDraft({ ...recipeDraft, name: event.target.value })} placeholder="name" />
                  <select className="rounded-xl border-2 border-slate-200 px-3 py-2 text-sm font-semibold" value={recipeDraft.optimizer} onChange={(event) => setRecipeDraft({ ...recipeDraft, optimizer: event.target.value })}>
                    {["AdamW", "Adam", "SGD"].map((optimizer) => <option key={optimizer}>{optimizer}</option>)}
                  </select>
                  <input className="rounded-xl border-2 border-slate-200 px-3 py-2 text-sm font-semibold" type="number" min={1} value={recipeDraft.batch_size} onChange={(event) => setRecipeDraft({ ...recipeDraft, batch_size: Number(event.target.value) })} placeholder="batch" />
                  <input className="rounded-xl border-2 border-slate-200 px-3 py-2 text-sm font-semibold" type="number" step="0.0001" value={recipeDraft.learning_rate} onChange={(event) => setRecipeDraft({ ...recipeDraft, learning_rate: Number(event.target.value) })} placeholder="lr" />
                  <input className="rounded-xl border-2 border-slate-200 px-3 py-2 text-sm font-semibold" type="number" step="0.001" value={recipeDraft.weight_decay} onChange={(event) => setRecipeDraft({ ...recipeDraft, weight_decay: Number(event.target.value) })} placeholder="weight decay" />
                  <select className="rounded-xl border-2 border-slate-200 px-3 py-2 text-sm font-semibold" value={recipeDraft.scheduler} onChange={(event) => setRecipeDraft({ ...recipeDraft, scheduler: event.target.value })}>
                    {["cosine", "step", "none"].map((scheduler) => <option key={scheduler}>{scheduler}</option>)}
                  </select>
                </div>
                <button onClick={handleSaveRecipe} disabled={busyAction !== null} className="mt-3 w-full rounded-2xl border-2 border-slate-900 bg-slate-900 px-4 py-3 text-sm font-bold text-white disabled:opacity-60">
                  레시피 저장
                </button>
              </div>
            </div>

            <div className="mt-4 rounded-2xl border-2 border-slate-200 bg-white p-4">
              <div className="mb-3 flex items-center justify-between gap-4">
                <div className="text-base font-black text-slate-950">학습 상태</div>
                <Badge tone={runTone}>{latestRun?.status ?? "not-created"}</Badge>
              </div>
              <div className="h-3 overflow-hidden rounded-full bg-slate-200">
                <div className="h-full rounded-full bg-blue-600 transition-all" style={{ width: `${runProgress}%` }} />
              </div>
              <div className="mt-2 flex justify-between text-sm font-bold text-slate-600">
                <span>{runProgress}%</span><span>{latestRun?.device ?? "cpu"}</span><span>{latestRun?.name ?? selectedRecipe?.name ?? "-"}</span>
              </div>
            </div>

            <div className="mt-4 rounded-2xl border-2 border-slate-200 bg-white p-4">
              <div className="mb-3 text-base font-black text-slate-950">아키텍처 등록</div>
              <div className="grid grid-cols-[.6fr_1fr_1fr] gap-3">
                <select value={architectureKind} onChange={(event) => setArchitectureKind(event.target.value as "gate" | "heatmap")} className="rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-bold">
                  <option value="gate">gate</option><option value="heatmap">heatmap</option>
                </select>
                <input value={architectureName} onChange={(event) => setArchitectureName(event.target.value)} className="rounded-xl border-2 border-slate-200 px-3 py-2 text-sm font-bold" placeholder="Architecture name" />
                <input type="file" accept=".json,.yaml,.yml,.txt,.py" onChange={(event) => setArchitectureFile(event.target.files?.[0] ?? null)} className="rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm" />
              </div>
              <button onClick={handleUploadArchitecture} disabled={busyAction !== null || !architectureFile || !architectureName.trim()} className="mt-3 w-full rounded-2xl border-2 border-blue-600 bg-blue-600 px-4 py-3 text-sm font-bold text-white disabled:opacity-60">
                아키텍처 등록
              </button>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <button onClick={handleStartTraining} disabled={busyAction !== null || Boolean(activeRun)} className="rounded-2xl border-2 border-blue-600 bg-blue-600 px-4 py-3 text-sm font-bold text-white disabled:opacity-60">
                {busyAction === "train" ? "학습 시작 중" : "학습 시작"}
              </button>
              <button onClick={handleStopTraining} disabled={busyAction !== null || !activeRun} className="rounded-2xl border-2 border-rose-600 bg-rose-600 px-4 py-3 text-sm font-bold text-white disabled:opacity-60">
                {busyAction === "stop" ? "중지 요청 중" : "학습 중지"}
              </button>
            </div>
          </div>
        </Card>

        <Card className="flex min-h-0 flex-col p-5">
          <div className="mb-4 text-xl font-black text-slate-950">배포</div>
          <div className="min-h-0 overflow-auto pr-1">
            <div className="grid gap-3 rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
              <label>
                <div className="flex items-center justify-between gap-2 text-sm font-black text-slate-500">
                  <span>Model Candidate</span><RequirementTag required />
                </div>
                <select value={selectedDeployModelId} onChange={(event) => setSelectedDeployModelId(event.target.value)} className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-bold">
                  {dashboard.model_versions.map((model) => <option key={model.id} value={model.id}>{model.name ?? model.id} · {model.status}</option>)}
                </select>
              </label>
              <label>
                <div className="flex items-center justify-between gap-2 text-sm font-black text-slate-500">
                  <span>Canary Line</span><RequirementTag required={false} />
                </div>
                <select value={selectedCanaryLine} onChange={(event) => setSelectedCanaryLine(event.target.value as LineId)} className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-bold">
                  {LINES.map((line) => <option key={line}>{line}</option>)}
                </select>
              </label>
            </div>

            <div className="mt-4 grid gap-3">
              <div className="grid grid-cols-3 gap-3">
                <DeploymentCard title="Production" model={productionModel} subtitle="운영 중" />
                <DeploymentCard title="Staging" model={stagingModel} subtitle="검증 대기" />
                <DeploymentCard title="Canary" model={canaryModel} subtitle={dashboard.deployment.canary_line ? `${dashboard.deployment.canary_line} 검증 중` : "없음"} />
              </div>
              <div className="grid grid-cols-3 gap-3">
                <button onClick={handleStartCanary} disabled={busyAction !== null} className="rounded-2xl border-2 border-slate-200 px-4 py-3 text-sm font-bold text-slate-800 disabled:opacity-60">
                  {busyAction === "canary" ? "시작 중" : "Canary 시작"}
                </button>
                <button onClick={handleApprove} disabled={busyAction !== null} className="rounded-2xl border-2 border-blue-600 bg-blue-600 px-4 py-3 text-sm font-bold text-white disabled:opacity-60">
                  {busyAction === "approve" ? "승인 중" : "배포 승인"}
                </button>
                <button onClick={handleRollback} disabled={busyAction !== null} className="rounded-2xl border-2 border-rose-600 bg-rose-600 px-4 py-3 text-sm font-bold text-white disabled:opacity-60">
                  {busyAction === "rollback" ? "롤백 중" : "롤백"}
                </button>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

