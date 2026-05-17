import { useEffect, useState } from "react";
import { materializeFeedbackDataset, uploadDatasetFiles } from "../api/mlops";
import type { DashboardResponse } from "../types/mlops";
import { formatDate, modelTone } from "../app/utils";
import { Badge, Card, EmptyState, MessageBanner } from "../components/ui";

export function VersionView({
  dashboard,
  onRefresh,
}: {
  dashboard: DashboardResponse;
  onRefresh: () => Promise<void>;
}) {
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [targetDatasetId, setTargetDatasetId] = useState(dashboard.active_dataset_id);
  const [datasetMode, setDatasetMode] = useState<"append" | "new">("append");
  const [datasetName, setDatasetName] = useState("");
  const [uploadLabel, setUploadLabel] = useState("normal");
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);

  useEffect(() => {
    if (!dashboard.dataset_versions.some((dataset) => dataset.id === targetDatasetId)) {
      setTargetDatasetId(dashboard.active_dataset_id);
    }
  }, [dashboard.active_dataset_id, dashboard.dataset_versions, targetDatasetId]);

  async function handleFeedbackMaterialize(mode: "append" | "new") {
    try {
      setBusy(true);
      setMessage("");
      await materializeFeedbackDataset({
        mode,
        targetDatasetId,
        datasetName: mode === "new" ? datasetName : undefined,
        feedbackItemIds: dashboard.feedback_items.map((item) => item.id),
        feedbackItems: dashboard.feedback_items,
      });
      setMessage(mode === "new" ? "피드백으로 새 데이터셋을 만들었습니다." : "피드백을 기존 데이터셋에 추가했습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "피드백 데이터셋 처리에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  }

  async function handleUploadFiles() {
    if (!uploadFiles.length) {
      setMessage("추가할 파일을 선택하세요.");
      return;
    }

    try {
      setBusy(true);
      setMessage("");
      await uploadDatasetFiles({
        files: uploadFiles,
        label: uploadLabel,
        sourceType: "bulk_upload",
        line: "",
        comment: "Uploaded from version screen",
        datasetMode,
        datasetVersionId: targetDatasetId,
        datasetName,
      });
      setUploadFiles([]);
      setMessage(datasetMode === "new" ? "파일로 새 데이터셋을 만들었습니다." : "파일을 데이터셋에 추가했습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "데이터 파일 업로드에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-[1.05fr_.95fr] gap-4">
      <Card className="flex min-h-0 flex-col p-5">
        <div className="mb-4 flex items-center justify-between gap-4">
          <div className="text-xl font-black text-slate-950">데이터 버전</div>
          <Badge tone="blue">{dashboard.dataset_versions.length} versions</Badge>
        </div>

        {message ? <div className="mb-4"><MessageBanner message={message} tone="blue" /></div> : null}

        <div className="mb-4 rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
          <div className="grid grid-cols-2 gap-3">
            <label>
              <div className="text-sm font-black text-slate-500">Target Dataset</div>
              <select value={targetDatasetId} onChange={(event) => setTargetDatasetId(event.target.value)} className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-bold">
                {dashboard.dataset_versions.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name} · {dataset.sample_count}</option>)}
              </select>
            </label>
            <label>
              <div className="text-sm font-black text-slate-500">New Dataset Name</div>
              <input value={datasetName} onChange={(event) => setDatasetName(event.target.value)} className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-bold" placeholder="데이터셋 이름" />
            </label>
          </div>

          <div className="mt-3 grid grid-cols-2 gap-3">
            <button onClick={() => handleFeedbackMaterialize("append")} disabled={busy || !dashboard.feedback_items.length} className="rounded-2xl border-2 border-slate-200 bg-white px-4 py-3 text-sm font-bold text-slate-800 disabled:opacity-60">
              피드백 추가
            </button>
            <button onClick={() => handleFeedbackMaterialize("new")} disabled={busy || !dashboard.feedback_items.length} className="rounded-2xl border-2 border-blue-600 bg-blue-600 px-4 py-3 text-sm font-bold text-white disabled:opacity-60">
              피드백으로 새 데이터셋
            </button>
          </div>

          <div className="mt-4 grid grid-cols-[1fr_.7fr_.7fr] gap-3">
            <input type="file" multiple onChange={(event) => setUploadFiles(Array.from(event.target.files ?? []))} className="rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm text-slate-700" />
            <select value={uploadLabel} onChange={(event) => setUploadLabel(event.target.value)} className="rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-bold">
              <option value="normal">normal</option>
              <option value="anomaly">anomaly</option>
              <option value="unlabeled">unlabeled</option>
            </select>
            <select value={datasetMode} onChange={(event) => setDatasetMode(event.target.value as "append" | "new")} className="rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-bold">
              <option value="append">기존에 추가</option>
              <option value="new">새 데이터셋</option>
            </select>
          </div>
          <button onClick={handleUploadFiles} disabled={busy || !uploadFiles.length} className="mt-3 w-full rounded-2xl border-2 border-slate-900 bg-slate-900 px-4 py-3 text-sm font-bold text-white disabled:opacity-60">
            파일 반영
          </button>
        </div>

        <div className="min-h-0 overflow-auto pr-1">
          <div className="space-y-3">
            {dashboard.dataset_versions.map((dataset) => (
              <div key={dataset.id} className="rounded-2xl border-2 border-slate-200 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-base font-black text-slate-900">{dataset.name}</div>
                    <div className="mt-1 text-sm font-semibold text-slate-500">{dataset.id}</div>
                  </div>
                  <Badge tone="blue">{dataset.status}</Badge>
                </div>
                <div className="mt-3 grid grid-cols-3 gap-3 text-sm font-semibold text-slate-700">
                  <div>샘플 {dataset.sample_count}</div>
                  <div>피드백 {dataset.feedback_count}</div>
                  <div>원본 {dataset.source_dataset_id ?? "-"}</div>
                </div>
                <div className="mt-2 text-sm font-semibold text-slate-500">{formatDate(dataset.updated_at)}</div>
              </div>
            ))}
            {!dashboard.dataset_versions.length ? <EmptyState>데이터셋 버전이 없습니다.</EmptyState> : null}
          </div>
        </div>
      </Card>

      <Card className="flex min-h-0 flex-col p-5">
        <div className="mb-4 flex items-center justify-between gap-4">
          <div className="text-xl font-black text-slate-950">모델 버전</div>
          <Badge tone="green">{dashboard.model_versions.length} models</Badge>
        </div>

        <div className="min-h-0 overflow-auto pr-1">
          <div className="space-y-3">
            {dashboard.model_versions.map((model) => (
              <div key={model.id} className="rounded-2xl border-2 border-slate-200 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-base font-black text-slate-900">{model.id}</div>
                    <div className="mt-1 text-sm font-semibold text-slate-500">{model.name}</div>
                  </div>
                  <Badge tone={modelTone(model)}>{model.status}</Badge>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3 text-sm font-semibold text-slate-700">
                  <div>Data {model.dataset_version_id ?? "-"}</div>
                  <div>Base {model.base_model_version_id ?? "-"}</div>
                  <div>Recipe {model.recipe_id ?? "-"}</div>
                  <div>F1 {model.metrics.f1 ?? "-"}</div>
                </div>
                <div className="mt-2 text-sm font-semibold text-slate-500">{formatDate(model.updated_at ?? model.created_at)}</div>
              </div>
            ))}
            {!dashboard.model_versions.length ? <EmptyState>모델 버전이 없습니다.</EmptyState> : null}
          </div>
        </div>
      </Card>
    </div>
  );
}

