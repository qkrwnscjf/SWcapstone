import { useEffect, useState } from "react";
import { CheckCircle2, Clock3, Cpu, ShieldAlert } from "lucide-react";
import { predictAnomaly } from "../api/anomaly";
import { uploadFeedback } from "../api/mlops";
import type { PredictResponse } from "../types/anomaly";
import type { DashboardResponse } from "../types/mlops";
import type { LineId, Tone, ViewMode } from "../app/types";
import { clamp01, formatMetric, formatPercent, pickCanaryModel, pickProductionModel, toDisplayDecision, toIssueType } from "../app/utils";
import { Badge, DarkCard, MessageBanner, Stat } from "../components/ui";
import { cls } from "../app/utils";

function ActionButton({
  title,
  tone,
  disabled,
  onClick,
}: {
  title: string;
  tone: Tone;
  disabled?: boolean;
  onClick: () => void;
}) {
  const toneMap: Record<Tone, string> = {
    blue: "border-sky-950 bg-sky-900 text-sky-50 hover:bg-sky-800",
    red: "border-rose-950 bg-rose-900 text-rose-50 hover:bg-rose-800",
    amber: "border-amber-950 bg-amber-800 text-amber-50 hover:bg-amber-700",
    green: "border-emerald-950 bg-emerald-900 text-emerald-50 hover:bg-emerald-800",
    slate: "border-slate-700 bg-slate-800 text-slate-100 hover:bg-slate-700",
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cls(
        "min-h-[70px] rounded-2xl border-2 px-4 text-left text-base font-black transition disabled:cursor-not-allowed disabled:opacity-50",
        toneMap[tone]
      )}
    >
      {title}
    </button>
  );
}

function ModeSwitch({ mode, setMode }: { mode: ViewMode; setMode: (mode: ViewMode) => void }) {
  return (
    <div className="grid h-full grid-cols-3 gap-3">
      {(["raw", "heatmap", "overlay"] as ViewMode[]).map((value) => (
        <button
          key={value}
          onClick={() => setMode(value)}
          className={cls(
            "rounded-2xl border-2 text-sm font-black transition",
            mode === value ? "border-cyan-400 bg-cyan-400 text-slate-950" : "border-slate-800 bg-slate-950 text-slate-300"
          )}
        >
          {value.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

function CameraViewport({
  mode,
  rawImageUrl,
  overlayImageUrl,
  heatmapImageUrl,
}: {
  mode: ViewMode;
  rawImageUrl: string;
  overlayImageUrl: string;
  heatmapImageUrl: string;
}) {
  const imageSrc = mode === "heatmap" && heatmapImageUrl ? heatmapImageUrl : rawImageUrl;
  const showOverlay = mode === "overlay" && rawImageUrl && overlayImageUrl;

  return (
    <div className="relative h-full min-h-0 overflow-hidden rounded-2xl border-2 border-slate-800 bg-[#040b17]">
      {!imageSrc ? (
        <div className="absolute inset-0 grid place-items-center text-base font-bold text-slate-400">
          이미지 업로드 후 검사 결과가 표시됩니다.
        </div>
      ) : (
        <>
          <img src={imageSrc} alt="inspection view" className="h-full w-full bg-slate-950 object-contain" />
          {showOverlay ? (
            <img src={overlayImageUrl} alt="heatmap overlay" className="pointer-events-none absolute inset-0 h-full w-full object-contain opacity-80" />
          ) : null}
        </>
      )}
    </div>
  );
}

export function FieldPage({
  selectedLine,
  dashboard,
  onRefresh,
}: {
  selectedLine: LineId;
  dashboard: DashboardResponse;
  onRefresh: () => Promise<void>;
}) {
  const [mode, setMode] = useState<ViewMode>("raw");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [rawImageUrl, setRawImageUrl] = useState("");
  const [overlayImageUrl, setOverlayImageUrl] = useState("");
  const [heatmapImageUrl, setHeatmapImageUrl] = useState("");
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [message, setMessage] = useState("");

  const productionModel = pickProductionModel(dashboard);
  const canaryModel = pickCanaryModel(dashboard);
  const activeModelId =
    canaryModel && dashboard.deployment.canary_line === selectedLine ? canaryModel.id : productionModel?.id ?? "-";

  useEffect(() => {
    return () => {
      if (rawImageUrl.startsWith("blob:")) URL.revokeObjectURL(rawImageUrl);
    };
  }, [rawImageUrl]);

  const decision = predictResult ? toDisplayDecision(predictResult.decision) : "대기";
  const confidence = predictResult ? clamp01(predictResult.gate_score) : 0;
  const latency = predictResult ? Math.round(predictResult.latency.total_latency_ms) : 0;
  const heatmapScore = predictResult?.heatmap_score ?? null;

  async function handlePredict() {
    if (!selectedFile) {
      setMessage("먼저 이미지를 업로드하세요.");
      return;
    }

    try {
      setIsPredicting(true);
      setMessage("");
      const result = await predictAnomaly(selectedFile);
      setPredictResult(result);
      const overlayUrl = result.heatmap_overlay ? `data:image/png;base64,${result.heatmap_overlay}` : "";
      const heatmapUrl = result.normalized_score_heatmap ? `data:image/png;base64,${result.normalized_score_heatmap}` : overlayUrl;
      setOverlayImageUrl(overlayUrl);
      setHeatmapImageUrl(heatmapUrl);
      setMode(overlayUrl ? "overlay" : heatmapUrl ? "heatmap" : "raw");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "검사 요청에 실패했습니다.");
    } finally {
      setIsPredicting(false);
    }
  }

  async function handleFeedback(
    feedbackType: "false_positive" | "false_negative" | "confirmed_anomaly" | "needs_review",
    label: "normal" | "anomaly" | "unlabeled",
    comment: string
  ) {
    if (!selectedFile) {
      setMessage("피드백을 남기려면 검사 이미지를 먼저 선택하세요.");
      return;
    }

    try {
      await uploadFeedback({
        file: selectedFile,
        feedbackType,
        label,
        operator: "operator-01",
        comment,
        line: selectedLine,
        predictedLabel: predictResult?.decision ?? "",
        gateScore: predictResult?.gate_score,
        heatmapScore: predictResult?.heatmap_score,
      });
      setMessage("피드백이 저장되었습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "피드백 저장에 실패했습니다.");
    }
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-[1.2fr_0.8fr] gap-4">
      <DarkCard className="flex min-h-0 flex-col p-5">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <div className="text-2xl font-black text-slate-50">현장 검사</div>
            <div className="mt-1 text-base font-semibold text-slate-300">{selectedLine} · 모델 {activeModelId}</div>
          </div>
          <Badge tone={decision === "이상" ? "red" : decision === "정상" ? "green" : "slate"} dark>
            {decision}
          </Badge>
        </div>

        <div className="mb-4 grid grid-cols-4 gap-3">
          <Stat label="상태" value={isPredicting ? "검사 중" : decision} icon={CheckCircle2} tone={decision === "이상" ? "red" : "green"} dark />
          <Stat label="확률" value={formatPercent(confidence)} icon={Cpu} tone="blue" dark />
          <Stat label="응답" value={latency ? `${latency}ms` : "-"} icon={Clock3} tone="slate" dark />
          <Stat label="Heatmap" value={heatmapScore == null ? "-" : formatMetric(heatmapScore)} icon={ShieldAlert} tone="amber" dark />
        </div>

        <div className="mb-4 grid grid-cols-[1fr_auto] gap-3">
          <label className="flex cursor-pointer items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900 px-4 py-3 text-base font-bold text-slate-100 hover:bg-slate-800">
            이미지 업로드
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (!file) return;
                if (rawImageUrl.startsWith("blob:")) URL.revokeObjectURL(rawImageUrl);
                setSelectedFile(file);
                setPredictResult(null);
                setMessage("");
                setOverlayImageUrl("");
                setHeatmapImageUrl("");
                setRawImageUrl(URL.createObjectURL(file));
                setMode("raw");
              }}
            />
          </label>
          <button
            onClick={handlePredict}
            disabled={isPredicting}
            className="rounded-2xl border-2 border-cyan-400 bg-cyan-400 px-6 py-3 text-base font-black text-slate-950 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isPredicting ? "검사 중" : "검사 실행"}
          </button>
        </div>

        {message ? <MessageBanner message={message} tone="blue" dark /> : null}

        <div className="mb-4 mt-4 h-[56px]">
          <ModeSwitch mode={mode} setMode={setMode} />
        </div>

        <div className="min-h-0 flex-1">
          <CameraViewport mode={mode} rawImageUrl={rawImageUrl} overlayImageUrl={overlayImageUrl} heatmapImageUrl={heatmapImageUrl} />
        </div>
      </DarkCard>

      <DarkCard className="grid min-h-0 grid-rows-[auto_auto_1fr] gap-4 p-5">
        <div className="grid grid-cols-2 gap-3">
          <Stat label="라인" value={selectedLine} dark />
          <Stat label="모델" value={activeModelId} dark />
          <Stat label="판정" value={decision} dark />
          <Stat label="유형" value={toIssueType(predictResult)} dark />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <ActionButton title="오탐 저장" tone="blue" onClick={() => handleFeedback("false_positive", "normal", "오탐 저장")} />
          <ActionButton title="미탐 저장" tone="red" onClick={() => handleFeedback("false_negative", "anomaly", "미탐 저장")} />
          <ActionButton title="이상 확정" tone="amber" onClick={() => handleFeedback("confirmed_anomaly", "anomaly", "이상 확정")} />
          <ActionButton title="검토 보류" tone="slate" onClick={() => handleFeedback("needs_review", "unlabeled", "검토 보류")} />
        </div>

        <div className="min-h-0 overflow-auto rounded-2xl border-2 border-slate-800 bg-slate-950 p-4">
          <div className="mb-3 text-lg font-black text-slate-100">최근 피드백</div>
          <div className="space-y-3">
            {dashboard.feedback_items.slice(0, 4).map((item) => (
              <div key={item.id} className="rounded-2xl border border-slate-800 bg-slate-900 p-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-bold text-slate-100">{item.feedback_type}</div>
                  <div className="text-sm text-slate-400">{item.line ?? selectedLine}</div>
                </div>
                <div className="mt-2 text-sm text-slate-300">{item.comment || "메모 없음"}</div>
              </div>
            ))}
            {!dashboard.feedback_items.length ? <div className="text-base font-semibold text-slate-400">아직 저장된 피드백이 없습니다.</div> : null}
          </div>
        </div>
      </DarkCard>
    </div>
  );
}

