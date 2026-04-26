import React, { useEffect, useState } from "react";
import type { LucideIcon } from "lucide-react";
import {
  AlertTriangle,
  Briefcase,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Cpu,
  Database,
  Factory,
  FileText,
  Flame,
  Layers,
  MonitorCog,
  Rocket,
  ShieldAlert,
} from "lucide-react";

import { predictAnomaly } from "./api/anomaly";
import {
  createTrainingRun,
  fetchDashboard,
  materializeFeedbackDataset,
  promoteModel,
  rollbackDeployment,
  saveTrainingRecipe,
  startCanary,
  stopTrainingRun,
  uploadFeedback,
  uploadDatasetFiles,
} from "./api/mlops";
import type { PredictResponse } from "./types/anomaly";
import type { DashboardResponse, ModelVersion, TrainingRecipe, TrainingRun } from "./types/mlops";

type Audience = "field" | "admin" | "summary";
type AdminTab = "ops" | "train" | "version" | "logs";
type LineId = "LINE-A" | "LINE-B" | "LINE-C";
type Tone = "slate" | "blue" | "green" | "amber" | "red";
type ViewMode = "raw" | "heatmap" | "overlay";
type BusyAction = "train" | "stop" | "canary" | "approve" | "rollback" | null;

type LiveSample = {
  sampleId: string;
  camera: string;
  label: string;
  confidence: number;
  inferMs: number;
  status: string;
  issueType: string;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const LINES: LineId[] = ["LINE-A", "LINE-B", "LINE-C"];
const CHART_BARS = [34, 48, 44, 57, 51, 63, 54, 46, 42, 58, 61, 49];

const audienceItems: Array<{ key: Audience; label: string; desc: string; icon: LucideIcon }> = [
  { key: "field", label: "현장", desc: "작업자 검사 HMI", icon: Factory },
  { key: "admin", label: "전산", desc: "운영 / 배포", icon: MonitorCog },
  { key: "summary", label: "요약", desc: "지표 / 상태", icon: Briefcase },
];

const adminTabs: Array<{ key: AdminTab; label: string; icon: LucideIcon }> = [
  { key: "ops", label: "운영", icon: MonitorCog },
  { key: "train", label: "학습·배포", icon: Rocket },
  { key: "version", label: "버전", icon: Layers },
  { key: "logs", label: "로그", icon: FileText },
];

const liveSamplesByLine: Record<LineId, LiveSample[]> = {
  "LINE-A": [
    {
      sampleId: "SMP-A1-1042",
      camera: "CAM-01",
      label: "정상",
      confidence: 0.97,
      inferMs: 41,
      status: "정상",
      issueType: "-",
    },
  ],
  "LINE-B": [
    {
      sampleId: "SMP-B3-2288",
      camera: "CAM-03",
      label: "주의",
      confidence: 0.61,
      inferMs: 96,
      status: "주의",
      issueType: "입력 품질 변동",
    },
  ],
  "LINE-C": [
    {
      sampleId: "SMP-C2-0102",
      camera: "CAM-02",
      label: "정상",
      confidence: 0.95,
      inferMs: 51,
      status: "정상",
      issueType: "-",
    },
  ],
};

const baseLineStatus: Record<
  LineId,
  { state: string; yieldRate: number; latency: number; camera: string; alert: string }
> = {
  "LINE-A": { state: "정상", yieldRate: 99.1, latency: 46, camera: "6/6", alert: "없음" },
  "LINE-B": { state: "주의", yieldRate: 97.8, latency: 92, camera: "5/6", alert: "P95 지연" },
  "LINE-C": { state: "정상", yieldRate: 98.7, latency: 51, camera: "4/4", alert: "없음" },
};

function cls(...names: Array<string | false | null | undefined>) {
  return names.filter(Boolean).join(" ");
}

function clamp01(value: number | undefined | null) {
  if (value == null || Number.isNaN(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

function formatDate(value?: string | null) {
  if (!value) return "-";
  return new Date(value).toLocaleString("ko-KR", { timeZone: "Asia/Seoul" });
}

function formatPercent(value: number | undefined | null) {
  return `${Math.round(clamp01(value) * 100)}%`;
}

function formatMetric(value?: number | null, digits = 3) {
  if (value == null || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

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

function asset(path: string) {
  if (!path) return "";
  return path.startsWith("http") || path.startsWith("data:") ? path : `${API_BASE}${path}`;
}

function toDisplayDecision(decision?: string | null) {
  if (decision === "anomaly") return "이상";
  if (decision === "normal" || decision === "normal (heatmap)") return "정상";
  return decision ?? "-";
}

function toIssueType(result: PredictResponse | null) {
  if (!result) return "-";
  if (result.decision === "anomaly") return "heatmap 이상 감지";
  if (result.heatmap_called) return "heatmap 확인 완료";
  return "gate 정상 통과";
}

function pickProductionModel(dashboard: DashboardResponse | null) {
  if (!dashboard) return null;
  const productionId = dashboard.deployment.production_model_id;
  return (
    dashboard.model_versions.find((model) => model.id === productionId) ??
    dashboard.model_versions.find((model) => model.status === "production") ??
    null
  );
}

function pickStagingModel(dashboard: DashboardResponse | null) {
  if (!dashboard) return null;
  const stagingId = dashboard.deployment.staging_model_id;
  return (
    dashboard.model_versions.find((model) => model.id === stagingId) ??
    dashboard.model_versions.find((model) => model.status === "staging") ??
    null
  );
}

function pickCanaryModel(dashboard: DashboardResponse | null) {
  if (!dashboard) return null;
  const canaryId = dashboard.deployment.canary_model_id;
  return (
    dashboard.model_versions.find((model) => model.id === canaryId) ??
    dashboard.model_versions.find((model) => model.status === "canary") ??
    null
  );
}

function recipeDraftFrom(recipe?: TrainingRecipe): TrainingRecipe {
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

function Card({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cls(
        "rounded-3xl border-2 border-slate-200 bg-white shadow-[0_10px_28px_rgba(15,23,42,0.06)]",
        className
      )}
    >
      {children}
    </div>
  );
}

function DarkCard({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cls(
        "rounded-3xl border-2 border-slate-800 bg-[#08111f] shadow-[0_16px_36px_rgba(0,0,0,0.28)]",
        className
      )}
    >
      {children}
    </div>
  );
}

function Badge({
  children,
  tone = "slate",
  dark = false,
}: {
  children: React.ReactNode;
  tone?: Tone;
  dark?: boolean;
}) {
  const toneMap = dark
    ? {
        slate: "border-slate-700 bg-slate-900 text-slate-300",
        blue: "border-sky-800 bg-sky-950/40 text-sky-300",
        green: "border-emerald-800 bg-emerald-950/40 text-emerald-300",
        amber: "border-amber-800 bg-amber-950/40 text-amber-300",
        red: "border-rose-800 bg-rose-950/40 text-rose-300",
      }
    : {
        slate: "bg-slate-100 text-slate-700",
        blue: "bg-blue-100 text-blue-700",
        green: "bg-emerald-100 text-emerald-700",
        amber: "bg-amber-100 text-amber-700",
        red: "bg-rose-100 text-rose-700",
      };

  return (
    <span
      className={cls(
        "inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold",
        dark && "border-2",
        toneMap[tone]
      )}
    >
      {children}
    </span>
  );
}

function Stat({
  label,
  value,
  sub,
  icon: Icon,
  tone = "slate",
  dark = false,
  compact = false,
}: {
  label: string;
  value: string;
  sub?: string;
  icon?: LucideIcon;
  tone?: Tone;
  dark?: boolean;
  compact?: boolean;
}) {
  const toneMap = dark
    ? {
        slate: "border-slate-800 bg-slate-950 text-slate-100",
        blue: "border-sky-900 bg-sky-950/30 text-sky-200",
        green: "border-emerald-900 bg-emerald-950/30 text-emerald-200",
        amber: "border-amber-900 bg-amber-950/30 text-amber-200",
        red: "border-rose-900 bg-rose-950/30 text-rose-200",
      }
    : {
        slate: "bg-white",
        blue: "border-blue-100 bg-blue-50",
        green: "border-emerald-100 bg-emerald-50",
        amber: "border-amber-100 bg-amber-50",
        red: "border-rose-100 bg-rose-50",
      };

  return (
    <div className={cls("min-w-0 rounded-2xl border-2", compact ? "px-3.5 py-2.5" : "p-5", toneMap[tone])}>
      <div className="flex h-full items-start justify-between gap-2">
        <div className="min-w-0">
          <div className={cls("font-bold tracking-[0.12em] text-slate-500", compact ? "text-[9px]" : "text-[11px]")}>
            {label}
          </div>
          <div className={cls("mt-2 truncate font-black leading-none", compact ? "text-[20px]" : "text-3xl")}>
            {value}
          </div>
          {!compact && sub ? (
            <div className={cls("mt-1 truncate text-[10px] leading-tight", dark ? "text-slate-400" : "text-slate-500")}>
              {sub}
            </div>
          ) : null}
        </div>
        {Icon ? <Icon className={cls("shrink-0 text-slate-500", compact ? "mt-0.5 h-4 w-4" : "mt-1 h-5 w-5")} /> : null}
      </div>
    </div>
  );
}

function AudienceSwitch({
  audience,
  setAudience,
}: {
  audience: Audience;
  setAudience: (audience: Audience) => void;
}) {
  return (
    <div className="inline-flex h-[58px] rounded-[22px] border-[3px] border-slate-300 bg-white p-1.5 shadow-[0_10px_26px_rgba(15,23,42,0.08)]">
      {audienceItems.map((item) => {
        const active = audience === item.key;
        const Icon = item.icon;
        return (
          <button
            key={item.key}
            onClick={() => setAudience(item.key)}
            className={cls(
              "flex h-full min-w-[160px] items-center gap-3 rounded-[16px] border-2 border-transparent px-5 font-semibold transition",
              active ? "bg-blue-600 text-white shadow-[0_8px_18px_rgba(37,99,235,0.35)]" : "text-slate-700 hover:bg-slate-50"
            )}
          >
            <div
              className={cls(
                "grid h-9 w-9 place-items-center rounded-2xl border-2",
                active ? "border-white/10 bg-white/15" : "border-transparent bg-slate-100"
              )}
            >
              <Icon className="h-5 w-5" />
            </div>
            <div className="text-left leading-tight">
              <div className="text-sm font-bold">{item.label}</div>
              <div className={cls("text-[11px]", active ? "text-blue-100" : "text-slate-500")}>{item.desc}</div>
            </div>
          </button>
        );
      })}
    </div>
  );
}

function ContextControls({
  audience,
  adminTab,
  setAdminTab,
  selectedLine,
  setSelectedLine,
}: {
  audience: Audience;
  adminTab: AdminTab;
  setAdminTab: (tab: AdminTab) => void;
  selectedLine: LineId;
  setSelectedLine: (line: LineId) => void;
}) {
  const [lineOpen, setLineOpen] = useState(false);

  if (audience === "admin") {
    return (
      <div className="inline-flex h-[58px] rounded-[22px] border-2 border-slate-200 bg-white p-1.5 shadow-[0_10px_26px_rgba(15,23,42,0.06)]">
        {adminTabs.map((item) => {
          const active = adminTab === item.key;
          const Icon = item.icon;
          return (
            <button
              key={item.key}
              onClick={() => setAdminTab(item.key)}
              className={cls(
                "flex h-full items-center gap-2 rounded-[16px] border-2 border-transparent px-4 text-sm font-semibold transition",
                active ? "bg-blue-600 text-white" : "text-slate-700 hover:bg-slate-50"
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </button>
          );
        })}
      </div>
    );
  }

  if (audience === "field") {
    return (
      <div className="relative">
        <button
          onClick={() => setLineOpen((value) => !value)}
          className="inline-flex h-[58px] items-center gap-2 rounded-[22px] border-2 border-slate-200 bg-white px-4 text-sm font-semibold text-slate-700 shadow-[0_10px_26px_rgba(15,23,42,0.08)]"
        >
          <Factory className="h-4 w-4" />
          {selectedLine}
          <ChevronDown className={cls("h-4 w-4 transition", lineOpen && "rotate-180")} />
        </button>

        {lineOpen ? (
          <div className="absolute left-0 top-[64px] z-30 w-[180px] rounded-2xl border-2 border-slate-200 bg-white p-2 shadow-[0_12px_30px_rgba(15,23,42,0.12)]">
            {LINES.map((line) => (
              <button
                key={line}
                onClick={() => {
                  setSelectedLine(line);
                  setLineOpen(false);
                }}
                className={cls(
                  "flex w-full items-center rounded-xl border-2 border-transparent px-3 py-2.5 text-left text-sm font-semibold transition",
                  selectedLine === line ? "bg-blue-50 text-blue-700" : "text-slate-700 hover:bg-slate-50"
                )}
              >
                {line}
              </button>
            ))}
          </div>
        ) : null}
      </div>
    );
  }

  return <div className="h-[58px]" />;
}

function TopBar({
  audience,
  setAudience,
  adminTab,
  setAdminTab,
  selectedLine,
  setSelectedLine,
}: {
  audience: Audience;
  setAudience: (audience: Audience) => void;
  adminTab: AdminTab;
  setAdminTab: (tab: AdminTab) => void;
  selectedLine: LineId;
  setSelectedLine: (line: LineId) => void;
}) {
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/95 px-7 py-4 backdrop-blur-xl">
      <div className="flex items-center justify-between gap-6">
        <div className="flex h-[58px] items-center text-[22px] font-black tracking-tight text-slate-900">
          (주)철강왕 MLOps Console
        </div>

        <div className="flex items-center gap-4">
          <ContextControls
            audience={audience}
            adminTab={adminTab}
            setAdminTab={setAdminTab}
            selectedLine={selectedLine}
            setSelectedLine={setSelectedLine}
          />
          <AudienceSwitch audience={audience} setAudience={setAudience} />
        </div>
      </div>
    </header>
  );
}

function ActionButton({
  title,
  desc,
  tone,
  disabled = false,
  onClick,
}: {
  title: string;
  desc: string;
  tone: Tone;
  disabled?: boolean;
  onClick: () => void;
}) {
  const toneMap: Record<Tone, string> = {
    blue: "border-sky-950 bg-sky-900/80 text-sky-50 hover:bg-sky-900",
    red: "border-rose-950 bg-rose-900/80 text-rose-50 hover:bg-rose-900",
    amber: "border-amber-950 bg-amber-800/90 text-amber-50 hover:bg-amber-800",
    green: "border-emerald-950 bg-emerald-900/80 text-emerald-50 hover:bg-emerald-900",
    slate: "border-slate-700 bg-slate-800 text-slate-100 hover:bg-slate-700",
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cls(
        "flex min-h-[96px] w-full flex-col overflow-hidden rounded-2xl border-2 px-4 py-4 text-left transition active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50",
        toneMap[tone]
      )}
    >
      <div className="break-keep text-[17px] font-black leading-tight tracking-tight">{title}</div>
      <div className="mt-auto break-keep pt-3 text-[12px] leading-[1.35] opacity-90">{desc}</div>
    </button>
  );
}

function ModeSwitch({
  mode,
  setMode,
}: {
  mode: ViewMode;
  setMode: (mode: ViewMode) => void;
}) {
  return (
    <div className="grid h-full grid-cols-3 gap-3">
      {(["raw", "heatmap", "overlay"] as ViewMode[]).map((value) => (
        <button
          key={value}
          onClick={() => setMode(value)}
          className={cls(
            "h-full rounded-2xl border-2 text-sm font-black tracking-wide transition",
            mode === value
              ? "border-cyan-400 bg-cyan-400 text-slate-950 shadow-[0_10px_22px_rgba(34,211,238,0.22)]"
              : "border-slate-800 bg-slate-950 text-slate-300 hover:bg-slate-900"
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
    <div className="relative h-full min-h-0 overflow-hidden rounded-[22px] border-2 border-slate-800 bg-[#040b17]">
      {!imageSrc ? (
        <>
          <div
            className="absolute inset-0 opacity-20"
            style={{
              backgroundImage:
                "linear-gradient(rgba(148,163,184,0.12) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.12) 1px, transparent 1px)",
              backgroundSize: "26px 26px",
            }}
          />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(30,41,59,0.32),rgba(2,6,23,0.94))]" />
          <div className="absolute inset-0 grid place-items-center text-sm text-slate-400">
            이미지를 업로드하면 검사 화면이 표시됩니다.
          </div>
        </>
      ) : (
        <>
          <img src={imageSrc} alt="inspection view" className="h-full w-full object-contain bg-slate-950" />
          {showOverlay ? (
            <img
              src={overlayImageUrl}
              alt="heatmap overlay"
              className="pointer-events-none absolute inset-0 h-full w-full object-contain opacity-80"
            />
          ) : null}
        </>
      )}
    </div>
  );
}

function InfoMini({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border-2 border-slate-800 bg-slate-950 p-2.5">
      <div className="text-[10px] text-slate-500">{label}</div>
      <div className="mt-1 text-sm font-bold text-slate-100">{value}</div>
    </div>
  );
}

function MessageBanner({
  message,
  tone = "slate",
  dark = false,
}: {
  message: string;
  tone?: Tone;
  dark?: boolean;
}) {
  const toneMap = dark
    ? {
        slate: "border-slate-700 bg-slate-950 text-slate-300",
        blue: "border-sky-900 bg-sky-950/50 text-sky-200",
        green: "border-emerald-900 bg-emerald-950/50 text-emerald-200",
        amber: "border-amber-900 bg-amber-950/50 text-amber-200",
        red: "border-rose-900 bg-rose-950/50 text-rose-200",
      }
    : {
        slate: "border-slate-200 bg-slate-50 text-slate-700",
        blue: "border-blue-200 bg-blue-50 text-blue-700",
        green: "border-emerald-200 bg-emerald-50 text-emerald-700",
        amber: "border-amber-200 bg-amber-50 text-amber-700",
        red: "border-rose-200 bg-rose-50 text-rose-700",
      };

  return <div className={cls("rounded-2xl border-2 px-4 py-3 text-sm", toneMap[tone])}>{message}</div>;
}

function ScrollTable({ children }: { children: React.ReactNode }) {
  return <div className="min-h-0 overflow-auto">{children}</div>;
}

function FieldPage({
  selectedLine,
  dashboard,
  onRefresh,
}: {
  selectedLine: LineId;
  dashboard: DashboardResponse | null;
  onRefresh: () => Promise<void>;
}) {
  const [mode, setMode] = useState<ViewMode>("raw");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [rawImageUrl, setRawImageUrl] = useState("");
  const [overlayImageUrl, setOverlayImageUrl] = useState("");
  const [heatmapImageUrl, setHeatmapImageUrl] = useState("");
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictError, setPredictError] = useState("");
  const [fieldMessage, setFieldMessage] = useState("");

  const fallbackSample = liveSamplesByLine[selectedLine][0];
  const productionModel = pickProductionModel(dashboard);
  const canaryModel = pickCanaryModel(dashboard);
  const activeModelId =
    canaryModel && dashboard?.deployment.canary_line === selectedLine
      ? canaryModel.id
      : productionModel?.id ?? "-";

  useEffect(() => {
    return () => {
      if (rawImageUrl.startsWith("blob:")) {
        URL.revokeObjectURL(rawImageUrl);
      }
    };
  }, [rawImageUrl]);

  const displayedDecision = predictResult ? toDisplayDecision(predictResult.decision) : fallbackSample.label;
  const displayedConfidence = predictResult ? clamp01(predictResult.gate_score) : fallbackSample.confidence;
  const displayedLatency = predictResult ? Math.round(predictResult.latency.total_latency_ms) : fallbackSample.inferMs;
  const displayedHeatmapScore = predictResult?.heatmap_score ?? null;
  const displayedIssueType = predictResult ? toIssueType(predictResult) : fallbackSample.issueType;
  const currentStatus = isPredicting ? "검사 중" : displayedDecision === "이상" ? "이상" : fallbackSample.status;

  async function handlePredict() {
    if (!selectedFile) {
      setPredictError("먼저 이미지를 업로드하세요.");
      return;
    }

    try {
      setIsPredicting(true);
      setPredictError("");
      setFieldMessage("");

      const result = await predictAnomaly(selectedFile);
      setPredictResult(result);
      const overlayUrl = result.heatmap_overlay ? `data:image/png;base64,${result.heatmap_overlay}` : "";
      const heatmapUrl = result.normalized_score_heatmap
        ? `data:image/png;base64,${result.normalized_score_heatmap}`
        : overlayUrl;
      setOverlayImageUrl(overlayUrl);
      setHeatmapImageUrl(heatmapUrl);

      if (overlayUrl) {
        setMode("overlay");
      } else if (heatmapUrl) {
        setMode("heatmap");
      } else {
        setMode("raw");
      }
    } catch (error) {
      setPredictError(error instanceof Error ? error.message : "추론 요청에 실패했습니다.");
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
      setFieldMessage("피드백을 남기려면 먼저 이미지를 업로드하고 검사하세요.");
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
      });
      setFieldMessage("피드백이 작업 데이터셋에 반영되었습니다.");
      await onRefresh();
    } catch (error) {
      setFieldMessage(error instanceof Error ? error.message : "피드백 저장에 실패했습니다.");
    }
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-[1.2fr_0.8fr] gap-4">
      <DarkCard className="flex min-h-0 flex-col p-5">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <div className="text-2xl font-black text-slate-50">현장 검사 HMI</div>
            <div className="mt-1 text-sm text-slate-400">
              {selectedLine} · 모델 {activeModelId}
            </div>
          </div>
          <Badge tone={displayedDecision === "이상" ? "red" : "green"} dark>
            {displayedDecision}
          </Badge>
        </div>

        <div className="mb-4 grid grid-cols-4 gap-3">
          <Stat label="검사 상태" value={currentStatus} icon={CheckCircle2} tone={currentStatus === "이상" ? "red" : "green"} dark compact />
          <Stat label="신뢰도" value={formatPercent(displayedConfidence)} icon={Cpu} tone="blue" dark compact />
          <Stat label="응답 속도" value={`${displayedLatency}ms`} icon={Clock3} tone="slate" dark compact />
          <Stat label="Heatmap" value={displayedHeatmapScore == null ? "-" : formatMetric(displayedHeatmapScore)} icon={ShieldAlert} tone="amber" dark compact />
        </div>

        <div className="mb-4 grid grid-cols-[1fr_auto] gap-3">
          <label className="flex cursor-pointer items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900 px-4 py-3 text-sm font-semibold text-slate-200 hover:bg-slate-800">
            이미지 업로드
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (!file) return;

                if (rawImageUrl.startsWith("blob:")) {
                  URL.revokeObjectURL(rawImageUrl);
                }

                setSelectedFile(file);
                setPredictResult(null);
                setPredictError("");
                setFieldMessage("");
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
            className="rounded-2xl border-2 border-cyan-400 bg-cyan-400 px-5 py-3 text-sm font-black text-slate-950 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isPredicting ? "검사 중..." : "이상탐지 실행"}
          </button>
        </div>

        {predictError ? <MessageBanner message={predictError} tone="red" dark /> : null}
        {fieldMessage ? <div className="mt-3"><MessageBanner message={fieldMessage} tone="blue" dark /></div> : null}

        <div className="mb-4 mt-4 h-[56px]">
          <ModeSwitch mode={mode} setMode={setMode} />
        </div>

        <div className="min-h-0 flex-1">
          <CameraViewport
            mode={mode}
            rawImageUrl={rawImageUrl}
            overlayImageUrl={overlayImageUrl}
            heatmapImageUrl={heatmapImageUrl}
          />
        </div>
      </DarkCard>

      <DarkCard className="grid min-h-0 grid-rows-[auto_auto_1fr] gap-4 p-5">
        <div className="grid grid-cols-2 gap-3">
          <InfoMini label="샘플 ID" value={fallbackSample.sampleId} />
          <InfoMini label="카메라" value={fallbackSample.camera} />
          <InfoMini label="판정 결과" value={displayedDecision} />
          <InfoMini label="현재 모델" value={activeModelId} />
          <InfoMini label="Gate 확률" value={formatPercent(displayedConfidence)} />
          <InfoMini label="이상 유형" value={displayedIssueType} />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <ActionButton
            title="오탐 등록"
            desc="정상이지만 이상으로 판정된 샘플로 누적"
            tone="blue"
            onClick={() => handleFeedback("false_positive", "normal", "오탐 등록")}
          />
          <ActionButton
            title="미탐 등록"
            desc="이상이지만 정상으로 놓친 샘플로 누적"
            tone="red"
            onClick={() => handleFeedback("false_negative", "anomaly", "미탐 등록")}
          />
        </div>

        <div className="grid min-h-0 grid-rows-[auto_1fr] gap-4">
          <div className="grid grid-cols-2 gap-3">
            <ActionButton
              title="입력 품질 저하"
              desc="조명, 초점, 반사 등으로 인한 품질 이슈 기록"
              tone="amber"
              onClick={() => handleFeedback("needs_review", "unlabeled", "입력 품질 저하")}
            />
            <ActionButton
              title="판정 보류"
              desc="사람 확인이 필요한 샘플로 보류"
              tone="slate"
              onClick={() => handleFeedback("needs_review", "unlabeled", "판정 보류")}
            />
          </div>

          <div className="min-h-0 overflow-auto rounded-2xl border-2 border-slate-800 bg-slate-950 p-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="text-sm font-bold text-slate-100">최근 피드백</div>
              <div className="text-xs text-slate-500">{dashboard?.active_dataset_id ?? "-"}</div>
            </div>
            <div className="space-y-3">
              {dashboard?.feedback_items.length ? (
                dashboard.feedback_items.slice(0, 4).map((item) => (
                  <div key={item.id} className="rounded-2xl border border-slate-800 bg-slate-900 p-3">
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-sm font-semibold text-slate-100">{item.feedback_type}</div>
                      <div className="text-xs text-slate-500">{formatDate(item.created_at)}</div>
                    </div>
                    <div className="mt-2 text-sm text-slate-400">{item.comment || "의견 없음"}</div>
                  </div>
                ))
              ) : (
                <div className="text-sm text-slate-500">아직 저장된 피드백이 없습니다.</div>
              )}
            </div>
          </div>
        </div>
      </DarkCard>
    </div>
  );
}

function OpsView({
  dashboard,
  onJumpToField,
  onOpenVersion,
  onOpenLogs,
}: {
  dashboard: DashboardResponse;
  onJumpToField: (line: LineId) => void;
  onOpenVersion: () => void;
  onOpenLogs: () => void;
}) {
  const productionModel = pickProductionModel(dashboard);
  const dynamicLineStatus = LINES.map((lineId) => {
    const base = baseLineStatus[lineId];
    if (dashboard.deployment.canary_line === lineId) {
      return {
        lineId,
        ...base,
        state: "주의",
        alert: `Canary ${dashboard.deployment.canary_model_id ?? "-"}`,
      };
    }
    return { lineId, ...base };
  });

  return (
    <div className="grid h-full min-h-0 grid-rows-[auto_1fr] gap-4">
      <div className="grid grid-cols-4 gap-4">
        <Stat label="운영 모델" value={productionModel?.id ?? "-"} sub="현재 Production" icon={Cpu} tone="blue" />
        <Stat label="최근 이상률" value={`${Math.min(99, dashboard.feedback_items.length + 3)}%`} sub="피드백 기준 체감치" icon={AlertTriangle} tone="amber" />
        <Stat label="누적 피드백" value={`${dashboard.feedback_items.length}건`} sub={dashboard.active_dataset_id} icon={ShieldAlert} tone="slate" />
        <Stat label="시스템 상태" value="정상" sub={`Canary ${dashboard.deployment.canary_model_id ? "동작 중" : "없음"}`} icon={CheckCircle2} tone="green" />
      </div>

      <div className="grid min-h-0 grid-cols-[1.15fr_.85fr] gap-4">
        <Card className="flex min-h-0 flex-col p-5">
          <div className="mb-4 flex items-center justify-between">
            <div className="text-xl font-bold text-slate-950">라인별 운영 상태</div>
            <Badge tone={dashboard.deployment.canary_model_id ? "amber" : "green"}>
              {dashboard.deployment.canary_model_id ? "Canary 검증중" : "안정 운영"}
            </Badge>
          </div>
          <div className="min-h-0 overflow-hidden rounded-2xl border-2 border-slate-200">
            <ScrollTable>
              <table className="min-w-full text-sm">
                <thead className="sticky top-0 bg-slate-50 text-slate-600">
                  <tr>
                    {["라인", "상태", "수율", "평균 지연", "카메라", "알람"].map((heading) => (
                      <th key={heading} className="px-4 py-3 text-left font-semibold">
                        {heading}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dynamicLineStatus.map((line) => (
                    <tr key={line.lineId} className="border-t border-slate-200 bg-white">
                      <td className="px-4 py-3 font-semibold text-slate-900">{line.lineId}</td>
                      <td className="px-4 py-3">
                        <Badge tone={line.state === "주의" ? "amber" : "green"}>{line.state}</Badge>
                      </td>
                      <td className="px-4 py-3">{line.yieldRate}%</td>
                      <td className="px-4 py-3">{line.latency}ms</td>
                      <td className="px-4 py-3">{line.camera}</td>
                      <td className="px-4 py-3">{line.alert}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ScrollTable>
          </div>
        </Card>

        <div className="grid min-h-0 grid-rows-[auto_1fr] gap-4">
          <Card className="p-5">
            <div className="mb-4 text-xl font-bold text-slate-950">즉시 대응</div>
            <div className="grid gap-3">
              <button
                onClick={() => onJumpToField("LINE-B")}
                className="rounded-2xl border-2 border-blue-600 bg-blue-600 px-4 py-3 text-left text-sm font-semibold text-white"
              >
                LINE-B 상세 보기
              </button>
              <button
                onClick={onOpenVersion}
                className="rounded-2xl border-2 border-slate-200 px-4 py-3 text-left text-sm font-semibold text-slate-800"
              >
                문제 샘플 모아보기
              </button>
              <button
                onClick={onOpenLogs}
                className="rounded-2xl border-2 border-slate-200 px-4 py-3 text-left text-sm font-semibold text-slate-800"
              >
                알람 로그 열기
              </button>
            </div>
          </Card>

          <Card className="flex min-h-0 flex-col p-5">
            <div className="mb-4 text-xl font-bold text-slate-950">최근 피드백</div>
            <div className="min-h-0 overflow-auto pr-1">
              <div className="space-y-3">
                {dashboard.feedback_items.length ? (
                  dashboard.feedback_items.map((item) => (
                    <div key={item.id} className="grid grid-cols-[72px_1fr] gap-3 rounded-2xl border-2 border-slate-200 p-3">
                      <img
                        src={asset(item.image_url)}
                        alt={item.id}
                        className="h-16 w-16 rounded-2xl bg-slate-100 object-cover"
                      />
                      <div className="min-w-0">
                        <div className="font-semibold text-slate-900">{item.feedback_type}</div>
                        <div className="mt-1 truncate text-sm text-slate-500">{item.comment || "의견 없음"}</div>
                        <div className="mt-2 text-xs text-slate-400">{formatDate(item.created_at)}</div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="rounded-2xl border-2 border-slate-200 p-4 text-sm text-slate-500">
                    최근 피드백이 아직 없습니다.
                  </div>
                )}
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function TrainingRunsTable({
  runs,
  selectedRunId,
  onSelectRun,
}: {
  runs: TrainingRun[];
  selectedRunId?: string;
  onSelectRun?: (runId: string) => void;
}) {
  return (
    <div className="min-h-0 overflow-hidden rounded-2xl border-2 border-slate-200">
      <ScrollTable>
        <table className="min-w-full text-sm">
          <thead className="sticky top-0 bg-slate-50 text-slate-600">
            <tr>
              {["Run", "Dataset", "Base", "Recipe", "진행률", "상태"].map((heading) => (
                <th key={heading} className="px-4 py-3 text-left font-semibold">
                  {heading}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {runs.length ? (
              runs.map((run) => {
                const progress = Math.min(100, Math.max(0, run.progress ?? 0));
                const selected = selectedRunId === run.id;

                return (
                  <tr
                    key={run.id}
                    onClick={() => onSelectRun?.(run.id)}
                    className={cls(
                      "border-t border-slate-200",
                      onSelectRun ? "cursor-pointer transition hover:bg-slate-50" : "",
                      selected ? "bg-blue-50" : ""
                    )}
                  >
                    <td className="px-4 py-3 font-semibold">{run.id}</td>
                    <td className="px-4 py-3">{run.dataset_version_id}</td>
                    <td className="px-4 py-3">{run.base_model_version_id ?? "-"}</td>
                    <td className="px-4 py-3">{run.recipe?.name ?? run.recipe_id ?? run.train_strategy}</td>
                    <td className="px-4 py-3">
                      <div className="h-2 w-28 overflow-hidden rounded-full bg-slate-200">
                        <div className="h-full rounded-full bg-blue-600" style={{ width: `${progress}%` }} />
                      </div>
                      <div className="mt-1 text-xs text-slate-500">{progress}% · {run.current_step ?? "configured"}</div>
                    </td>
                    <td className="px-4 py-3">
                      <Badge tone={run.status === "configured" ? "blue" : "green"}>{run.status}</Badge>
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={6} className="px-4 py-6 text-center text-slate-500">
                  아직 생성된 학습 런이 없습니다.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </ScrollTable>
    </div>
  );
}

function DeploymentCard({
  title,
  tone,
  model,
  subtitle,
}: {
  title: string;
  tone: Tone;
  model: ModelVersion | null;
  subtitle: string;
}) {
  const toneMap: Record<Tone, string> = {
    slate: "border-slate-200 bg-slate-50 text-slate-900",
    blue: "border-blue-200 bg-blue-50 text-blue-950",
    green: "border-emerald-200 bg-emerald-50 text-emerald-950",
    amber: "border-amber-200 bg-amber-50 text-amber-950",
    red: "border-rose-200 bg-rose-50 text-rose-950",
  };

  return (
    <div className={cls("rounded-2xl border-2 p-4", toneMap[tone])}>
      <div className="text-sm font-semibold">{title}</div>
      <div className="mt-2 text-2xl font-bold">{model?.name ?? model?.id ?? "-"}</div>
      {model?.name ? <div className="mt-1 text-xs opacity-70">{model.id}</div> : null}
      <div className="mt-2 text-sm opacity-80">{subtitle}</div>
      <div className="mt-2 text-sm opacity-80">
        F1 {model?.metrics.f1 ?? "-"} / {model?.metrics.latency_ms ?? "-"}ms
      </div>
    </div>
  );
}

function TrainDeployView({
  dashboard,
  onRefresh,
}: {
  dashboard: DashboardResponse;
  onRefresh: () => Promise<void>;
}) {
  const productionModel = pickProductionModel(dashboard);
  const stagingModel = pickStagingModel(dashboard);
  const canaryModel = pickCanaryModel(dashboard);
  const candidateModel = canaryModel ?? stagingModel ?? null;
  const recipes = dashboard.training_recipes ?? [];
  const firstRecipeId = recipes[0]?.id ?? "balanced-finetune-v1";
  const defaultBaseModelId = productionModel?.id ?? dashboard.model_versions[0]?.id ?? "";
  const defaultDeployModelId = (candidateModel ?? productionModel ?? dashboard.model_versions[0] ?? null)?.id ?? "";
  const gateArchitectureId =
    dashboard.architectures.find((arch) => arch.kind === "gate")?.id ?? "ARCH-GATE-EFF";
  const heatmapArchitectureId =
    dashboard.architectures.find((arch) => arch.kind === "heatmap")?.id ?? "ARCH-HM-PC";

  const [busyAction, setBusyAction] = useState<BusyAction>(null);
  const [message, setMessage] = useState("");
  const [selectedDatasetId, setSelectedDatasetId] = useState(dashboard.active_dataset_id);
  const [selectedBaseModelId, setSelectedBaseModelId] = useState(defaultBaseModelId);
  const [selectedRecipeId, setSelectedRecipeId] = useState(firstRecipeId);
  const [selectedRunId, setSelectedRunId] = useState(dashboard.training_runs[0]?.id ?? "");
  const [selectedDeployModelId, setSelectedDeployModelId] = useState(defaultDeployModelId);
  const [selectedTargetLine, setSelectedTargetLine] = useState<LineId>("LINE-B");
  const [modelName, setModelName] = useState(() => formatKoreaTimestamp());

  useEffect(() => {
    if (!dashboard.dataset_versions.some((dataset) => dataset.id === selectedDatasetId)) {
      setSelectedDatasetId(dashboard.active_dataset_id);
    }
  }, [dashboard.active_dataset_id, dashboard.dataset_versions, selectedDatasetId]);

  useEffect(() => {
    if (defaultBaseModelId && !dashboard.model_versions.some((model) => model.id === selectedBaseModelId)) {
      setSelectedBaseModelId(defaultBaseModelId);
    }
  }, [dashboard.model_versions, defaultBaseModelId, selectedBaseModelId]);

  useEffect(() => {
    if (firstRecipeId && !recipes.some((recipe) => recipe.id === selectedRecipeId)) {
      setSelectedRecipeId(firstRecipeId);
    }
  }, [recipes, firstRecipeId, selectedRecipeId]);

  useEffect(() => {
    if (!dashboard.training_runs.length) {
      setSelectedRunId("");
      return;
    }
    if (!dashboard.training_runs.some((run) => run.id === selectedRunId)) {
      setSelectedRunId(dashboard.training_runs[0].id);
    }
  }, [dashboard.training_runs, selectedRunId]);

  useEffect(() => {
    if (defaultDeployModelId && !dashboard.model_versions.some((model) => model.id === selectedDeployModelId)) {
      setSelectedDeployModelId(defaultDeployModelId);
    }
  }, [dashboard.model_versions, defaultDeployModelId, selectedDeployModelId]);

  const selectedDataset = dashboard.dataset_versions.find((dataset) => dataset.id === selectedDatasetId);
  const selectedBaseModel = dashboard.model_versions.find((model) => model.id === selectedBaseModelId) ?? null;
  const selectedRecipe = recipes.find((recipe) => recipe.id === selectedRecipeId) ?? recipes[0];
  const selectedRun =
    dashboard.training_runs.find((run) => run.id === selectedRunId) ?? dashboard.training_runs[0] ?? null;
  const selectedDeployModel =
    dashboard.model_versions.find((model) => model.id === selectedDeployModelId) ?? candidateModel ?? null;
  const selectedRunProgress = Math.min(100, Math.max(0, selectedRun?.progress ?? 0));
  const selectedRunTone: Tone =
    selectedRun?.status === "completed" ? "green" : selectedRun?.status === "failed" ? "red" : "blue";

  async function handleCreateRun() {
    if (!selectedDatasetId || !selectedBaseModelId || !selectedRecipeId) {
      setMessage("학습 데이터셋, 기준 모델, 레시피를 모두 선택하세요.");
      return;
    }

    try {
      setBusyAction("train");
      setMessage("");
      const result = await createTrainingRun({
        modelName: modelName.trim() || formatKoreaTimestamp(),
        knownModelIds: dashboard.model_versions.map((model) => model.id),
        datasetVersionId: selectedDatasetId,
        baseModelVersionId: selectedBaseModelId,
        recipeId: selectedRecipeId,
        targetLine: selectedTargetLine,
        gateArchitectureId,
        heatmapArchitectureId,
        trainStrategy: "cascade",
        notes: `UI configured run from ${selectedBaseModelId} using ${selectedRecipeId}`,
      });
      const createdRun = (result as { training_run?: TrainingRun }).training_run;
      const createdModel = (result as { model_version?: ModelVersion }).model_version;
      if (createdRun?.id) setSelectedRunId(createdRun.id);
      if (createdModel?.id) setSelectedDeployModelId(createdModel.id);
      setModelName(formatKoreaTimestamp());
      setMessage("학습 구성이 저장되고 staging 후보 모델이 등록되었습니다. 실제 학습은 아직 시작되지 않았습니다.");
      await onRefresh();
    } catch (error) {
      if (error instanceof Error && /training in progress/i.test(error.message)) {
        await onRefresh();
      }
      setMessage(error instanceof Error ? error.message : "학습 구성 저장에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  function handleFinalTrainClick() {
    window.alert("아직은 작동하지 않습니다");
    setMessage("최종 학습 실행은 아직 연결되지 않았습니다. 현재는 구성 저장과 상태 표시만 지원합니다.");
  }

  async function handleStartCanary() {
    if (!selectedDeployModel) {
      setMessage("Canary를 시작할 모델을 선택하세요.");
      return;
    }
    if (selectedDeployModel.status === "production") {
      setMessage("현재 production 모델은 Canary 후보로 다시 사용할 수 없습니다. staging 또는 canary 후보를 선택하세요.");
      return;
    }

    try {
      setBusyAction("canary");
      setMessage("");
      await startCanary(selectedDeployModel.id, selectedTargetLine);
      setMessage(`Canary가 ${selectedTargetLine}에서 시작되었습니다.`);
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Canary 시작에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleApprove() {
    if (!selectedDeployModel) {
      setMessage("배포 승인할 모델을 선택하세요.");
      return;
    }
    if (selectedDeployModel.status === "production") {
      setMessage("이미 production 상태인 모델입니다.");
      return;
    }

    try {
      setBusyAction("approve");
      setMessage("");
      await promoteModel(selectedDeployModel.id, "production");
      setMessage("배포 승인이 완료되어 production 모델이 갱신되었습니다.");
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
      setMessage("이전 production 후보로 롤백되었습니다.");
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
        <Stat label="학습 런" value={`${dashboard.training_runs.length}개`} sub="최근 생성 기준" icon={Flame} tone="blue" />
        <Stat label="배포 후보" value={`${dashboard.model_versions.filter((model) => model.status !== "production").length}개`} sub="staging / canary" icon={Rocket} tone="green" />
        <Stat label="선택 데이터" value={`${selectedDataset?.sample_count ?? 0}개`} sub={selectedDataset?.id ?? "데이터셋 미선택"} icon={Database} tone="amber" />
        <Stat label="현재 production" value={productionModel?.id ?? "-"} sub="실 배포 모델" icon={Cpu} tone="slate" />
      </div>

      <div className="grid min-h-0 grid-cols-[1.08fr_.92fr] gap-4">
        <Card className="flex min-h-0 flex-col p-5">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <div className="text-xl font-bold text-slate-950">학습 구성</div>
              <div className="mt-1 text-sm text-slate-500">데이터셋, 기준 모델, 레시피를 고른 뒤 학습 계획을 저장합니다.</div>
            </div>
            <Badge tone={selectedRunTone}>{selectedRun?.current_step ?? "ready"}</Badge>
          </div>

          {message ? <div className="mb-4"><MessageBanner message={message} tone="blue" /></div> : null}

          <div className="min-h-0 overflow-auto pr-1">
            <div className="grid grid-cols-4 gap-3">
              <label className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="text-xs font-bold uppercase tracking-[0.18em] text-slate-400">Dataset</div>
                <select
                  value={selectedDatasetId}
                  onChange={(event) => setSelectedDatasetId(event.target.value)}
                  className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900 outline-none focus:border-blue-500"
                >
                  {dashboard.dataset_versions.map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>
                      {dataset.name} · {dataset.sample_count}
                    </option>
                  ))}
                </select>
              </label>

              <label className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="text-xs font-bold uppercase tracking-[0.18em] text-slate-400">Base Model</div>
                <select
                  value={selectedBaseModelId}
                  onChange={(event) => setSelectedBaseModelId(event.target.value)}
                  className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900 outline-none focus:border-blue-500"
                >
                  {dashboard.model_versions.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.id} · {model.status}
                    </option>
                  ))}
                </select>
              </label>

              <label className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="text-xs font-bold uppercase tracking-[0.18em] text-slate-400">Target Line</div>
                <select
                  value={selectedTargetLine}
                  onChange={(event) => setSelectedTargetLine(event.target.value as LineId)}
                  className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900 outline-none focus:border-blue-500"
                >
                  {LINES.map((line) => (
                    <option key={line} value={line}>
                      {line}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="mt-4 grid grid-cols-3 gap-3">
              {recipes.map((recipe) => {
                const selected = recipe.id === selectedRecipeId;
                return (
                  <button
                    key={recipe.id}
                    type="button"
                    onClick={() => setSelectedRecipeId(recipe.id)}
                    className={cls(
                      "rounded-2xl border-2 p-4 text-left transition",
                      selected ? "border-blue-600 bg-blue-50" : "border-slate-200 bg-white hover:border-blue-300"
                    )}
                  >
                    <div className="text-sm font-bold text-slate-950">{recipe.name}</div>
                    <div className="mt-1 min-h-10 text-xs text-slate-500">{recipe.description}</div>
                    <div className="mt-3 grid grid-cols-2 gap-2 text-xs font-semibold text-slate-700">
                      <span>batch {recipe.batch_size}</span>
                      <span>lr {recipe.learning_rate}</span>
                      <span>{recipe.optimizer}</span>
                      <span>{recipe.epochs} epochs</span>
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="mt-4 rounded-2xl border-2 border-slate-200 bg-white p-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-sm font-bold text-slate-950">학습 상태</div>
                  <div className="mt-1 text-sm text-slate-500">
                    {selectedRun
                      ? `${selectedRun.id} · ${selectedRun.current_step ?? selectedRun.status}`
                      : "아직 생성된 학습 런이 없습니다."}
                  </div>
                </div>
                <Badge tone={selectedRunTone}>{selectedRun?.status ?? "not-created"}</Badge>
              </div>
              <div className="mt-4 h-3 overflow-hidden rounded-full bg-slate-200">
                <div className="h-full rounded-full bg-blue-600 transition-all" style={{ width: `${selectedRunProgress}%` }} />
              </div>
              <div className="mt-2 flex justify-between text-xs font-semibold text-slate-500">
                <span>{selectedRunProgress}%</span>
                <span>base {selectedRun?.base_model_version_id ?? selectedBaseModel?.id ?? "-"}</span>
                <span>{selectedRun?.recipe?.name ?? selectedRecipe?.name ?? "-"}</span>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <button
                onClick={handleCreateRun}
                disabled={busyAction !== null}
                className="rounded-2xl border-2 border-blue-600 bg-blue-600 px-4 py-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
              >
                {busyAction === "train" ? "구성 저장 중..." : "학습 구성 저장"}
              </button>
              <button
                onClick={handleFinalTrainClick}
                disabled={busyAction !== null}
                className="rounded-2xl border-2 border-slate-900 bg-slate-900 px-4 py-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
              >
                최종 학습 시작
              </button>
            </div>

            <div className="mt-4">
              <TrainingRunsTable
                runs={dashboard.training_runs}
                selectedRunId={selectedRun?.id}
                onSelectRun={setSelectedRunId}
              />
            </div>
          </div>
        </Card>

        <Card className="flex min-h-0 flex-col p-5">
          <div className="mb-4">
            <div className="text-xl font-bold text-slate-950">배포 / Canary 선택</div>
            <div className="mt-1 text-sm text-slate-500">staging, canary, 과거 모델 중에서 직접 후보를 골라 조작합니다.</div>
          </div>

          <div className="min-h-0 overflow-auto pr-1">
            <div className="grid gap-3 rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
              <label>
                <div className="text-xs font-bold uppercase tracking-[0.18em] text-slate-400">Model Candidate</div>
                <select
                  value={selectedDeployModelId}
                  onChange={(event) => setSelectedDeployModelId(event.target.value)}
                  className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900 outline-none focus:border-blue-500"
                >
                  {dashboard.model_versions.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name ?? model.id} · {model.status} · {model.recipe_id ?? "baseline"}
                    </option>
                  ))}
                </select>
              </label>
              <div className="grid grid-cols-2 gap-3 text-sm text-slate-600">
                <div>모델 이름: {selectedDeployModel?.name ?? "-"}</div>
                <div>기준 모델: {selectedDeployModel?.base_model_version_id ?? "-"}</div>
                <div>데이터셋: {selectedDeployModel?.dataset_version_id ?? "-"}</div>
                <div>레시피: {selectedDeployModel?.recipe_id ?? "-"}</div>
                <div>목표 라인: {selectedTargetLine}</div>
              </div>
            </div>

            <div className="mt-4 grid gap-4">
              <DeploymentCard title="Production" tone="slate" model={productionModel} subtitle="실 운영 중" />
              <DeploymentCard
                title="Staging"
                tone="blue"
                model={stagingModel}
                subtitle="검증 대기 또는 바로 Canary 진입 가능"
              />
              <DeploymentCard
                title="Canary"
                tone="amber"
                model={canaryModel}
                subtitle={dashboard.deployment.canary_line ? `${dashboard.deployment.canary_line} 검증중` : "현재 없음"}
              />

              <div className="grid grid-cols-3 gap-3">
                <button
                  onClick={handleStartCanary}
                  disabled={busyAction !== null}
                  className="rounded-2xl border-2 border-slate-200 px-4 py-3 text-sm font-semibold text-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {busyAction === "canary" ? "시작 중..." : "Canary 시작"}
                </button>
                <button
                  onClick={handleApprove}
                  disabled={busyAction !== null}
                  className="rounded-2xl border-2 border-blue-600 bg-blue-600 px-4 py-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {busyAction === "approve" ? "승인 중..." : "배포 승인"}
                </button>
                <button
                  onClick={handleRollback}
                  disabled={busyAction !== null}
                  className="rounded-2xl border-2 border-rose-600 bg-rose-600 px-4 py-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {busyAction === "rollback" ? "롤백 중..." : "롤백"}
                </button>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

function TrainDeployViewV2({
  dashboard,
  onRefresh,
}: {
  dashboard: DashboardResponse;
  onRefresh: () => Promise<void>;
}) {
  const productionModel = pickProductionModel(dashboard);
  const stagingModel = pickStagingModel(dashboard);
  const canaryModel = pickCanaryModel(dashboard);
  const candidateModel = canaryModel ?? stagingModel ?? null;
  const recipes = dashboard.training_recipes ?? [];
  const firstRecipe = recipes[0];
  const defaultBaseModelId = productionModel?.id ?? dashboard.model_versions[0]?.id ?? "";
  const defaultDeployModelId = (candidateModel ?? productionModel ?? dashboard.model_versions[0] ?? null)?.id ?? "";
  const gateArchitectureId =
    dashboard.architectures.find((arch) => arch.kind === "gate")?.id ?? "ARCH-GATE-EFF";
  const heatmapArchitectureId =
    dashboard.architectures.find((arch) => arch.kind === "heatmap")?.id ?? "ARCH-HM-PC";

  const [busyAction, setBusyAction] = useState<BusyAction>(null);
  const [message, setMessage] = useState("");
  const [selectedDatasetId, setSelectedDatasetId] = useState(dashboard.active_dataset_id);
  const [selectedBaseModelId, setSelectedBaseModelId] = useState(defaultBaseModelId);
  const [selectedRecipeId, setSelectedRecipeId] = useState(firstRecipe?.id ?? "cpu-balanced-effnetb0");
  const [selectedDeployModelId, setSelectedDeployModelId] = useState(defaultDeployModelId);
  const [selectedCanaryLine, setSelectedCanaryLine] = useState<LineId>("LINE-B");
  const [epochCount, setEpochCount] = useState(firstRecipe?.default_epochs ?? 3);
  const [modelName, setModelName] = useState(() => formatKoreaTimestamp());
  const [recipeDraft, setRecipeDraft] = useState<TrainingRecipe>(() => recipeDraftFrom(firstRecipe));

  const activeRun =
    dashboard.training_runs.find((run) => ["preparing", "running", "stopping"].includes(run.status)) ?? null;
  const latestRun = activeRun ?? dashboard.training_runs[0] ?? null;
  const selectedDataset = dashboard.dataset_versions.find((dataset) => dataset.id === selectedDatasetId);
  const selectedBaseModel = dashboard.model_versions.find((model) => model.id === selectedBaseModelId) ?? null;
  const selectedRecipe = recipes.find((recipe) => recipe.id === selectedRecipeId) ?? firstRecipe;
  const selectedDeployModel =
    dashboard.model_versions.find((model) => model.id === selectedDeployModelId) ?? candidateModel ?? null;
  const runProgress = Math.min(100, Math.max(0, latestRun?.progress ?? 0));
  const runTone: Tone =
    latestRun?.status === "completed"
      ? "green"
      : latestRun?.status === "failed" || latestRun?.status === "stopped"
        ? "red"
        : "blue";

  useEffect(() => {
    if (!dashboard.dataset_versions.some((dataset) => dataset.id === selectedDatasetId)) {
      setSelectedDatasetId(dashboard.active_dataset_id);
    }
  }, [dashboard.active_dataset_id, dashboard.dataset_versions, selectedDatasetId]);

  useEffect(() => {
    if (defaultBaseModelId && !dashboard.model_versions.some((model) => model.id === selectedBaseModelId)) {
      setSelectedBaseModelId(defaultBaseModelId);
    }
  }, [dashboard.model_versions, defaultBaseModelId, selectedBaseModelId]);

  useEffect(() => {
    if (firstRecipe && !recipes.some((recipe) => recipe.id === selectedRecipeId)) {
      setSelectedRecipeId(firstRecipe.id);
      setRecipeDraft(recipeDraftFrom(firstRecipe));
      setEpochCount(firstRecipe.default_epochs ?? 3);
    }
  }, [recipes, firstRecipe, selectedRecipeId]);

  useEffect(() => {
    if (defaultDeployModelId && !dashboard.model_versions.some((model) => model.id === selectedDeployModelId)) {
      setSelectedDeployModelId(defaultDeployModelId);
    }
  }, [dashboard.model_versions, defaultDeployModelId, selectedDeployModelId]);

  function selectRecipe(recipe: TrainingRecipe) {
    setSelectedRecipeId(recipe.id);
    setRecipeDraft(recipeDraftFrom(recipe));
    setEpochCount(recipe.default_epochs ?? 3);
    setMessage(`${recipe.name} 레시피를 선택했습니다.`);
  }

  async function handleStartTraining() {
    if (!selectedDatasetId || !selectedBaseModelId || !selectedRecipeId) {
      setMessage("학습 데이터셋, 기준 모델, 레시피를 모두 선택하세요.");
      return;
    }
    if (activeRun) {
      setMessage(`이미 학습이 진행 중입니다: ${activeRun.id}`);
      return;
    }

    try {
      setBusyAction("train");
      setMessage("");
      const result = await createTrainingRun({
        modelName: modelName.trim() || formatKoreaTimestamp(),
        knownModelIds: dashboard.model_versions.map((model) => model.id),
        datasetVersionId: selectedDatasetId,
        baseModelVersionId: selectedBaseModelId,
        recipeId: selectedRecipeId,
        epochs: epochCount,
        gateArchitectureId,
        heatmapArchitectureId,
        trainStrategy: "cascade",
        notes: `CPU training from ${selectedBaseModelId} using ${selectedRecipeId}`,
      });
      const createdModel = (result as { model_version?: ModelVersion }).model_version;
      if (createdModel?.id) setSelectedDeployModelId(createdModel.id);
      setModelName(formatKoreaTimestamp());
      setMessage("CPU 학습을 시작했습니다. 상태 바는 자동 갱신됩니다.");
      await onRefresh();
    } catch (error) {
      if (error instanceof Error && /training in progress/i.test(error.message)) {
        await onRefresh();
      }
      setMessage(error instanceof Error ? error.message : "학습 시작에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleStopTraining() {
    if (!activeRun) {
      setMessage("중지할 학습 런이 없습니다.");
      return;
    }

    try {
      setBusyAction("stop");
      setMessage("");
      await stopTrainingRun(activeRun.id);
      setMessage(`${activeRun.id} 중지 요청을 보냈습니다.`);
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "학습 중지에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleSaveRecipe() {
    try {
      setBusyAction("train");
      const response = await saveTrainingRecipe({
        ...recipeDraft,
        default_epochs: epochCount,
      });
      setSelectedRecipeId(response.recipe.id);
      setRecipeDraft(recipeDraftFrom(response.recipe));
      setMessage(`새 레시피 JSON을 저장했습니다: ${response.recipe.id}`);
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "레시피 저장에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleStartCanary() {
    if (!selectedDeployModel) {
      setMessage("Canary를 시작할 모델을 선택하세요.");
      return;
    }
    if (selectedDeployModel.status === "production") {
      setMessage("현재 production 모델은 Canary 후보로 다시 사용할 수 없습니다.");
      return;
    }

    try {
      setBusyAction("canary");
      setMessage("");
      await startCanary(selectedDeployModel.id, selectedCanaryLine);
      setMessage(`Canary가 ${selectedCanaryLine}에서 시작되었습니다.`);
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Canary 시작에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleApprove() {
    if (!selectedDeployModel) {
      setMessage("배포 승인할 모델을 선택하세요.");
      return;
    }
    if (selectedDeployModel.status === "production") {
      setMessage("이미 production 상태인 모델입니다.");
      return;
    }

    try {
      setBusyAction("approve");
      setMessage("");
      await promoteModel(selectedDeployModel.id, "production");
      setMessage("배포 승인이 완료되어 production 모델이 갱신되었습니다.");
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
      setMessage("이전 production 후보로 롤백되었습니다.");
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
        <Stat label="학습 상태" value={activeRun ? "진행 중" : "대기"} sub={latestRun?.id ?? "런 없음"} icon={Flame} tone="blue" />
        <Stat label="선택 데이터" value={`${selectedDataset?.sample_count ?? 0}개`} sub={selectedDataset?.id ?? "데이터셋 미선택"} icon={Database} tone="amber" />
        <Stat label="기준 모델" value={selectedBaseModel?.id ?? "-"} sub={selectedBaseModel?.status ?? "모델 미선택"} icon={Cpu} tone="slate" />
        <Stat label="현재 production" value={productionModel?.id ?? "-"} sub="실 배포 모델" icon={Rocket} tone="green" />
      </div>

      <div className="grid min-h-0 grid-cols-[1.08fr_.92fr] gap-4">
        <Card className="flex min-h-0 flex-col p-5">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <div className="text-xl font-bold text-slate-950">CPU 학습</div>
              <div className="mt-1 text-sm text-slate-500">데이터셋과 기준 모델을 고른 뒤 JSON 레시피를 바로 사용하거나 수정 저장합니다.</div>
            </div>
            <Badge tone={runTone}>{latestRun?.current_step ?? "ready"}</Badge>
          </div>

          {message ? <div className="mb-4"><MessageBanner message={message} tone="blue" /></div> : null}

          <div className="min-h-0 overflow-auto pr-1">
            <div className="grid grid-cols-3 gap-3">
              <label className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="text-xs font-bold uppercase tracking-[0.18em] text-slate-400">Dataset</div>
                <select
                  value={selectedDatasetId}
                  onChange={(event) => setSelectedDatasetId(event.target.value)}
                  className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900 outline-none focus:border-blue-500"
                >
                  {dashboard.dataset_versions.map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>
                      {dataset.name} · {dataset.sample_count}
                    </option>
                  ))}
                </select>
              </label>

              <label className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="text-xs font-bold uppercase tracking-[0.18em] text-slate-400">Base Model</div>
                <select
                  value={selectedBaseModelId}
                  onChange={(event) => setSelectedBaseModelId(event.target.value)}
                  className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900 outline-none focus:border-blue-500"
                >
                  {dashboard.model_versions.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.id} · {model.status}
                    </option>
                  ))}
                </select>
              </label>

              <label className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="text-xs font-bold uppercase tracking-[0.18em] text-slate-400">Model Name</div>
                <div className="mt-2 flex gap-2">
                  <input
                    value={modelName}
                    onChange={(event) => setModelName(event.target.value)}
                    className="min-w-0 flex-1 rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900 outline-none focus:border-blue-500"
                  />
                  <button
                    type="button"
                    onClick={() => setModelName(formatKoreaTimestamp())}
                    className="rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-xs font-bold text-slate-700"
                  >
                    KST
                  </button>
                </div>
              </label>

              <label className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="text-xs font-bold uppercase tracking-[0.18em] text-slate-400">Epoch</div>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={epochCount}
                  onChange={(event) => setEpochCount(Math.max(1, Number(event.target.value)))}
                  className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900 outline-none focus:border-blue-500"
                />
              </label>
            </div>

            <div className="mt-4 grid grid-cols-[.88fr_1.12fr] gap-4">
              <div className="max-h-72 overflow-auto rounded-2xl border-2 border-slate-200 bg-white p-3">
                <div className="mb-2 text-sm font-bold text-slate-950">레시피 JSON 목록</div>
                <div className="space-y-2">
                  {recipes.map((recipe) => (
                    <div key={recipe.id} className={cls("rounded-2xl border-2 p-3", recipe.id === selectedRecipeId ? "border-blue-600 bg-blue-50" : "border-slate-200")}>
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <div className="text-sm font-bold text-slate-950">{recipe.name}</div>
                          <div className="mt-1 text-xs text-slate-500">{recipe.description}</div>
                        </div>
                        <Badge tone={recipe.source === "custom" ? "green" : "blue"}>{recipe.source ?? "json"}</Badge>
                      </div>
                      <div className="mt-2 grid grid-cols-2 gap-1 text-xs font-semibold text-slate-600">
                        <span>batch {recipe.batch_size}</span>
                        <span>lr {recipe.learning_rate}</span>
                        <span>{recipe.optimizer}</span>
                        <span>{recipe.scheduler}</span>
                      </div>
                      <div className="mt-3 grid grid-cols-2 gap-2">
                        <button
                          type="button"
                          onClick={() => selectRecipe(recipe)}
                          className="rounded-xl border-2 border-blue-600 bg-blue-600 px-3 py-2 text-xs font-semibold text-white"
                        >
                          바로 사용
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setRecipeDraft(recipeDraftFrom(recipe));
                            setEpochCount(recipe.default_epochs ?? 3);
                            setMessage(`${recipe.name} 레시피를 수정 모드로 불러왔습니다.`);
                          }}
                          className="rounded-xl border-2 border-slate-200 px-3 py-2 text-xs font-semibold text-slate-700"
                        >
                          수정
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="text-sm font-bold text-slate-950">레시피 수정 후 새 JSON 저장</div>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <input className="rounded-xl border-2 border-slate-200 px-3 py-2 text-sm" value={recipeDraft.name} onChange={(e) => setRecipeDraft({ ...recipeDraft, name: e.target.value })} placeholder="name" />
                  <select className="rounded-xl border-2 border-slate-200 px-3 py-2 text-sm" value={recipeDraft.optimizer} onChange={(e) => setRecipeDraft({ ...recipeDraft, optimizer: e.target.value })}>
                    {["AdamW", "Adam", "SGD"].map((optimizer) => <option key={optimizer}>{optimizer}</option>)}
                  </select>
                  <input className="rounded-xl border-2 border-slate-200 px-3 py-2 text-sm" type="number" min={1} value={recipeDraft.batch_size} onChange={(e) => setRecipeDraft({ ...recipeDraft, batch_size: Number(e.target.value) })} placeholder="batch" />
                  <input className="rounded-xl border-2 border-slate-200 px-3 py-2 text-sm" type="number" step="0.0001" value={recipeDraft.learning_rate} onChange={(e) => setRecipeDraft({ ...recipeDraft, learning_rate: Number(e.target.value) })} placeholder="lr" />
                  <input className="rounded-xl border-2 border-slate-200 px-3 py-2 text-sm" type="number" step="0.001" value={recipeDraft.weight_decay} onChange={(e) => setRecipeDraft({ ...recipeDraft, weight_decay: Number(e.target.value) })} placeholder="weight decay" />
                  <select className="rounded-xl border-2 border-slate-200 px-3 py-2 text-sm" value={recipeDraft.scheduler} onChange={(e) => setRecipeDraft({ ...recipeDraft, scheduler: e.target.value })}>
                    {["cosine", "step", "none"].map((scheduler) => <option key={scheduler}>{scheduler}</option>)}
                  </select>
                  <textarea className="col-span-2 rounded-xl border-2 border-slate-200 px-3 py-2 text-sm" rows={2} value={recipeDraft.description} onChange={(e) => setRecipeDraft({ ...recipeDraft, description: e.target.value })} placeholder="description" />
                </div>
                <button
                  onClick={handleSaveRecipe}
                  disabled={busyAction !== null}
                  className="mt-3 w-full rounded-2xl border-2 border-slate-900 bg-slate-900 px-4 py-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
                >
                  수정본을 새 JSON으로 저장
                </button>
              </div>
            </div>

            <div className="mt-4 rounded-2xl border-2 border-slate-200 bg-white p-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-sm font-bold text-slate-950">학습 상태</div>
                  <div className="mt-1 text-sm text-slate-500">
                    {latestRun ? `${latestRun.name ?? latestRun.id} · ${latestRun.status}` : "아직 학습 런이 없습니다."}
                  </div>
                </div>
                <Badge tone={runTone}>{latestRun?.current_step ?? "ready"}</Badge>
              </div>
              <div className="mt-4 h-3 overflow-hidden rounded-full bg-slate-200">
                <div className="h-full rounded-full bg-blue-600 transition-all" style={{ width: `${runProgress}%` }} />
              </div>
              <div className="mt-2 flex justify-between text-xs font-semibold text-slate-500">
                <span>{runProgress}%</span>
                <span>{latestRun?.device ?? "cpu"}</span>
                <span>{latestRun?.name ?? latestRun?.recipe?.name ?? selectedRecipe?.name ?? "-"}</span>
              </div>
              {latestRun?.logs?.length ? (
                <div className="mt-3 max-h-24 overflow-auto rounded-xl bg-slate-950 p-3 font-mono text-xs text-slate-100">
                  {latestRun.logs.slice(-5).map((log, index) => (
                    <div key={`${log.time}-${index}`}>{log.message}</div>
                  ))}
                </div>
              ) : null}
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <button
                onClick={handleStartTraining}
                disabled={busyAction !== null || Boolean(activeRun)}
                className="rounded-2xl border-2 border-blue-600 bg-blue-600 px-4 py-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
              >
                {busyAction === "train" ? "학습 시작 중..." : "최종 학습 시작"}
              </button>
              <button
                onClick={handleStopTraining}
                disabled={busyAction !== null || !activeRun}
                className="rounded-2xl border-2 border-rose-600 bg-rose-600 px-4 py-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
              >
                {busyAction === "stop" ? "중지 요청 중..." : "학습 중지"}
              </button>
            </div>
          </div>
        </Card>

        <Card className="flex min-h-0 flex-col p-5">
          <div className="mb-4">
            <div className="text-xl font-bold text-slate-950">배포 / Canary 선택</div>
            <div className="mt-1 text-sm text-slate-500">학습 완료 후 staging 모델을 선택해서 canary 또는 production으로 올립니다.</div>
          </div>

          <div className="min-h-0 overflow-auto pr-1">
            <div className="grid gap-3 rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
              <label>
                <div className="text-xs font-bold uppercase tracking-[0.18em] text-slate-400">Model Candidate</div>
                <select
                  value={selectedDeployModelId}
                  onChange={(event) => setSelectedDeployModelId(event.target.value)}
                  className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900 outline-none focus:border-blue-500"
                >
                  {dashboard.model_versions.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name ?? model.id} · {model.status} · {model.recipe_id ?? "baseline"}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <div className="text-xs font-bold uppercase tracking-[0.18em] text-slate-400">Canary Line</div>
                <select
                  value={selectedCanaryLine}
                  onChange={(event) => setSelectedCanaryLine(event.target.value as LineId)}
                  className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900 outline-none focus:border-blue-500"
                >
                  {LINES.map((line) => <option key={line}>{line}</option>)}
                </select>
              </label>
              <div className="grid grid-cols-2 gap-3 text-sm text-slate-600">
                <div>기준 모델: {selectedDeployModel?.base_model_version_id ?? "-"}</div>
                <div>데이터셋: {selectedDeployModel?.dataset_version_id ?? "-"}</div>
                <div>레시피: {selectedDeployModel?.recipe_id ?? "-"}</div>
                <div>모델 이름: {selectedDeployModel?.name ?? "-"}</div>
                <div>Canary 라인: {selectedCanaryLine}</div>
              </div>
            </div>

            <div className="mt-4 grid gap-4">
              <DeploymentCard title="Production" tone="slate" model={productionModel} subtitle="실 운영 중" />
              <DeploymentCard title="Staging" tone="blue" model={stagingModel} subtitle="검증 대기 또는 바로 Canary 진입 가능" />
              <DeploymentCard title="Canary" tone="amber" model={canaryModel} subtitle={dashboard.deployment.canary_line ? `${dashboard.deployment.canary_line} 검증중` : "현재 없음"} />

              <div className="grid grid-cols-3 gap-3">
                <button onClick={handleStartCanary} disabled={busyAction !== null} className="rounded-2xl border-2 border-slate-200 px-4 py-3 text-sm font-semibold text-slate-800 disabled:cursor-not-allowed disabled:opacity-60">
                  {busyAction === "canary" ? "시작 중..." : "Canary 시작"}
                </button>
                <button onClick={handleApprove} disabled={busyAction !== null} className="rounded-2xl border-2 border-blue-600 bg-blue-600 px-4 py-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60">
                  {busyAction === "approve" ? "승인 중..." : "배포 승인"}
                </button>
                <button onClick={handleRollback} disabled={busyAction !== null} className="rounded-2xl border-2 border-rose-600 bg-rose-600 px-4 py-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60">
                  {busyAction === "rollback" ? "롤백 중..." : "롤백"}
                </button>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

function VersionViewV2({
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
      setMessage(mode === "new" ? "피드백 묶음으로 새 데이터셋을 만들었습니다." : "피드백 묶음을 기존 데이터셋에 추가했습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "피드백 데이터셋 처리에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  }

  async function handleUploadFiles() {
    if (!uploadFiles.length) {
      setMessage("추가할 데이터 파일을 선택하세요.");
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
      setMessage(datasetMode === "new" ? "파일로 새 데이터셋을 만들었습니다." : "파일을 선택한 데이터셋에 추가했습니다.");
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
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <div className="text-xl font-bold text-slate-950">데이터 버전</div>
            <div className="mt-1 text-sm text-slate-500">저장된 데이터셋을 확인하고, 피드백 묶음 또는 파일 업로드로 확장합니다.</div>
          </div>
          <Badge tone="blue">{dashboard.dataset_versions.length} versions</Badge>
        </div>

        {message ? <div className="mb-4"><MessageBanner message={message} tone="blue" /></div> : null}

        <div className="mb-4 rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
          <div className="grid grid-cols-2 gap-3">
            <label>
              <div className="text-xs font-bold uppercase tracking-[0.18em] text-slate-400">Target Dataset</div>
              <select
                value={targetDatasetId}
                onChange={(event) => setTargetDatasetId(event.target.value)}
                className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900 outline-none focus:border-blue-500"
              >
                {dashboard.dataset_versions.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name} · {dataset.sample_count}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <div className="text-xs font-bold uppercase tracking-[0.18em] text-slate-400">New Dataset Name</div>
              <input
                value={datasetName}
                onChange={(event) => setDatasetName(event.target.value)}
                className="mt-2 w-full rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900 outline-none focus:border-blue-500"
                placeholder="새 데이터셋 이름"
              />
            </label>
          </div>

          <div className="mt-3 grid grid-cols-2 gap-3">
            <button
              onClick={() => handleFeedbackMaterialize("append")}
              disabled={busy || !dashboard.feedback_items.length}
              className="rounded-2xl border-2 border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              피드백 묶음 기존 데이터셋에 추가
            </button>
            <button
              onClick={() => handleFeedbackMaterialize("new")}
              disabled={busy || !dashboard.feedback_items.length}
              className="rounded-2xl border-2 border-blue-600 bg-blue-600 px-4 py-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
            >
              피드백 묶음으로 새 데이터셋 생성
            </button>
          </div>

          <div className="mt-4 grid grid-cols-[1fr_.7fr_.7fr] gap-3">
            <input
              type="file"
              multiple
              onChange={(event) => setUploadFiles(Array.from(event.target.files ?? []))}
              className="rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm text-slate-700"
            />
            <select
              value={uploadLabel}
              onChange={(event) => setUploadLabel(event.target.value)}
              className="rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900"
            >
              <option value="normal">normal</option>
              <option value="anomaly">anomaly</option>
              <option value="unlabeled">unlabeled</option>
            </select>
            <select
              value={datasetMode}
              onChange={(event) => setDatasetMode(event.target.value as "append" | "new")}
              className="rounded-xl border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900"
            >
              <option value="append">기존에 추가</option>
              <option value="new">새 데이터셋</option>
            </select>
          </div>
          <button
            onClick={handleUploadFiles}
            disabled={busy || !uploadFiles.length}
            className="mt-3 w-full rounded-2xl border-2 border-slate-900 bg-slate-900 px-4 py-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
          >
            선택 파일 데이터셋에 반영
          </button>
        </div>

        <div className="min-h-0 overflow-auto pr-1">
          <div className="space-y-3">
            {dashboard.dataset_versions.map((dataset) => (
              <div key={dataset.id} className="rounded-2xl border-2 border-slate-200 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-bold text-slate-900">{dataset.name}</div>
                    <div className="mt-1 text-xs text-slate-500">{dataset.id}</div>
                  </div>
                  <Badge tone="blue">{dataset.status}</Badge>
                </div>
                <div className="mt-2 grid grid-cols-3 gap-3 text-sm text-slate-600">
                  <div>샘플 {dataset.sample_count}</div>
                  <div>피드백 {dataset.feedback_count}</div>
                  <div>원본 {dataset.source_dataset_id ?? "-"}</div>
                </div>
                <div className="mt-2 text-sm text-slate-500">{dataset.notes || "메모 없음"}</div>
                <div className="mt-2 text-xs text-slate-400">{formatDate(dataset.updated_at)}</div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      <Card className="flex min-h-0 flex-col p-5">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <div className="text-xl font-bold text-slate-950">학습된 모델 리스트</div>
            <div className="mt-1 text-sm text-slate-500">production, staging, canary, 실패/중지 모델까지 이력으로 확인합니다.</div>
          </div>
          <Badge tone="green">{dashboard.model_versions.length} models</Badge>
        </div>

        <div className="min-h-0 overflow-auto pr-1">
          <div className="space-y-3">
            {dashboard.model_versions.map((model) => (
              <div key={model.id} className="rounded-2xl border-2 border-slate-200 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-bold text-slate-900">{model.id}</div>
                    <div className="mt-1 text-xs text-slate-500">{model.name}</div>
                  </div>
                  <Badge tone={model.status === "production" ? "green" : model.status === "canary" ? "amber" : model.status === "failed" || model.status === "stopped" ? "red" : "blue"}>
                    {model.status}
                  </Badge>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-3 text-sm text-slate-600">
                  <div>Data {model.dataset_version_id ?? "-"}</div>
                  <div>Base {model.base_model_version_id ?? "-"}</div>
                  <div>Recipe {model.recipe_id ?? "-"}</div>
                  <div>F1 {model.metrics.f1 ?? "-"}</div>
                </div>
                <div className="mt-2 text-sm text-slate-500">{model.lineage}</div>
                <div className="mt-2 text-xs text-slate-400">{formatDate(model.updated_at ?? model.created_at)}</div>
              </div>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}

function VersionView({ dashboard }: { dashboard: DashboardResponse }) {
  return (
    <div className="grid h-full min-h-0 grid-cols-[1fr_1fr] gap-4">
      <Card className="flex min-h-0 flex-col p-5">
        <div className="mb-4 text-xl font-bold text-slate-950">데이터 버전</div>
        <div className="min-h-0 overflow-auto pr-1">
          <div className="space-y-3">
            {dashboard.dataset_versions.map((dataset) => (
              <div key={dataset.id} className="rounded-2xl border-2 border-slate-200 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-bold text-slate-900">{dataset.id}</div>
                  <Badge tone="blue">{dataset.status}</Badge>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-3 text-sm text-slate-600">
                  <div>샘플 {dataset.sample_count}</div>
                  <div>피드백 {dataset.feedback_count}</div>
                </div>
                <div className="mt-2 text-sm text-slate-500">{dataset.notes || "메모 없음"}</div>
                <div className="mt-2 text-xs text-slate-400">{formatDate(dataset.updated_at)}</div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      <Card className="flex min-h-0 flex-col p-5">
        <div className="mb-4 text-xl font-bold text-slate-950">모델 버전</div>
        <div className="min-h-0 overflow-auto pr-1">
          <div className="space-y-3">
            {dashboard.model_versions.map((model) => (
              <div key={model.id} className="rounded-2xl border-2 border-slate-200 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-bold text-slate-900">{model.id}</div>
                  <Badge tone={model.status === "production" ? "green" : model.status === "canary" ? "amber" : "blue"}>
                    {model.status}
                  </Badge>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-3 text-sm text-slate-600">
                  <div>Data {model.dataset_version_id ?? "-"}</div>
                  <div>F1 {model.metrics.f1 ?? "-"}</div>
                </div>
                <div className="mt-2 text-sm text-slate-500">{model.lineage}</div>
                <div className="mt-2 text-xs text-slate-400">{formatDate(model.updated_at ?? model.created_at)}</div>
              </div>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}

function LogsView({ dashboard }: { dashboard: DashboardResponse }) {
  return (
    <div className="grid h-full min-h-0 grid-cols-[.95fr_1.05fr] gap-4">
      <Card className="flex min-h-0 flex-col p-5">
        <div className="mb-4 text-xl font-bold text-slate-950">시스템 로그</div>
        <div className="min-h-0 overflow-auto pr-1">
          <div className="space-y-3">
            {dashboard.logs.map((log) => (
              <div key={log.id} className="rounded-2xl border-2 border-slate-200 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="font-semibold text-slate-900">{log.message}</div>
                  <Badge tone={log.level === "warning" ? "amber" : log.level === "error" ? "red" : "blue"}>
                    {log.level}
                  </Badge>
                </div>
                <div className="mt-2 text-sm text-slate-500">{formatDate(log.time)}</div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      <Card className="flex min-h-0 flex-col p-5">
        <div className="mb-4 text-xl font-bold text-slate-950">인터페이스</div>
        <div className="min-h-0 overflow-auto pr-1">
          <div className="space-y-3">
            {Object.entries(dashboard.interfaces).map(([key, value]) => (
              <div key={key} className="rounded-2xl border-2 border-slate-200 p-4">
                <div className="text-xs font-bold uppercase tracking-[0.12em] text-slate-500">{key}</div>
                <div className="mt-2 text-sm text-slate-800">{value}</div>
              </div>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}

function MiniChart() {
  const max = Math.max(...CHART_BARS);
  return (
    <div className="flex h-[180px] items-end gap-2">
      {CHART_BARS.map((value, index) => (
        <div key={value + index} className="flex flex-1 flex-col items-center gap-2">
          <div
            className={cls("w-full rounded-t-lg", index >= 8 && index <= 10 ? "bg-amber-400" : "bg-blue-500")}
            style={{ height: `${(value / max) * 140}px` }}
          />
          <div className="text-[10px] text-slate-400">{index + 1}h</div>
        </div>
      ))}
    </div>
  );
}

function SummaryPage({ dashboard }: { dashboard: DashboardResponse }) {
  const productionModel = pickProductionModel(dashboard);
  const canaryModel = pickCanaryModel(dashboard);
  const latestFeedback = dashboard.feedback_items[0];

  return (
    <div className="grid h-full grid-rows-[auto_1fr] gap-4">
      <div className="grid grid-cols-4 gap-4">
        <Stat label="오늘 이상률" value={`${Math.min(99, dashboard.feedback_items.length + 3)}%`} sub="현장 피드백 기준" icon={AlertTriangle} tone="amber" />
        <Stat label="평균 추론시간" value={`${productionModel?.metrics.latency_ms ?? 53}ms`} sub="실시간 허용 범위" icon={Clock3} tone="blue" />
        <Stat label="운영 모델" value={productionModel?.id ?? "-"} sub="현재 Production" icon={Cpu} tone="slate" />
        <Stat label="개선 대기" value={`${dashboard.dataset_versions[0]?.feedback_count ?? 0}건`} sub="검수 후 재학습 예정" icon={Database} tone="green" />
      </div>

      <div className="grid min-h-0 grid-cols-[1.1fr_.9fr] gap-4">
        <Card className="p-5">
          <div className="mb-4 text-xl font-bold text-slate-950">현재 상황 요약</div>
          <MiniChart />
          <div className="mt-4 grid grid-cols-3 gap-3">
            <div className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
              <div className="text-xs text-slate-500">활성 데이터셋</div>
              <div className="mt-2 text-2xl font-bold text-slate-950">{dashboard.active_dataset_id}</div>
              <div className="mt-1 text-sm text-slate-500">누적 샘플 {dashboard.dataset_versions[0]?.sample_count ?? 0}</div>
            </div>
            <div className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
              <div className="text-xs text-slate-500">운영 안정도</div>
              <div className="mt-2 text-2xl font-bold text-slate-950">
                {dashboard.deployment.canary_model_id ? "92/100" : "97/100"}
              </div>
              <div className="mt-1 text-sm text-slate-500">로그 / 배포 상태 기반</div>
            </div>
            <div className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
              <div className="text-xs text-slate-500">배포 후보</div>
              <div className="mt-2 text-2xl font-bold text-slate-950">{canaryModel?.id ?? pickStagingModel(dashboard)?.id ?? "-"}</div>
              <div className="mt-1 text-sm text-slate-500">
                {dashboard.deployment.canary_line ? `${dashboard.deployment.canary_line} Canary` : "후보 없음"}
              </div>
            </div>
          </div>
        </Card>

        <div className="grid gap-4">
          <Card className="p-5">
            <div className="mb-4 text-xl font-bold text-slate-950">핵심 리스크</div>
            <div className="space-y-3">
              <div className="rounded-2xl border-2 border-amber-200 bg-amber-50 p-4">
                <div className="font-semibold text-amber-900">신뢰도 표시는 gate 확률 기준으로 고정</div>
                <div className="mt-1 text-sm text-amber-800">heatmap score는 별도 지표로 표시해 100% 초과 문제를 방지했습니다.</div>
              </div>
              <div className="rounded-2xl border-2 border-blue-200 bg-blue-50 p-4">
                <div className="font-semibold text-blue-900">Heatmap / Overlay 응답 형식 정렬</div>
                <div className="mt-1 text-sm text-blue-800">백엔드가 base64 heatmap과 overlay를 모두 반환하도록 맞췄습니다.</div>
              </div>
            </div>
          </Card>

          <Card className="p-5">
            <div className="mb-4 text-xl font-bold text-slate-950">오늘의 흐름</div>
            <div className="space-y-3 text-sm text-slate-700">
              <div className="rounded-2xl border-2 border-transparent bg-slate-50 p-4">
                현장 검사는 업로드 이미지 기준으로 즉시 추론하고 피드백까지 저장할 수 있습니다.
              </div>
              <div className="rounded-2xl border-2 border-transparent bg-slate-50 p-4">
                전산 화면에서는 새 학습, Canary 시작, 배포 승인, 롤백이 실제 상태를 갱신합니다.
              </div>
              <div className="rounded-2xl border-2 border-transparent bg-slate-50 p-4">
                최근 피드백: {latestFeedback ? `${latestFeedback.feedback_type} / ${formatDate(latestFeedback.created_at)}` : "아직 없음"}
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function AdminPage({
  adminTab,
  dashboard,
  onRefresh,
  onJumpToField,
  onOpenVersion,
  onOpenLogs,
}: {
  adminTab: AdminTab;
  dashboard: DashboardResponse;
  onRefresh: () => Promise<void>;
  onJumpToField: (line: LineId) => void;
  onOpenVersion: () => void;
  onOpenLogs: () => void;
}) {
  if (adminTab === "ops") {
    return (
      <OpsView
        dashboard={dashboard}
        onJumpToField={onJumpToField}
        onOpenVersion={onOpenVersion}
        onOpenLogs={onOpenLogs}
      />
    );
  }
  if (adminTab === "train") {
    return <TrainDeployViewV2 dashboard={dashboard} onRefresh={onRefresh} />;
  }
  if (adminTab === "version") {
    return <VersionViewV2 dashboard={dashboard} onRefresh={onRefresh} />;
  }
  return <LogsView dashboard={dashboard} />;
}

export default function App() {
  const [audience, setAudience] = useState<Audience>("field");
  const [adminTab, setAdminTab] = useState<AdminTab>("ops");
  const [selectedLine, setSelectedLine] = useState<LineId>("LINE-A");
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function refresh() {
    try {
      const response = await fetchDashboard();
      setDashboard(response);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "대시보드를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  const hasActiveTraining = Boolean(
    dashboard?.training_runs.some((run) => ["preparing", "running", "stopping"].includes(run.status))
  );

  useEffect(() => {
    if (!hasActiveTraining) return;
    const timer = window.setInterval(() => {
      void refresh();
    }, 1000);
    return () => window.clearInterval(timer);
  }, [hasActiveTraining]);

  let content: React.ReactNode = null;
  if (dashboard) {
    if (audience === "field") {
      content = <FieldPage selectedLine={selectedLine} dashboard={dashboard} onRefresh={refresh} />;
    } else if (audience === "admin") {
      content = (
        <AdminPage
          adminTab={adminTab}
          dashboard={dashboard}
          onRefresh={refresh}
          onJumpToField={(line) => {
            setSelectedLine(line);
            setAudience("field");
          }}
          onOpenVersion={() => setAdminTab("version")}
          onOpenLogs={() => setAdminTab("logs")}
        />
      );
    } else {
      content = <SummaryPage dashboard={dashboard} />;
    }
  }

  return (
    <div
      className={cls(
        "h-screen w-screen overflow-hidden font-['Pretendard','Inter','system-ui','sans-serif']",
        audience === "field" ? "bg-[#030712] text-slate-100" : "bg-slate-100 text-slate-900"
      )}
    >
      <div className="flex h-full flex-col">
        <TopBar
          audience={audience}
          setAudience={setAudience}
          adminTab={adminTab}
          setAdminTab={setAdminTab}
          selectedLine={selectedLine}
          setSelectedLine={setSelectedLine}
        />

        <main className={cls("min-h-0 flex-1 overflow-auto px-6 py-4", audience === "field" ? "bg-[#030712]" : "")}>
          {loading ? <MessageBanner message="대시보드를 불러오는 중입니다..." tone="slate" dark={audience === "field"} /> : null}
          {error ? <div className="mb-4"><MessageBanner message={error} tone="red" dark={audience === "field"} /></div> : null}
          {content}
        </main>
      </div>
    </div>
  );
}
