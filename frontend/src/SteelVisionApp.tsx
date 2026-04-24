import React, { useEffect, useState, useRef } from "react";
import { AlertTriangle, Cpu, Database, Factory, FileText, Layers, MonitorCog, Rocket, Upload, Trash2, RefreshCcw, CheckCircle2 } from "lucide-react";
import { predictAnomaly } from "./api/anomaly";
import { 
  createTrainingRun, 
  fetchDashboard, 
  deployModel, 
  uploadArchitecture, 
  uploadDatasetFiles, 
  uploadFeedback,
  fetchTrainingStatus,
  deleteFeedback,
  deleteTrainingRun
} from "./api/mlops";
import type { PredictResponse } from "./types/anomaly";
import type { DashboardResponse } from "./types/mlops";

type Audience = "field" | "admin" | "summary";
type AdminTab = "ops" | "train" | "version" | "logs";
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const cx = (...v: Array<string | false | undefined>) => v.filter(Boolean).join(" ");
const dt = (v?: string | null) => (v ? new Date(v).toLocaleString("ko-KR", { timeZone: "Asia/Seoul" }) : "-");
const asset = (p: string) => (p.startsWith("http") ? p : `${API_BASE}${p}`);

function Card({ children, dark = false, className = "" }: { children: React.ReactNode; dark?: boolean; className?: string }) {
  return <div className={cx("rounded-3xl border-2 p-5", dark ? "border-slate-800 bg-[#08111f] text-slate-100" : "border-slate-200 bg-white", className)}>{children}</div>;
}

export default function SteelVisionApp() {
  const [audience, setAudience] = useState<Audience>("field");
  const [tab, setTab] = useState<AdminTab>("ops");
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  
  // Inference State
  const [file, setFile] = useState<File | null>(null);
  const [rawUrl, setRawUrl] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [overlayUrl, setOverlayUrl] = useState("");
  const [heatmapUrl, setHeatmapUrl] = useState("");
  const [mode, setMode] = useState<"raw" | "overlay" | "heatmap">("raw");
  const [fieldMsg, setFieldMsg] = useState("");

  // MLOps State
  const [bulkFiles, setBulkFiles] = useState<File[]>([]);
  const [uploadMsg, setUploadMsg] = useState("");
  const [trainMsg, setTrainMsg] = useState("");
  const [ensembleEnabled, setEnsembleEnabled] = useState(true);
  const [trainingStatus, setTrainingStatus] = useState<any>({
    is_running: false,
    progress: 0,
    message: "IDLE",
    epoch: 0,
    metrics: { loss: 0, acc: 0 }
  });

  const pollInterval = useRef<NodeJS.Timeout | null>(null);

  async function refresh() {
    try {
      const data = await fetchDashboard();
      setDashboard(data);
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "불러오기 실패");
    } finally {
      setLoading(false);
    }
  }

  // Poll training status
  useEffect(() => {
    refresh();
    pollInterval.current = setInterval(async () => {
      try {
        const status = await fetchTrainingStatus();
        setTrainingStatus(status);
        if (!status.is_running && trainingStatus.is_running) {
          await refresh();
        }
      } catch (e) { /* ignore */ }
    }, 2000);
    return () => { if (pollInterval.current) clearInterval(pollInterval.current); };
  }, [trainingStatus.is_running]);

  async function runPredict() {
    if (!file) return setFieldMsg("이미지를 먼저 선택해 주세요.");
    try {
      const r = await predictAnomaly(file);
      setResult(r);
      setOverlayUrl(r.heatmap_overlay ? `data:image/png;base64,${r.heatmap_overlay}` : "");
      setHeatmapUrl(r.normalized_score_heatmap ? `data:image/png;base64,${r.normalized_score_heatmap}` : "");
      setMode(r.heatmap_overlay ? "overlay" : "raw");
      setFieldMsg("");
    } catch (e) {
      setFieldMsg(e instanceof Error ? e.message : "추론 실패");
    }
  }

  async function sendFeedback(type: string, label: string) {
    if (!file) return setFieldMsg("피드백에 사용할 이미지를 먼저 선택해 주세요.");
    try {
      await uploadFeedback({ 
        file, feedbackType: type, label, operator: "operator-01", comment: type, line: "LINE-A", 
        predictedLabel: result?.decision ?? "",
        gateScore: result?.gate_score,
        heatmapScore: result?.heatmap_score
      });
      setFieldMsg("작업자 피드백이 학습용 데이터셋에 누적되었습니다.");
      await refresh();
    } catch (e) {
      setFieldMsg(e instanceof Error ? e.message : "피드백 저장 실패");
    }
  }

  async function uploadBulk() {
    if (!bulkFiles.length) return setUploadMsg("파일을 먼저 선택해 주세요.");
    try {
      await uploadDatasetFiles({ files: bulkFiles, label: "unlabeled", sourceType: "bulk_upload", line: "LINE-A", comment: "bulk ingest" });
      setBulkFiles([]);
      setUploadMsg("데이터가 작업 데이터셋에 누적되었습니다.");
      await refresh();
    } catch (e) {
      setUploadMsg(e instanceof Error ? e.message : "업로드 실패");
    }
  }

  async function createRun() {
    try {
      setTrainMsg("학습 프로세스를 시작합니다...");
      await createTrainingRun({ epochs: 10, learningRate: 0.001 });
      setTrainMsg("");
    } catch (e) {
      setTrainMsg(e instanceof Error ? e.message : "런 생성 실패");
    }
  }

  async function promote(id: string, gateFile?: string, heatmapFile?: string) {
    if (!confirm(`모델 ${id}를 운영 환경으로 배포하시겠습니까?`)) return;
    try {
      await deployModel({ 
        model_id: id, 
        gate_file: gateFile, 
        heatmap_file: heatmapFile, 
        ensemble_enabled: ensembleEnabled 
      });
      await refresh();
      alert("배포 성공 (Hot-swap 완료)!");
    } catch (e) { alert("배포 실패"); }
  }

  async function removeFeedback(id: string) {
    if (!confirm("이 피드백 데이터를 삭제하시겠습니까?")) return;
    try {
      await deleteFeedback(id);
      await refresh();
    } catch (e) { alert("삭제 실패"); }
  }

  async function removeRun(id: string) {
    if (!confirm("이 학습 이력을 삭제하시겠습니까?")) return;
    try {
      await deleteTrainingRun(id);
      await refresh();
    } catch (e) { alert("삭제 실패"); }
  }

  const preview = mode === "overlay" && overlayUrl ? overlayUrl : mode === "heatmap" && heatmapUrl ? heatmapUrl : rawUrl;
  const production = dashboard?.model_versions.find((m) => m.status === "production");

  return (
    <div className={cx("h-screen w-screen overflow-hidden font-['Pretendard','Inter','system-ui','sans-serif']", audience === "field" ? "bg-[#030712] text-slate-100" : "bg-slate-100 text-slate-900")}>
      <div className="flex h-full flex-col">
        <header className="border-b border-slate-200 bg-white/95 px-6 py-4 shadow-sm">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="bg-blue-600 p-2 rounded-2xl text-white"><MonitorCog className="h-6 w-6" /></div>
              <div>
                <div className="text-xl font-black text-slate-950">SteelVision <span className="text-blue-600">MLOps</span></div>
                <div className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">EfficientNet + PatchCore Cascade</div>
              </div>
            </div>
            <div className="flex gap-2 bg-slate-100 p-1 rounded-2xl border border-slate-200">
              {[["field", "현장 모니터링", Factory], ["admin", "관리자 센터", MonitorCog], ["summary", "통계 요약", AlertTriangle]].map(([k, label, Icon]) => {
                const I = Icon as typeof Factory;
                return <button key={String(k)} onClick={() => setAudience(k as Audience)} className={cx("flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-bold transition-all", audience === k ? "bg-white text-blue-600 shadow-sm" : "text-slate-500 hover:text-slate-800")}><I className="h-3.5 w-3.5" />{label}</button>;
              })}
            </div>
          </div>
          {audience === "admin" ? <div className="mt-4 flex gap-2 border-t border-slate-100 pt-3">{[["ops", "실시간 운영", MonitorCog], ["train", "데이터 & 학습", Rocket], ["version", "버전 & 배포", Layers], ["logs", "시스템 로그", FileText]].map(([k, label, Icon]) => { const I = Icon as typeof Factory; return <button key={String(k)} onClick={() => setTab(k as AdminTab)} className={cx("flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-bold transition-all", tab === k ? "bg-slate-900 text-white shadow-lg" : "text-slate-500 hover:bg-slate-50")}><I className="h-3.5 w-3.5" />{label}</button>; })}</div> : null}
        </header>

        <main className="min-h-0 flex-1 overflow-auto p-6 bg-[#f8fafc]">
          {loading ? <div className="flex h-full items-center justify-center text-slate-400 font-bold animate-pulse">데이터 로딩 중...</div> : null}
          
          {dashboard && audience === "field" ? (
            <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr] max-w-7xl mx-auto">
              <Card dark className="shadow-2xl">
                <div className="mb-6 flex items-center justify-between">
                  <div>
                    <div className="text-2xl font-black">현장 이상탐지 HMI</div>
                    <div className="flex items-center gap-2 mt-1">
                      <div className="text-sm text-slate-500 font-medium tracking-tight">실시간 추론 및 시각화</div>
                      {result?.issueType && (
                        <div className="px-2 py-0.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-[10px] font-black text-blue-400 uppercase tracking-tighter animate-in fade-in slide-in-from-left-1">
                          AI 진단: {result.issueType}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className={cx("rounded-full px-4 py-1.5 text-xs font-black uppercase border", result?.decision === "anomaly" ? "bg-rose-500/10 border-rose-500 text-rose-500" : "bg-emerald-500/10 border-emerald-500 text-emerald-500")}>
                    {result?.decision ?? "IDLE"}
                  </div>
                </div>

                <div className="mb-6 grid grid-cols-4 gap-3 text-center">
                  {[["Gate", `${Math.round((result?.gate_score ?? 0) * 100)}%`], ["Heatmap", result?.heatmap_score == null ? "-" : `${Math.round(result.heatmap_score * 100)}%`], ["Latency", `${Math.round(result?.latency.total_latency_ms ?? 0)}ms`], ["Dataset", dashboard.active_dataset_id]].map(([l, v]) => (
                    <div key={l} className="rounded-2xl bg-slate-900 p-3"><div className="text-[10px] font-black text-slate-500 uppercase mb-1">{l}</div><div className="text-lg font-black text-white">{v}</div></div>
                  ))}
                </div>
                <div className="mb-6 flex flex-wrap gap-3">
                  <label className="cursor-pointer rounded-2xl bg-white text-slate-950 px-5 py-3 text-sm font-black shadow-lg">이미지 선택<input type="file" accept="image/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (!f) return; setFile(f); setRawUrl(URL.createObjectURL(f)); setResult(null); setOverlayUrl(""); setHeatmapUrl(""); setMode("raw"); }} /></label>
                  <button onClick={runPredict} className="rounded-2xl bg-blue-600 text-white px-6 py-3 text-sm font-black shadow-lg hover:bg-blue-700 transition-all">이상탐지 실행</button>
                  <div className="flex bg-slate-900 rounded-2xl p-1 border border-slate-800 ml-auto">
                    {(["raw", "overlay", "heatmap"] as const).map((m) => <button key={m} onClick={() => setMode(m)} className={cx("rounded-xl px-4 py-2 text-[10px] font-black uppercase transition-all", mode === m ? "bg-slate-700 text-white shadow-sm" : "text-slate-500 hover:text-slate-300")}>{m}</button>)}
                  </div>
                </div>
                <div className="aspect-video overflow-hidden rounded-3xl border-2 border-slate-800 bg-black flex items-center justify-center">
                  {preview ? <img src={preview} alt="preview" className="h-full w-full object-contain" /> : <div className="text-slate-700 font-black text-sm">WAITING FOR INPUT...</div>}
                </div>
              </Card>
              <Card dark>
                <div className="mb-6"><div className="text-xl font-black">작업자 피드백</div><div className="text-sm text-slate-500">정밀 학습을 위한 실사 데이터 누적</div></div>
                <div className="grid gap-3">
                  {[["confirmed_anomaly", "anomaly", "문제 확정", "bg-blue-600"], ["false_positive", "normal", "오탐 보고 (FP)", "bg-rose-600"], ["false_negative", "anomaly", "미탐 보고 (FN)", "bg-amber-600"]].map(([t, l, label, color]) => (
                    <button key={t} onClick={() => sendFeedback(t, l)} className={cx("w-full rounded-2xl px-5 py-4 text-xs font-black flex items-center justify-between transition-all", color, "hover:brightness-110")}>
                      {label} <CheckCircle2 className="h-4 w-4 opacity-50" />
                    </button>
                  ))}
                </div>
                {fieldMsg ? <div className="mt-4 p-3 rounded-xl bg-slate-900 text-[11px] font-bold text-blue-400 text-center animate-pulse">{fieldMsg}</div> : null}
                <div className="mt-8 space-y-4">
                  <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest">최근 피드백</div>
                  {dashboard.feedback_items.slice(0, 3).map((item) => (
                    <div key={item.id} className="rounded-2xl bg-slate-900/50 p-4 border border-slate-800/50">
                      <div className="text-xs font-black text-white mb-1 flex items-center gap-2"><div className={cx("w-1.5 h-1.5 rounded-full", item.feedback_type.includes("false") ? "bg-rose-500" : "bg-blue-500")}></div>{item.feedback_type}</div>
                      <div className="text-[10px] text-slate-500 font-bold">{item.comment}</div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          ) : null}

          {dashboard && audience === "admin" && tab === "ops" ? (
            <div className="grid gap-6 lg:grid-cols-2 max-w-7xl mx-auto">
              <Card>
                <div className="mb-6 text-xl font-black">운영 현황 요약</div>
                <div className="grid grid-cols-2 gap-4">
                  {[["수집 데이터", dashboard.dataset_versions[0]?.sample_count ?? 0, Database], ["사용자 피드백", dashboard.feedback_items.length, AlertTriangle], ["학습 실험 런", dashboard.training_runs.length, Rocket], ["운영 모델", production?.id ?? "-", Cpu]].map(([l, v, Icon]) => (
                    <div key={String(l)} className="p-5 rounded-3xl bg-slate-50 border border-slate-100 flex items-start gap-4">
                      <div className="bg-white p-3 rounded-2xl shadow-sm"><Icon className="h-5 w-5 text-blue-600" /></div>
                      <div><div className="text-[11px] font-black text-slate-400 uppercase">{String(l)}</div><div className="text-lg font-black text-slate-900">{String(v)}</div></div>
                    </div>
                  ))}
                </div>
              </Card>
              <Card>
                <div className="mb-6 text-xl font-black">최근 수집 피드백</div>
                <div className="space-y-3 max-h-[400px] overflow-auto pr-2">
                  {dashboard.feedback_items.map((f) => (
                    <div key={f.id} className="grid grid-cols-[80px_1fr_40px] gap-4 rounded-3xl border border-slate-100 bg-white p-3 items-center shadow-sm">
                      <img src={asset(f.image_url)} alt={f.id} className="h-16 w-20 rounded-2xl object-cover bg-slate-100" />
                      <div><div className="text-xs font-black text-slate-900">{f.feedback_type}</div><div className="text-[11px] text-slate-500 font-bold mt-1">{f.comment || "-"}</div></div>
                      <button onClick={() => removeFeedback(f.id)} className="grid h-10 w-10 place-items-center rounded-2xl text-slate-300 hover:text-rose-500 hover:bg-rose-50 transition-all"><Trash2 className="h-4 w-4" /></button>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          ) : null}

          {dashboard && audience === "admin" && tab === "train" ? (
            <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr] max-w-7xl mx-auto">
              <Card>
                <div className="mb-6 text-xl font-black">대량 데이터 적재</div>
                <div onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); setBulkFiles((c) => [...c, ...Array.from(e.dataTransfer.files)]); }} className="rounded-3xl border-3 border-dashed border-blue-100 bg-blue-50/30 p-10 flex flex-col items-center group transition-all">
                  <div className="h-16 w-16 rounded-3xl bg-blue-600 text-white shadow-xl flex items-center justify-center"><Upload className="h-7 w-7" /></div>
                  <div className="mt-6 text-center"><div className="text-lg font-black">Drag & Drop 또는 파일 선택</div><div className="text-sm text-slate-500 font-bold">학습용 원본 이미지를 업로드합니다.</div></div>
                  <div className="mt-6 flex gap-3">
                    <label className="cursor-pointer rounded-2xl bg-blue-600 text-white px-6 py-3 text-sm font-black shadow-lg">파일 선택<input type="file" multiple className="hidden" onChange={(e) => setBulkFiles((c) => [...c, ...Array.from(e.target.files ?? [])])} /></label>
                    <button onClick={uploadBulk} className="rounded-2xl border-2 border-slate-200 bg-white px-6 py-3 text-sm font-black text-slate-700">누적 적재 실행</button>
                  </div>
                </div>
                {uploadMsg && <div className="mt-4 text-center text-xs font-black text-blue-600 animate-pulse">{uploadMsg}</div>}
              </Card>
              <Card>
                <div className="mb-6 flex items-center justify-between"><div className="text-xl font-black">학습 엔진 제어</div>{trainingStatus.is_running && <RefreshCcw className="h-5 w-5 animate-spin text-blue-600" />}</div>
                <div className="space-y-6">
                  <div className={cx("rounded-3xl border-2 p-6 transition-all", trainingStatus.is_running ? "bg-blue-50 border-blue-100" : "bg-slate-50 border-slate-100")}>
                    <div className="flex items-center justify-between mb-4"><div className="text-xs font-black text-slate-400 uppercase tracking-widest">Progress</div><div className="text-xs font-black text-blue-600 italic">{trainingStatus.message}</div></div>
                    <div className="h-3 w-full bg-slate-200 rounded-full overflow-hidden"><div className="h-full bg-blue-600 transition-all duration-700 relative" style={{ width: `${trainingStatus.progress}%` }}></div></div>
                    <div className="mt-4 grid grid-cols-2 text-sm font-black text-slate-800"><div>{trainingStatus.progress}% COMPLETE</div><div className="text-right">EPOCH {trainingStatus.epoch}</div></div>
                    {trainingStatus.is_running && <div className="mt-4 pt-4 border-t border-blue-100 grid grid-cols-2 gap-4 text-center"><div><div className="text-[10px] font-black text-blue-400 uppercase tracking-tighter">Loss</div><div className="text-lg font-black text-blue-600">{trainingStatus.metrics.loss.toFixed(4)}</div></div><div><div className="text-[10px] font-black text-blue-400 uppercase tracking-tighter">Accuracy</div><div className="text-lg font-black text-blue-600">{(trainingStatus.metrics.acc * 100).toFixed(1)}%</div></div></div>}
                  </div>
                  <button onClick={createRun} disabled={trainingStatus.is_running} className={cx("w-full rounded-2xl py-5 text-sm font-black transition-all flex items-center justify-center gap-3", trainingStatus.is_running ? "bg-slate-100 text-slate-300 border-2 border-slate-200" : "bg-blue-600 text-white shadow-xl hover:bg-blue-700")}><Rocket className="h-5 w-5" /> 재학습 시작 (Retrain)</button>
                  {trainMsg && <div className="text-center text-xs font-bold text-slate-400 italic">※ {trainMsg}</div>}
                </div>
              </Card>
            </div>
          ) : null}

          {dashboard && audience === "admin" && tab === "version" ? (
            <div className="grid gap-6 lg:grid-cols-2 max-w-7xl mx-auto">
              <Card>
                <div className="mb-6 text-xl font-black tracking-tight flex items-center gap-2"><Database className="h-5 w-5 text-blue-600" /> 데이터 버전 이력</div>
                <div className="space-y-3">
                  {dashboard.dataset_versions.map((d) => (
                    <div key={d.id} className="rounded-3xl border border-slate-100 bg-white p-5 shadow-sm">
                      <div className="flex items-center justify-between mb-2"><div className="text-sm font-black text-slate-900 uppercase">{d.id}</div><div className="px-2 py-0.5 rounded-lg bg-emerald-100 text-[10px] font-black text-emerald-700 uppercase">{d.status}</div></div>
                      <div className="text-[11px] text-slate-500 font-bold">{d.sample_count} Samples · {d.feedback_count} Feedbacks</div>
                      <div className="text-[9px] text-slate-400 uppercase mt-2 font-black">{dt(d.updated_at)}</div>
                    </div>
                  ))}
                </div>
              </Card>
              <Card>
                <div className="mb-6 text-xl font-black tracking-tight flex items-center gap-2"><Layers className="h-5 w-5 text-blue-600" /> 운영 모델 관리</div>
                <div className="space-y-4">
                  {dashboard.model_versions.map((m) => {
                    const isProd = m.status === "production";
                    return (
                      <div key={m.id} className={cx("relative rounded-3xl border-2 p-5 transition-all", isProd ? "bg-blue-600 border-blue-600 shadow-xl" : "bg-white border-slate-100 shadow-sm")}>
                        <div className="flex items-center justify-between mb-2"><div className={cx("text-sm font-black uppercase", isProd ? "text-white" : "text-slate-900")}>{m.id}</div><div className={cx("px-2 py-0.5 rounded-lg text-[10px] font-black uppercase", isProd ? "bg-white/20 text-white" : "bg-slate-100 text-slate-500")}>{m.status}</div></div>
                        <div className={cx("text-[11px] font-bold", isProd ? "text-blue-100" : "text-slate-400")}>{m.lineage} · F1: {m.metrics.f1 ?? "-"}</div>
                        
                        <div className={cx("mt-4 grid grid-cols-2 gap-2 p-3 rounded-2xl", isProd ? "bg-white/10" : "bg-slate-50")}>
                          <div className="space-y-1">
                            <div className={cx("text-[9px] font-black uppercase ml-1", isProd ? "text-blue-100" : "text-slate-400")}>Gate File</div>
                            <select id={`gate-select-${m.id}`} className="w-full bg-white border border-slate-200 rounded-xl px-2 py-1.5 text-[10px] font-bold text-slate-700 outline-none focus:border-blue-400 shadow-sm">
                              <option value="">Default (.pt)</option>
                              {dashboard.available_model_files?.filter(f => f.includes("gate")).map(f => (
                                <option key={f} value={f}>{f}</option>
                              ))}
                            </select>
                          </div>
                          <div className="space-y-1">
                            <div className={cx("text-[9px] font-black uppercase ml-1", isProd ? "text-blue-100" : "text-slate-400")}>Heatmap File</div>
                            <select id={`hm-select-${m.id}`} className="w-full bg-white border border-slate-200 rounded-xl px-2 py-1.5 text-[10px] font-bold text-slate-700 outline-none focus:border-blue-400 shadow-sm">
                              <option value="">Default (.pt)</option>
                              {dashboard.available_model_files?.filter(f => f.includes("patchcore")).map(f => (
                                <option key={f} value={f}>{f}</option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div className="mt-5 flex items-center gap-3">

                          {!isProd ? (
                            <>
                              <button 
                                onClick={() => {
                                  const g = (document.getElementById(`gate-select-${m.id}`) as HTMLSelectElement)?.value;
                                  const h = (document.getElementById(`hm-select-${m.id}`) as HTMLSelectElement)?.value;
                                  promote(m.id, g, h);
                                }} 
                                className="flex-1 rounded-2xl bg-blue-600 text-white px-4 py-3 text-xs font-black shadow-lg"
                              >
                                배포 승격
                              </button>
                              <label className="flex items-center gap-2 text-[10px] font-black text-slate-400 cursor-pointer bg-slate-50 px-3 py-3 rounded-2xl border border-slate-100"><input type="checkbox" checked={ensembleEnabled} onChange={(e) => setEnsembleEnabled(e.target.checked)} className="rounded-full w-4 h-4 text-blue-600 border-slate-300" /> 앙상블</label>
                            </>
                          ) : <div className="flex-1 text-xs font-black text-white flex items-center gap-2"><CheckCircle2 className="h-4 w-4" /> LIVE IN PRODUCTION</div>}
                          <button onClick={() => removeRun(m.id)} className={cx("grid h-10 w-10 place-items-center rounded-2xl transition-all", isProd ? "bg-white/10 text-white hover:bg-white/20" : "bg-slate-50 text-slate-300 hover:text-rose-500")}><Trash2 className="h-4 w-4" /></button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </Card>
            </div>
          ) : null}

          {dashboard && audience === "admin" && tab === "logs" ? (
            <div className="grid gap-6 lg:grid-cols-2 max-w-7xl mx-auto">
              <Card><div className="mb-6 text-xl font-black tracking-tight flex items-center gap-2"><FileText className="h-5 w-5 text-blue-600" /> 시스템 이벤트 로그</div><div className="space-y-3 max-h-[600px] overflow-auto pr-2">{dashboard.logs.map((l) => (<div key={l.id} className="rounded-2xl border border-slate-100 bg-white p-4 shadow-sm"><div className="flex items-start justify-between gap-3"><div className="flex-1 text-xs font-bold text-slate-700 leading-relaxed">{l.message}</div><div className="text-[9px] font-black text-slate-300 uppercase shrink-0">{dt(l.time)}</div></div></div>))}</div></Card>
              <Card><div className="mb-6 text-xl font-black tracking-tight">모델 아키텍처 명세</div><div className="space-y-3">{Object.entries(dashboard.interfaces || {}).map(([k, v]) => (<div key={k} className="p-5 rounded-3xl bg-slate-50 border border-slate-100"><div className="text-[10px] font-black text-slate-400 uppercase mb-2">{k}</div><div className="text-xs font-bold text-slate-900 font-mono bg-white p-3 rounded-xl border border-slate-100">{String(v)}</div></div>))}</div></Card>
            </div>
          ) : null}

          {dashboard && audience === "summary" ? (
            <div className="grid gap-6 grid-cols-1 md:grid-cols-2 lg:grid-cols-4 max-w-7xl mx-auto">
              {[["작업 데이터셋", dashboard.active_dataset_id, Database], ["누적 이미지", dashboard.dataset_versions[0]?.sample_count ?? 0, Layers], ["운영 모델", production?.id ?? "-", Cpu], ["최근 학습 런", dashboard.training_runs?.[0]?.id ?? "-", Rocket]].map(([l, v, Icon]) => (
                <Card key={String(l)} className="flex flex-col items-center text-center p-8 border-none shadow-xl shadow-blue-900/5">
                  <div className="h-12 w-12 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center mb-4"><Icon className="h-6 w-6" /></div>
                  <div className="text-[10px] font-black text-slate-400 uppercase mb-1">{String(l)}</div>
                  <div className="text-2xl font-black text-slate-900">{String(v)}</div>
                </Card>
              ))}
            </div>
          ) : null}
        </main>
      </div>
    </div>
  );
}
