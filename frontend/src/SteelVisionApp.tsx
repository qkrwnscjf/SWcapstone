import React, { useEffect, useState } from "react";
import { AlertTriangle, Cpu, Database, Factory, FileText, Layers, MonitorCog, Rocket, Upload } from "lucide-react";
import { predictAnomaly } from "./api/anomaly";
import { createTrainingRun, fetchDashboard, promoteModel, uploadArchitecture, uploadDatasetFiles, uploadFeedback } from "./api/mlops";
import type { PredictResponse } from "./types/anomaly";
import type { DashboardResponse } from "./types/mlops";

type Audience = "field" | "admin" | "summary";
type AdminTab = "ops" | "train" | "version" | "logs";
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const cx = (...v: Array<string | false | undefined>) => v.filter(Boolean).join(" ");
const dt = (v?: string | null) => (v ? new Date(v).toLocaleString("ko-KR", { timeZone: "Asia/Seoul" }) : "-");
const asset = (p: string) => (p.startsWith("http") ? p : `${API_BASE}${p}`);

function Card({ children, dark = false }: { children: React.ReactNode; dark?: boolean }) {
  return <div className={cx("rounded-3xl border-2 p-5", dark ? "border-slate-800 bg-[#08111f] text-slate-100" : "border-slate-200 bg-white")}>{children}</div>;
}

export default function SteelVisionApp() {
  const [audience, setAudience] = useState<Audience>("field");
  const [tab, setTab] = useState<AdminTab>("ops");
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [file, setFile] = useState<File | null>(null);
  const [rawUrl, setRawUrl] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [overlayUrl, setOverlayUrl] = useState("");
  const [heatmapUrl, setHeatmapUrl] = useState("");
  const [mode, setMode] = useState<"raw" | "overlay" | "heatmap">("raw");
  const [fieldMsg, setFieldMsg] = useState("");
  const [bulkFiles, setBulkFiles] = useState<File[]>([]);
  const [uploadMsg, setUploadMsg] = useState("");
  const [trainMsg, setTrainMsg] = useState("");
  const [archMsg, setArchMsg] = useState("");

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

  useEffect(() => {
    refresh();
  }, []);

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
      await uploadFeedback({ file, feedbackType: type, label, operator: "operator-01", comment: type, line: "LINE-A", predictedLabel: result?.decision ?? "" });
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

  async function addArch(kind: "gate" | "heatmap", input: HTMLInputElement | null, name: string) {
    const f = input?.files?.[0];
    if (!f || !name.trim()) return setArchMsg("이름과 파일을 함께 선택해 주세요.");
    try {
      await uploadArchitecture({ file: f, kind, name });
      setArchMsg(`${kind} 구조가 등록되었습니다.`);
      await refresh();
    } catch (e) {
      setArchMsg(e instanceof Error ? e.message : "구조 등록 실패");
    }
  }

  async function createRun() {
    if (!dashboard) return;
    const gate = dashboard.architectures.find((a) => a.kind === "gate")?.id ?? "ARCH-GATE-EFFNETB0";
    const heatmap = dashboard.architectures.find((a) => a.kind === "heatmap")?.id ?? "ARCH-HEATMAP-PATCHCORE-R18";
    try {
      await createTrainingRun({ datasetVersionId: dashboard.active_dataset_id, gateArchitectureId: gate, heatmapArchitectureId: heatmap, trainStrategy: "cascade", notes: "UI created run" });
      setTrainMsg("학습 런이 생성되고 데이터 버전 스냅샷이 고정되었습니다.");
      await refresh();
    } catch (e) {
      setTrainMsg(e instanceof Error ? e.message : "런 생성 실패");
    }
  }

  async function promote(id: string) {
    await promoteModel(id, "production");
    await refresh();
  }

  const preview = mode === "overlay" && overlayUrl ? overlayUrl : mode === "heatmap" && heatmapUrl ? heatmapUrl : rawUrl;
  const production = dashboard?.model_versions.find((m) => m.status === "production");

  return (
    <div className={cx("h-screen w-screen overflow-hidden font-['Pretendard','Inter','system-ui','sans-serif']", audience === "field" ? "bg-[#030712] text-slate-100" : "bg-slate-100 text-slate-900")}>
      <div className="flex h-full flex-col">
        <header className="border-b border-slate-200 bg-white/95 px-6 py-4">
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="text-2xl font-black text-slate-950">이상탐지 + MLOps 통합 UI</div>
              <div className="text-sm text-slate-500">EfficientNet gate {'->'} PatchCore cascade</div>
            </div>
            <div className="flex gap-3">
              {[["field", "현장", Factory], ["admin", "전산", MonitorCog], ["summary", "요약", AlertTriangle]].map(([k, label, Icon]) => {
                const I = Icon as typeof Factory;
                return <button key={String(k)} onClick={() => setAudience(k as Audience)} className={cx("flex items-center gap-2 rounded-2xl px-4 py-3 text-sm font-semibold", audience === k ? "bg-blue-600 text-white" : "bg-white text-slate-700 border-2 border-slate-200")}><I className="h-4 w-4" />{label}</button>;
              })}
            </div>
          </div>
          {audience === "admin" ? <div className="mt-4 flex gap-3">{[["ops", "운영", MonitorCog], ["train", "학습·배포", Rocket], ["version", "버전", Layers], ["logs", "로그", FileText]].map(([k, label, Icon]) => { const I = Icon as typeof Factory; return <button key={String(k)} onClick={() => setTab(k as AdminTab)} className={cx("flex items-center gap-2 rounded-2xl px-4 py-2 text-sm font-semibold", tab === k ? "bg-blue-600 text-white" : "border-2 border-slate-200 bg-white text-slate-700")}><I className="h-4 w-4" />{label}</button>; })}</div> : null}
        </header>
        <main className="min-h-0 flex-1 overflow-auto p-6">
          {loading ? <div>불러오는 중...</div> : null}
          {error ? <div className="rounded-2xl border-2 border-rose-200 bg-rose-50 px-4 py-3 text-rose-700">{error}</div> : null}
          {dashboard && audience === "field" ? (
            <div className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
              <Card dark>
                <div className="mb-4 flex items-center justify-between"><div><div className="text-xl font-black">현장 이상탐지 HMI</div><div className="text-sm text-slate-400">추론 + 피드백 적재</div></div><div className="rounded-full border border-slate-700 px-3 py-1 text-xs">{result?.decision ?? "대기"}</div></div>
                <div className="mb-4 grid grid-cols-4 gap-3 text-center text-sm"><div className="rounded-2xl border border-slate-800 p-3">Gate<br />{Math.round((result?.gate_score ?? 0) * 100)}%</div><div className="rounded-2xl border border-slate-800 p-3">Heatmap<br />{result?.heatmap_score == null ? "-" : `${Math.round(result.heatmap_score * 100)}%`}</div><div className="rounded-2xl border border-slate-800 p-3">Latency<br />{Math.round(result?.latency.total_latency_ms ?? 0)}ms</div><div className="rounded-2xl border border-slate-800 p-3">Dataset<br />{dashboard.active_dataset_id}</div></div>
                <div className="mb-4 flex flex-wrap gap-3">
                  <label className="cursor-pointer rounded-2xl border-2 border-slate-700 bg-slate-900 px-4 py-3 text-sm font-semibold">이미지 선택<input type="file" accept="image/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (!f) return; setFile(f); setRawUrl(URL.createObjectURL(f)); setResult(null); setOverlayUrl(""); setHeatmapUrl(""); setMode("raw"); }} /></label>
                  <button onClick={runPredict} className="rounded-2xl border-2 border-cyan-400 bg-cyan-400 px-4 py-3 text-sm font-black text-slate-950">이상탐지 실행</button>
                  {(["raw", "overlay", "heatmap"] as const).map((m) => <button key={m} onClick={() => setMode(m)} className={cx("rounded-2xl px-4 py-3 text-sm font-semibold", mode === m ? "bg-cyan-400 text-slate-950" : "border-2 border-slate-800 bg-slate-950 text-slate-300")}>{m.toUpperCase()}</button>)}
                </div>
                <div className="h-[420px] overflow-hidden rounded-3xl border-2 border-slate-800 bg-slate-950">{preview ? <img src={preview} alt="preview" className="h-full w-full object-contain" /> : <div className="grid h-full place-items-center text-slate-400">이미지를 올리면 결과를 볼 수 있습니다.</div>}</div>
              </Card>
              <Card dark>
                <div className="mb-4 text-xl font-black">작업자 피드백</div>
                <div className="space-y-3">
                  <button onClick={() => sendFeedback("confirmed_anomaly", "anomaly")} className="w-full rounded-2xl bg-blue-600 px-4 py-3 text-sm font-bold">문제 확정으로 적재</button>
                  <button onClick={() => sendFeedback("false_positive", "normal")} className="w-full rounded-2xl bg-rose-600 px-4 py-3 text-sm font-bold">오탐으로 적재</button>
                  <button onClick={() => sendFeedback("false_negative", "anomaly")} className="w-full rounded-2xl bg-amber-600 px-4 py-3 text-sm font-bold">미탐으로 적재</button>
                </div>
                {fieldMsg ? <div className="mt-4 text-sm text-slate-300">{fieldMsg}</div> : null}
                <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-950 p-4 text-sm">현재 작업 데이터셋: {dashboard.active_dataset_id}<br />누적 샘플: {dashboard.dataset_versions[0]?.sample_count ?? 0}<br />누적 피드백: {dashboard.dataset_versions[0]?.feedback_count ?? 0}</div>
                <div className="mt-6 space-y-3">{dashboard.feedback_items.slice(0, 3).map((item) => <div key={item.id} className="rounded-2xl border border-slate-800 bg-slate-950 p-3"><div className="text-sm font-bold">{item.feedback_type}</div><div className="mt-1 text-sm text-slate-400">{item.comment}</div><div className="mt-2 text-xs text-slate-500">{dt(item.created_at)}</div></div>)}</div>
              </Card>
            </div>
          ) : null}
          {dashboard && audience === "admin" && tab === "ops" ? <div className="grid gap-4 lg:grid-cols-2"><Card><div className="mb-4 text-xl font-black">운영 현황</div><div className="grid grid-cols-4 gap-3 text-center text-sm"><div className="rounded-2xl border-2 border-slate-200 p-3">데이터<br />{dashboard.dataset_versions[0]?.sample_count ?? 0}</div><div className="rounded-2xl border-2 border-slate-200 p-3">피드백<br />{dashboard.feedback_items.length}</div><div className="rounded-2xl border-2 border-slate-200 p-3">런<br />{dashboard.training_runs.length}</div><div className="rounded-2xl border-2 border-slate-200 p-3">운영 모델<br />{production?.id ?? "-"}</div></div></Card><Card><div className="mb-4 text-xl font-black">최근 피드백</div><div className="space-y-3">{dashboard.feedback_items.map((f) => <div key={f.id} className="grid grid-cols-[72px_1fr] gap-3 rounded-2xl border-2 border-slate-200 p-3"><img src={asset(f.image_url)} alt={f.id} className="h-16 w-16 rounded-2xl object-cover" /><div><div className="font-semibold">{f.feedback_type}</div><div className="text-sm text-slate-500">{f.comment || "의견 없음"}</div></div></div>)}</div></Card></div> : null}
          {dashboard && audience === "admin" && tab === "train" ? <div className="grid gap-4 lg:grid-cols-[1.05fr_0.95fr]"><Card><div className="mb-4 text-xl font-black">데이터 적재</div><div onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); setBulkFiles((c) => [...c, ...Array.from(e.dataTransfer.files)]); }} className="rounded-3xl border-2 border-dashed border-blue-200 bg-blue-50 p-6"><div className="flex items-center gap-3"><div className="grid h-12 w-12 place-items-center rounded-2xl bg-blue-600 text-white"><Upload className="h-5 w-5" /></div><div><div className="font-black text-slate-950">drag & drop / 파일 선택</div><div className="text-sm text-slate-600">새 데이터를 계속 누적합니다.</div></div></div><div className="mt-4 flex gap-3"><label className="cursor-pointer rounded-2xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white">파일 선택<input type="file" multiple className="hidden" onChange={(e) => setBulkFiles((c) => [...c, ...Array.from(e.target.files ?? [])])} /></label><button onClick={uploadBulk} className="rounded-2xl border-2 border-slate-200 px-4 py-3 text-sm font-semibold">누적 적재</button></div><div className="mt-4 max-h-40 overflow-auto rounded-2xl bg-white p-4 text-sm text-slate-500">{bulkFiles.length ? bulkFiles.map((f) => <div key={`${f.name}-${f.lastModified}`}>{f.name}</div>) : "선택된 파일 없음"}</div></div><div className="mt-4 text-sm text-slate-600">{uploadMsg}</div><div className="mt-6 text-xl font-black">Custom 구조 등록</div><div className="mt-3 grid gap-3"><input id="gateName" placeholder="gate 구조 이름" className="rounded-2xl border-2 border-slate-200 px-3 py-3 text-sm" /><input id="gateFile" type="file" className="rounded-2xl border-2 border-slate-200 px-3 py-3 text-sm" /><button onClick={() => addArch("gate", document.getElementById("gateFile") as HTMLInputElement, (document.getElementById("gateName") as HTMLInputElement)?.value || "")} className="rounded-2xl border-2 border-slate-900 px-4 py-3 text-sm font-semibold">Gate 구조 등록</button><input id="heatName" placeholder="heatmap 구조 이름" className="rounded-2xl border-2 border-slate-200 px-3 py-3 text-sm" /><input id="heatFile" type="file" className="rounded-2xl border-2 border-slate-200 px-3 py-3 text-sm" /><button onClick={() => addArch("heatmap", document.getElementById("heatFile") as HTMLInputElement, (document.getElementById("heatName") as HTMLInputElement)?.value || "")} className="rounded-2xl border-2 border-slate-900 px-4 py-3 text-sm font-semibold">Heatmap 구조 등록</button><div className="text-sm text-slate-600">{archMsg}</div></div></Card><Card><div className="mb-4 text-xl font-black">학습 런 생성</div><div className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4 text-sm">활성 데이터셋: {dashboard.active_dataset_id}<br />Gate/Heatmap input interface: {dashboard.interfaces.gate_input}<br />기본 cascade: {dashboard.interfaces.cascade}</div><button onClick={createRun} className="mt-4 w-full rounded-2xl bg-blue-600 px-4 py-3 text-sm font-black text-white">새 학습 런 생성</button><div className="mt-3 text-sm text-slate-600">{trainMsg}</div><div className="mt-6 space-y-3">{dashboard.training_runs.map((r) => <div key={r.id} className="rounded-2xl border-2 border-slate-200 p-4"><div className="font-semibold">{r.id}</div><div className="text-sm text-slate-500">{r.dataset_version_id} · {r.lineage}</div></div>)}</div></Card></div> : null}
          {dashboard && audience === "admin" && tab === "version" ? <div className="grid gap-4 lg:grid-cols-2"><Card><div className="mb-4 text-xl font-black">데이터 버전</div><div className="space-y-3">{dashboard.dataset_versions.map((d) => <div key={d.id} className="rounded-2xl border-2 border-slate-200 p-4"><div className="font-semibold">{d.id}</div><div className="text-sm text-slate-500">{d.status} · {d.sample_count} samples · feedback {d.feedback_count}</div><div className="text-xs text-slate-400">{dt(d.updated_at)}</div></div>)}</div></Card><Card><div className="mb-4 text-xl font-black">모델 버전</div><div className="space-y-3">{dashboard.model_versions.map((m) => <div key={m.id} className="rounded-2xl border-2 border-slate-200 p-4"><div className="flex items-center justify-between"><div className="font-semibold">{m.id}</div><div className="text-sm text-slate-500">{m.status}</div></div><div className="text-sm text-slate-500">{m.lineage}</div>{m.status !== "production" ? <button onClick={() => promote(m.id)} className="mt-3 rounded-xl bg-blue-600 px-3 py-2 text-sm font-semibold text-white">Production 승격</button> : null}</div>)}</div></Card></div> : null}
          {dashboard && audience === "admin" && tab === "logs" ? <div className="grid gap-4 lg:grid-cols-2"><Card><div className="mb-4 text-xl font-black">시스템 로그</div><div className="space-y-3">{dashboard.logs.map((l) => <div key={l.id} className="rounded-2xl border-2 border-slate-200 p-4"><div className="font-semibold">{l.message}</div><div className="text-xs text-slate-400">{dt(l.time)}</div></div>)}</div></Card><Card><div className="mb-4 text-xl font-black">인터페이스</div><div className="space-y-3">{Object.entries(dashboard.interfaces).map(([k, v]) => <div key={k} className="rounded-2xl border-2 border-slate-200 p-4"><div className="text-xs font-bold uppercase tracking-[0.12em] text-slate-500">{k}</div><div className="mt-2 text-sm">{v}</div></div>)}</div></Card></div> : null}
          {dashboard && audience === "summary" ? <div className="grid gap-4 lg:grid-cols-4"><Card><div className="text-sm text-slate-500">작업 데이터셋</div><div className="mt-2 text-3xl font-black">{dashboard.active_dataset_id}</div></Card><Card><div className="text-sm text-slate-500">누적 샘플</div><div className="mt-2 text-3xl font-black">{dashboard.dataset_versions[0]?.sample_count ?? 0}</div></Card><Card><div className="text-sm text-slate-500">운영 모델</div><div className="mt-2 text-3xl font-black">{production?.id ?? "-"}</div></Card><Card><div className="text-sm text-slate-500">최근 런</div><div className="mt-2 text-3xl font-black">{dashboard.training_runs[0]?.id ?? "-"}</div></Card></div> : null}
        </main>
      </div>
    </div>
  );
}
