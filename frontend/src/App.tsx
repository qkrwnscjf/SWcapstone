import React, { useMemo, useState } from 'react';
import { predictAnomaly } from './api/anomaly';

import {
  AlertTriangle,
  Briefcase,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
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
} from 'lucide-react';

function cls(...names) {
  return names.filter(Boolean).join(' ');
}

const audienceItems = [
  { key: 'field', label: '현장', desc: '작업자/검사 HMI', icon: Factory },
  { key: 'admin', label: '전산', desc: '운영/NOC', icon: MonitorCog },
  { key: 'summary', label: '요약', desc: '임원/KPI', icon: Briefcase },
];

const adminTabs = [
  { key: 'ops', label: '운영', icon: MonitorCog },
  { key: 'train', label: '학습·배포', icon: Rocket },
  { key: 'version', label: '버전', icon: Layers },
  { key: 'logs', label: '로그', icon: FileText },
];

const lines = ['LINE-A', 'LINE-B', 'LINE-C'];

const liveSamplesByLine = {
  'LINE-A': [
    {
      id: 'INF-20260307-001',
      sampleId: 'SMP-A1-1042',
      camera: 'CAM-01',
      ts: '10:14:22',
      label: '이상',
      confidence: 0.84,
      threshold: 0.68,
      inferMs: 44,
      status: '정상',
      issueType: '표면 스크래치 추정',
      model: 'MODEL-2.1.0',
    },
    {
      id: 'INF-20260307-002',
      sampleId: 'SMP-A1-1043',
      camera: 'CAM-01',
      ts: '10:14:24',
      label: '정상',
      confidence: 0.97,
      threshold: 0.68,
      inferMs: 41,
      status: '정상',
      issueType: '-',
      model: 'MODEL-2.1.0',
    },
  ],
  'LINE-B': [
    {
      id: 'INF-20260307-101',
      sampleId: 'SMP-B3-2288',
      camera: 'CAM-03',
      ts: '10:14:31',
      label: '이상',
      confidence: 0.61,
      threshold: 0.68,
      inferMs: 96,
      status: '지연',
      issueType: '반사/조명 영향 가능',
      model: 'MODEL-2.1.0',
    },
    {
      id: 'INF-20260307-102',
      sampleId: 'SMP-B3-2289',
      camera: 'CAM-03',
      ts: '10:14:36',
      label: '정상',
      confidence: 0.92,
      threshold: 0.68,
      inferMs: 89,
      status: '지연',
      issueType: '-',
      model: 'MODEL-2.1.0',
    },
  ],
  'LINE-C': [
    {
      id: 'INF-20260307-201',
      sampleId: 'SMP-C2-0102',
      camera: 'CAM-02',
      ts: '10:15:10',
      label: '정상',
      confidence: 0.95,
      threshold: 0.68,
      inferMs: 51,
      status: '정상',
      issueType: '-',
      model: 'MODEL-2.1.0',
    },
    {
      id: 'INF-20260307-202',
      sampleId: 'SMP-C2-0103',
      camera: 'CAM-02',
      ts: '10:15:13',
      label: '정상',
      confidence: 0.94,
      threshold: 0.68,
      inferMs: 49,
      status: '정상',
      issueType: '-',
      model: 'MODEL-2.1.0',
    },
  ],
};

const lineStatus = [
  { line: 'LINE-A', state: '정상', yield: 99.1, latency: 46, camera: '6/6', alert: '없음' },
  { line: 'LINE-B', state: '주의', yield: 97.8, latency: 92, camera: '5/6', alert: 'P95 지연' },
  { line: 'LINE-C', state: '정상', yield: 98.7, latency: 51, camera: '4/4', alert: '없음' },
];

const trainingRuns = [
  { run: 'RUN-260307-01', model: 'EfficientAD-v4', data: 'DATA-2026.03.07-r2', f1: 0.941, auc: 0.987, speed: '43ms', status: '완료' },
  { run: 'RUN-260307-02', model: 'PatchCore-r50', data: 'DATA-2026.03.07-r3', f1: 0.934, auc: 0.984, speed: '58ms', status: '학습중' },
];

const versions = {
  data: [
    { ver: 'DATA-2026.03.07-r3', total: '14,850', pending: 420, note: '조명 교체 이후 반영' },
    { ver: 'DATA-2026.03.07-r2', total: '14,110', pending: 390, note: 'FP/FN 샘플 증량' },
  ],
  model: [
    { ver: 'MODEL-2.2.0-rc1', data: 'r3', f1: 0.944, status: 'Staging' },
    { ver: 'MODEL-2.1.0', data: 'r2', f1: 0.938, status: 'Production' },
  ],
};

const logItems = [
  { time: '09:10', type: '주의', msg: 'LINE-B 지연이 기준치를 초과했습니다.' },
  { time: '09:22', type: '치명', msg: 'CAM-03 프레임 드랍이 감지되었습니다.' },
  { time: '09:31', type: '안내', msg: '작업자 재확인 요청이 등록되었습니다.' },
  { time: '09:42', type: '안내', msg: '조명 조정 후 지연이 안정화되고 있습니다.' },
  { time: '09:49', type: '주의', msg: 'LINE-B 입력 품질 변화가 관측되고 있습니다.' },
  { time: '10:01', type: '안내', msg: 'Staging 모델 검증 샘플이 추가되었습니다.' },
  { time: '10:12', type: '안내', msg: '재학습 후보 샘플 검수가 진행 중입니다.' },
];

const chartBars = [34, 48, 44, 57, 51, 63, 54, 46, 42, 58, 61, 49];

function toDisplayDecision(decision) {
  if (decision === 'anomaly') return '이상';
  if (decision === 'normal') return '정상';
  if (decision === 'normal (heatmap)') return '정상';
  return decision ?? '-';
}

function toIssueType(decision, heatmapCalled) {
  if (decision === 'anomaly') return '표면 이상 탐지';
  if (heatmapCalled) return 'heatmap 재확인 완료';
  return '-';
}

function Card({ children, className = '' }) {
  return (
    <div className={cls('rounded-3xl border-2 border-slate-200 bg-white shadow-[0_10px_28px_rgba(15,23,42,0.06)]', className)}>
      {children}
    </div>
  );
}

function DarkCard({ children, className = '' }) {
  return (
    <div className={cls('rounded-3xl border-2 border-slate-800 bg-[#08111f] shadow-[0_16px_36px_rgba(0,0,0,0.28)]', className)}>
      {children}
    </div>
  );
}

function Badge({ children, tone = 'slate', dark = false }) {
  const map = dark
    ? {
        slate: 'border-slate-700 bg-slate-900 text-slate-300',
        blue: 'border-sky-800 bg-sky-950/40 text-sky-300',
        green: 'border-emerald-800 bg-emerald-950/40 text-emerald-300',
        amber: 'border-amber-800 bg-amber-950/40 text-amber-300',
        red: 'border-rose-800 bg-rose-950/40 text-rose-300',
      }
    : {
        slate: 'bg-slate-100 text-slate-700',
        blue: 'bg-blue-100 text-blue-700',
        green: 'bg-emerald-100 text-emerald-700',
        amber: 'bg-amber-100 text-amber-700',
        red: 'bg-rose-100 text-rose-700',
      };

  return (
    <span className={cls('inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold', dark && 'border-2', map[tone])}>
      {children}
    </span>
  );
}

function Stat({ label, value, sub, icon: Icon, tone = 'slate', dark = false, compact = false }) {
  const toneCls = dark
    ? {
        slate: 'border-slate-800 bg-slate-950 text-slate-100',
        blue: 'border-sky-900 bg-sky-950/30 text-sky-200',
        green: 'border-emerald-900 bg-emerald-950/30 text-emerald-200',
        amber: 'border-amber-900 bg-amber-950/30 text-amber-200',
        red: 'border-rose-900 bg-rose-950/30 text-rose-200',
      }
    : {
        slate: 'bg-white',
        blue: 'bg-blue-50 border-blue-100',
        green: 'bg-emerald-50 border-emerald-100',
        amber: 'bg-amber-50 border-amber-100',
        red: 'bg-rose-50 border-rose-100',
      };

  return (
    <div className={cls('min-w-0 rounded-2xl border-2', compact ? 'px-3.5 py-2.5' : 'p-5', toneCls[tone])}>
      <div className="flex h-full items-start justify-between gap-2">
        <div className="min-w-0">
          <div className={cls('font-bold tracking-[0.12em] text-slate-500', compact ? 'text-[9px]' : 'text-[11px]')}>
            {label}
          </div>
          <div className={cls('mt-2 truncate font-black leading-none', compact ? 'text-[20px]' : 'text-3xl')}>
            {value}
          </div>
          {!compact && sub && (
            <div className={cls('mt-1 truncate text-[10px] leading-tight', dark ? 'text-slate-400' : 'text-slate-500')}>
              {sub}
            </div>
          )}
        </div>
        {Icon && <Icon className={cls('shrink-0 text-slate-500', compact ? 'mt-0.5 h-4 w-4' : 'mt-1 h-5 w-5')} />}
      </div>
    </div>
  );
}

function AudienceSwitch({ audience, setAudience }) {
  return (
    <div className="inline-flex h-[58px] rounded-[22px] border-[3px] border-slate-300 bg-white p-1.5 shadow-[0_10px_26px_rgba(15,23,42,0.08)]">
      {audienceItems.map((item) => {
        const Icon = item.icon;
        const active = audience === item.key;
        return (
          <button
            key={item.key}
            onClick={() => setAudience(item.key)}
            className={cls(
              'flex h-full min-w-[160px] items-center gap-3 rounded-[16px] border-2 border-transparent px-5 font-semibold transition',
              active
                ? 'bg-blue-600 text-white shadow-[0_8px_18px_rgba(37,99,235,0.35)]'
                : 'text-slate-700 hover:bg-slate-50'
            )}
          >
            <div className={cls('grid h-9 w-9 place-items-center rounded-2xl border-2', active ? 'border-white/10 bg-white/15' : 'border-transparent bg-slate-100')}>
              <Icon className="h-5 w-5" />
            </div>
            <div className="text-left leading-tight">
              <div className="text-sm font-bold">{item.label}</div>
              <div className={cls('text-[11px]', active ? 'text-blue-100' : 'text-slate-500')}>{item.desc}</div>
            </div>
          </button>
        );
      })}
    </div>
  );
}

function ContextControls({ audience, adminTab, setAdminTab, selectedLine, setSelectedLine }) {
  const [lineOpen, setLineOpen] = useState(false);

  if (audience === 'admin') {
    return (
      <div className="inline-flex h-[58px] rounded-[22px] border-2 border-slate-200 bg-white p-1.5 shadow-[0_10px_26px_rgba(15,23,42,0.06)]">
        {adminTabs.map((item) => {
          const Icon = item.icon;
          const active = adminTab === item.key;
          return (
            <button
              key={item.key}
              onClick={() => setAdminTab(item.key)}
              className={cls(
                'flex h-full items-center gap-2 rounded-[16px] border-2 border-transparent px-4 text-sm font-semibold transition',
                active ? 'bg-blue-600 text-white' : 'text-slate-700 hover:bg-slate-50'
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

  if (audience === 'field') {
    return (
      <div className="relative">
        <button
          onClick={() => setLineOpen((v) => !v)}
          className="inline-flex h-[58px] items-center gap-2 rounded-[22px] border-2 border-slate-200 bg-white px-4 text-sm font-semibold text-slate-700 shadow-[0_10px_26px_rgba(15,23,42,0.08)]"
        >
          <Factory className="h-4 w-4" />
          {selectedLine}
          <ChevronDown className={cls('h-4 w-4 transition', lineOpen && 'rotate-180')} />
        </button>

        {lineOpen && (
          <div className="absolute left-0 top-[64px] z-30 w-[180px] rounded-2xl border-2 border-slate-200 bg-white p-2 shadow-[0_12px_30px_rgba(15,23,42,0.12)]">
            {lines.map((line) => (
              <button
                key={line}
                onClick={() => {
                  setSelectedLine(line);
                  setLineOpen(false);
                }}
                className={cls(
                  'flex w-full items-center rounded-xl border-2 border-transparent px-3 py-2.5 text-left text-sm font-semibold transition',
                  selectedLine === line ? 'bg-blue-50 text-blue-700' : 'text-slate-700 hover:bg-slate-50'
                )}
              >
                {line}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }

  return <div className="h-[58px]" />;
}

function TopBar({ audience, setAudience, adminTab, setAdminTab, selectedLine, setSelectedLine }) {
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/95 px-7 py-4 backdrop-blur-xl">
      <div className="flex items-center justify-between gap-6">
        <div className="flex h-[58px] items-center text-[22px] font-black tracking-tight text-slate-900">
          (주)철강왕 MLOps 시스템
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

function HMIButton({ title, desc, tone, onClick }) {
  const toneCls = {
    blue: 'border-sky-950 bg-sky-900/80 text-sky-50 hover:bg-sky-900',
    red: 'border-rose-950 bg-rose-900/80 text-rose-50 hover:bg-rose-900',
    amber: 'border-amber-950 bg-amber-800/90 text-amber-50 hover:bg-amber-800',
    slate: 'border-slate-700 bg-slate-800 text-slate-100 hover:bg-slate-700',
  };

  return (
    <button
      onClick={onClick}
      className={cls(
        'flex h-full min-h-0 w-full flex-col overflow-hidden rounded-2xl border-2 px-4 py-4 text-left transition active:scale-[0.99]',
        toneCls[tone]
      )}
    >
      <div className="break-keep text-[17px] font-black leading-tight tracking-tight">{title}</div>
      <div className="mt-auto break-keep pt-3 text-[12px] leading-[1.35] opacity-90">{desc}</div>
    </button>
  );
}

function CameraViewport({ mode, rawImageUrl, overlayImageUrl, heatmapImageUrl }) {
  let imageSrc = rawImageUrl;

  if (mode === 'overlay' && overlayImageUrl) {
    imageSrc = overlayImageUrl;
  } else if (mode === 'heatmap' && heatmapImageUrl) {
    imageSrc = heatmapImageUrl;
  }

  return (
    <div className="relative h-full min-h-0 overflow-hidden rounded-[22px] border-2 border-slate-800 bg-[#040b17]">
      {!imageSrc ? (
        <>
          <div
            className="absolute inset-0 opacity-20"
            style={{
              backgroundImage:
                'linear-gradient(rgba(148,163,184,0.12) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.12) 1px, transparent 1px)',
              backgroundSize: '26px 26px',
            }}
          />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(30,41,59,0.32),rgba(2,6,23,0.94))]" />
          <div className="absolute inset-0 grid place-items-center text-sm text-slate-400">
            이미지를 업로드하면 검사 화면이 표시됩니다.
          </div>
        </>
      ) : (
        <img
          src={imageSrc}
          alt="inspection view"
          className="h-full w-full object-contain bg-slate-950"
        />
      )}
    </div>
  );
}

function FieldTopInfo({ current }) {
  return (
    <div className="ml-auto grid h-[68px] w-full max-w-[760px] grid-cols-4 gap-3">
      <Stat
        label="검사 상태"
        value={current.status}
        icon={CheckCircle2}
        tone={current.status === '정상' ? 'green' : 'amber'}
        dark
        compact
      />
      <Stat
        label="응답 속도"
        value={`${current.inferMs}ms`}
        icon={Clock3}
        tone="slate"
        dark
        compact
      />
      <Stat
        label="판정 결과"
        value={current.label}
        icon={ShieldAlert}
        tone={current.label === '이상' ? 'red' : 'green'}
        dark
        compact
      />
      <Stat
        label="신뢰도"
        value={`${Math.round(current.confidence * 100)}%`}
        icon={Cpu}
        tone="blue"
        dark
        compact
      />
    </div>
  );
}

function ModeSwitch({ mode, setMode }) {
  return (
    <div className="grid h-full grid-cols-3 gap-3">
      {['raw', 'heatmap', 'overlay'].map((m) => (
        <button
          key={m}
          onClick={() => setMode(m)}
          className={cls(
            'h-full rounded-2xl border-2 text-sm font-black tracking-wide transition',
            mode === m
              ? 'border-cyan-400 bg-cyan-400 text-slate-950 shadow-[0_10px_22px_rgba(34,211,238,0.22)]'
              : 'border-slate-800 bg-slate-950 text-slate-300 hover:bg-slate-900'
          )}
        >
          {m.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

function InfoMini({ label, value }) {
  return (
    <div className="rounded-xl border-2 border-slate-800 bg-slate-950 p-2.5">
      <div className="text-[10px] text-slate-500">{label}</div>
      <div className="mt-1 text-sm font-bold text-slate-100">{value}</div>
    </div>
  );
}

function FieldPage({ selectedLine }) {
  const [mode, setMode] = useState('raw');
  const [sampleIndex, setSampleIndex] = useState(0);

  const [selectedFile, setSelectedFile] = useState(null);
  const [rawImageUrl, setRawImageUrl] = useState('');
  const [overlayImageUrl, setOverlayImageUrl] = useState('');
  const [heatmapImageUrl, setHeatmapImageUrl] = useState('');
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictError, setPredictError] = useState('');
  const [predictResult, setPredictResult] = useState(null);

  const samples = liveSamplesByLine[selectedLine] ?? [];
  const fallbackCurrent = samples[sampleIndex] ?? {
    id: '-',
    sampleId: '-',
    camera: '-',
    ts: '-',
    label: '대기',
    confidence: 0,
    threshold: 0,
    inferMs: 0,
    status: '대기',
    issueType: '-',
    model: '-',
  };

  const current = useMemo(() => {
    if (!predictResult) return fallbackCurrent;

    return {
      ...fallbackCurrent,
      label: toDisplayDecision(predictResult.decision),
      confidence:
        predictResult.heatmap_score ??
        predictResult.gate_score ??
        0,
      inferMs: Math.round(predictResult.latency?.total_latency_ms ?? 0),
      status: isPredicting ? '검사 중' : '정상',
      issueType: toIssueType(
        predictResult.decision,
        predictResult.heatmap_called
      ),
      model: predictResult.heatmap_called ? 'Gate + PatchCore' : 'Gate',
    };
  }, [fallbackCurrent, predictResult, isPredicting]);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setPredictError('');
    setPredictResult(null);
    setOverlayImageUrl('');
    setHeatmapImageUrl('');

    const localUrl = URL.createObjectURL(file);
    setRawImageUrl(localUrl);
    setMode('raw');
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setPredictError('먼저 이미지를 업로드하세요.');
      return;
    }

    try {
      setIsPredicting(true);
      setPredictError('');

      const result = await predictAnomaly(selectedFile);
      setPredictResult(result);

      if (result.heatmap_overlay) {
        setOverlayImageUrl(`data:image/png;base64,${result.heatmap_overlay}`);
      }

      if (result.normalized_score_heatmap) {
        setHeatmapImageUrl(`data:image/png;base64,${result.normalized_score_heatmap}`);
      }

      if (result.heatmap_overlay) {
        setMode('overlay');
      }
    } catch (err) {
      setPredictError(err instanceof Error ? err.message : '추론 요청에 실패했습니다.');
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="grid h-full min-h-0 grid-cols-[1.2fr_0.8fr] gap-4">
      <DarkCard className="flex min-h-0 flex-col p-5">
        <div className="mb-4 flex items-center gap-3">
          <div className="text-xl font-bold text-white">{selectedLine} 실시간 검사</div>
          <Badge dark tone="blue">{current.model}</Badge>
          <div className="ml-auto">
            <FieldTopInfo current={current} />
          </div>
        </div>

        <div className="mb-4 grid grid-cols-[1fr_auto] gap-3">
          <label className="flex cursor-pointer items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900 px-4 py-3 text-sm font-semibold text-slate-200 hover:bg-slate-800">
            이미지 업로드
            <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
          </label>

          <button
            onClick={handlePredict}
            disabled={isPredicting}
            className="rounded-2xl border-2 border-cyan-400 bg-cyan-400 px-5 py-3 text-sm font-black text-slate-950 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isPredicting ? '검사 중...' : '이상탐지 실행'}
          </button>
        </div>

        {predictError && (
          <div className="mb-4 rounded-2xl border-2 border-rose-900 bg-rose-950/40 px-4 py-3 text-sm text-rose-200">
            {predictError}
          </div>
        )}

        <div className="mb-4 h-[56px]">
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
          <InfoMini label="샘플 ID" value={current.sampleId} />
          <InfoMini label="카메라" value={current.camera} />
          <InfoMini label="판정 결과" value={current.label} />
          <InfoMini label="추론 시간" value={`${current.inferMs}ms`} />
          <InfoMini label="이상 점수" value={`${Math.round((current.confidence ?? 0) * 100)}%`} />
          <InfoMini label="이상 유형" value={current.issueType} />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <HMIButton title="오탐 등록" desc="정상이지만 이상으로 판정된 샘플 등록" tone="blue" onClick={() => {}} />
          <HMIButton title="미탐 등록" desc="이상이지만 정상으로 판정된 샘플 등록" tone="red" onClick={() => {}} />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <HMIButton title="입력 품질 저하" desc="조명/초점/반사로 인해 품질 저하" tone="amber" onClick={() => {}} />
          <HMIButton title="판정 보류" desc="사람 확인이 필요한 샘플로 분류" tone="slate" onClick={() => {}} />
        </div>
      </DarkCard>
    </div>
  );
}

function ScrollTable({ children }) {
  return <div className="min-h-0 overflow-auto">{children}</div>;
}

function OpsView() {
  return (
    <div className="grid h-full min-h-0 grid-rows-[auto_1fr] gap-4">
      <div className="grid grid-cols-4 gap-4">
        <Stat label="운영 모델" value="MODEL-2.1.0" sub="3개 라인 운영" icon={Cpu} tone="blue" />
        <Stat label="최근 1시간 이상률" value="3.2%" sub="전일 대비 +0.4%p" icon={AlertTriangle} tone="amber" />
        <Stat label="오탐 피드백률" value="0.31%" sub="경고 기준 이내" icon={ShieldAlert} tone="slate" />
        <Stat label="시스템 상태" value="정상" sub="카메라 15/16 연결" icon={CheckCircle2} tone="green" />
      </div>

      <div className="grid min-h-0 grid-cols-[1.15fr_.85fr] gap-4">
        <Card className="flex min-h-0 flex-col p-5">
          <div className="mb-4 flex items-center justify-between">
            <div className="text-xl font-bold text-slate-950">라인별 운영 상태</div>
            <Badge tone="amber">우선 확인: LINE-B</Badge>
          </div>
          <div className="min-h-0 overflow-hidden rounded-2xl border-2 border-slate-200">
            <ScrollTable>
              <table className="min-w-full text-sm">
                <thead className="sticky top-0 bg-slate-50 text-slate-600">
                  <tr>
                    {['라인', '상태', '양품률', '평균 지연', '카메라', '알람'].map((h) => (
                      <th key={h} className="px-4 py-3 text-left font-semibold">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {lineStatus.map((line) => (
                    <tr key={line.line} className="border-t border-slate-200 bg-white">
                      <td className="px-4 py-3 font-semibold text-slate-900">{line.line}</td>
                      <td className="px-4 py-3"><Badge tone={line.state === '주의' ? 'amber' : 'green'}>{line.state}</Badge></td>
                      <td className="px-4 py-3">{line.yield}%</td>
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
              <button className="rounded-2xl border-2 border-blue-600 bg-blue-600 px-4 py-3 text-left text-sm font-semibold text-white">LINE-B 상세 보기</button>
              <button className="rounded-2xl border-2 border-slate-200 px-4 py-3 text-left text-sm font-semibold text-slate-800">문제 샘플 모아보기</button>
              <button className="rounded-2xl border-2 border-slate-200 px-4 py-3 text-left text-sm font-semibold text-slate-800">알람 로그 열기</button>
            </div>
          </Card>

          <Card className="flex min-h-0 flex-col p-5">
            <div className="mb-4 text-xl font-bold text-slate-950">최근 알림</div>
            <div className="min-h-0 overflow-auto pr-1">
              <div className="space-y-3">
                {logItems.map((log, i) => (
                  <div key={i} className="rounded-2xl border-2 border-slate-200 p-3">
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-sm font-semibold text-slate-900">{log.msg}</div>
                      <Badge tone={log.type === '치명' ? 'red' : log.type === '주의' ? 'amber' : 'blue'}>
                        {log.type}
                      </Badge>
                    </div>
                    <div className="mt-1 text-xs text-slate-500">{log.time}</div>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function TrainDeployView() {
  return (
    <div className="grid h-full min-h-0 grid-rows-[auto_1fr] gap-4">
      <div className="grid grid-cols-4 gap-4">
        <Stat label="학습 중인 모델" value="1개" sub="현재 실행 중" icon={Flame} tone="blue" />
        <Stat label="학습 완료된 모델" value="18개" sub="최근 7일 기준" icon={CheckCircle2} tone="green" />
        <Stat label="재학습 후보 데이터" value="54개" sub="검수 확정 샘플" icon={Database} tone="amber" />
        <Stat label="최근 학습된 배포 후보 모델" value="1개" sub="MODEL-2.2.0-rc1" icon={Rocket} tone="slate" />
      </div>

      <div className="grid min-h-0 grid-cols-[1fr_.95fr] gap-4">
        <Card className="flex min-h-0 flex-col p-5">
          <div className="mb-4 flex items-center justify-between">
            <div className="text-xl font-bold text-slate-950">학습</div>
            <button className="rounded-2xl border-2 border-blue-600 bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white">새 학습 실행</button>
          </div>
          <div className="min-h-0 overflow-hidden rounded-2xl border-2 border-slate-200">
            <ScrollTable>
              <table className="min-w-full text-sm">
                <thead className="sticky top-0 bg-slate-50 text-slate-600">
                  <tr>
                    {['Run', '모델', '데이터', 'F1', 'AUC', '속도', '상태'].map((h) => (
                      <th key={h} className="px-4 py-3 text-left font-semibold">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {trainingRuns.map((r) => (
                    <tr key={r.run} className="border-t border-slate-200">
                      <td className="px-4 py-3 font-semibold">{r.run}</td>
                      <td className="px-4 py-3">{r.model}</td>
                      <td className="px-4 py-3">{r.data}</td>
                      <td className="px-4 py-3">{r.f1}</td>
                      <td className="px-4 py-3">{r.auc}</td>
                      <td className="px-4 py-3">{r.speed}</td>
                      <td className="px-4 py-3">{r.status === '학습중' ? <Badge tone="amber">학습중</Badge> : <Badge tone="green">완료</Badge>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ScrollTable>
          </div>
        </Card>

        <Card className="p-5">
          <div className="mb-4 text-xl font-bold text-slate-950">배포</div>
          <div className="grid gap-4">
            <div className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
              <div className="text-sm font-semibold text-slate-800">Production</div>
              <div className="mt-2 text-2xl font-bold text-slate-950">MODEL-2.1.0</div>
              <div className="mt-2 text-sm text-slate-500">LINE-A, LINE-B, LINE-C</div>
            </div>
            <div className="rounded-2xl border-2 border-blue-200 bg-blue-50 p-4">
              <div className="text-sm font-semibold text-blue-900">Staging</div>
              <div className="mt-2 text-2xl font-bold text-blue-950">MODEL-2.2.0-rc1</div>
              <div className="mt-2 text-sm text-blue-700">F1 0.944 / 44ms / 대상 LINE-B</div>
            </div>
            <div className="grid grid-cols-3 gap-3">
              <button className="rounded-2xl border-2 border-slate-200 px-4 py-3 text-sm font-semibold text-slate-800">Canary 시작</button>
              <button className="rounded-2xl border-2 border-blue-600 bg-blue-600 px-4 py-3 text-sm font-semibold text-white">배포 승인</button>
              <button className="rounded-2xl border-2 border-rose-600 bg-rose-600 px-4 py-3 text-sm font-semibold text-white">롤백</button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

function VersionView() {
  return (
    <div className="grid h-full min-h-0 grid-cols-[1fr_1fr] gap-4">
      <Card className="flex min-h-0 flex-col p-5">
        <div className="mb-4 text-xl font-bold text-slate-950">데이터 버전</div>
        <div className="min-h-0 overflow-auto pr-1">
          <div className="space-y-3">
            {versions.data.map((d) => (
              <div key={d.ver} className="rounded-2xl border-2 border-slate-200 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-bold text-slate-900">{d.ver}</div>
                  <Badge tone="blue">데이터</Badge>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-3 text-sm text-slate-600">
                  <div>총 이미지 {d.total}</div>
                  <div>미확정 {d.pending}건</div>
                </div>
                <div className="mt-2 text-sm text-slate-500">{d.note}</div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      <Card className="flex min-h-0 flex-col p-5">
        <div className="mb-4 text-xl font-bold text-slate-950">모델 버전</div>
        <div className="min-h-0 overflow-auto pr-1">
          <div className="space-y-3">
            {versions.model.map((m) => (
              <div key={m.ver} className="rounded-2xl border-2 border-slate-200 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-bold text-slate-900">{m.ver}</div>
                  <Badge tone={m.status === 'Production' ? 'green' : 'blue'}>
                    {m.status === 'Production' ? '운영중' : '검증중'}
                  </Badge>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-3 text-sm text-slate-600">
                  <div>Data {m.data}</div>
                  <div>F1 {m.f1}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}

function LogsView() {
  return (
    <div className="grid h-full min-h-0 grid-cols-[.95fr_1.05fr] gap-4">
      <Card className="flex min-h-0 flex-col p-5">
        <div className="mb-4 text-xl font-bold text-slate-950">실험 결과 로그</div>
        <div className="min-h-0 overflow-auto pr-1">
          <div className="space-y-3">
            {trainingRuns.map((r) => (
              <div key={r.run} className="rounded-2xl border-2 border-slate-200 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="font-semibold text-slate-900">{r.run}</div>
                  <Badge tone={r.status === '학습중' ? 'amber' : 'green'}>{r.status}</Badge>
                </div>
                <div className="mt-2 text-sm text-slate-600">{r.model} · {r.data} · F1 {r.f1} · {r.speed}</div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      <Card className="flex min-h-0 flex-col p-5">
        <div className="mb-4 text-xl font-bold text-slate-950">오류 / 운영 로그</div>
        <div className="min-h-0 overflow-auto pr-1">
          <div className="space-y-3">
            {logItems.map((log, i) => (
              <div key={i} className="rounded-2xl border-2 border-slate-200 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="font-semibold text-slate-900">{log.msg}</div>
                  <Badge tone={log.type === '치명' ? 'red' : log.type === '주의' ? 'amber' : 'blue'}>
                    {log.type}
                  </Badge>
                </div>
                <div className="mt-2 text-sm text-slate-500">{log.time}</div>
              </div>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}

function AdminPage({ adminTab }) {
  if (adminTab === 'ops') return <OpsView />;
  if (adminTab === 'train') return <TrainDeployView />;
  if (adminTab === 'version') return <VersionView />;
  return <LogsView />;
}

function MiniChart() {
  const max = Math.max(...chartBars);
  return (
    <div className="flex h-[180px] items-end gap-2">
      {chartBars.map((v, i) => (
        <div key={i} className="flex flex-1 flex-col items-center gap-2">
          <div
            className={cls('w-full rounded-t-lg', i >= 8 && i <= 10 ? 'bg-amber-400' : 'bg-blue-500')}
            style={{ height: `${(v / max) * 140}px` }}
          />
          <div className="text-[10px] text-slate-400">{i + 1}h</div>
        </div>
      ))}
    </div>
  );
}

function SummaryPage() {
  return (
    <div className="grid h-full grid-rows-[auto_1fr] gap-4">
      <div className="grid grid-cols-4 gap-4">
        <Stat label="오늘 이상률" value="3.2%" sub="전일 대비 +0.4%p" icon={AlertTriangle} tone="amber" />
        <Stat label="평균 추론시간" value="53ms" sub="실시간 허용 범위" icon={Clock3} tone="blue" />
        <Stat label="운영 모델" value="1개" sub="Production 1개 운영" icon={Cpu} tone="slate" />
        <Stat label="개선 대기" value="54건" sub="검수 후 재학습 예정" icon={Database} tone="green" />
      </div>

      <div className="grid min-h-0 grid-cols-[1.1fr_.9fr] gap-4">
        <Card className="p-5">
          <div className="mb-4 text-xl font-bold text-slate-950">현재 상황 요약</div>
          <MiniChart />
          <div className="mt-4 grid grid-cols-3 gap-3">
            <div className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
              <div className="text-xs text-slate-500">품질</div>
              <div className="mt-2 text-2xl font-bold text-slate-950">98.8%</div>
              <div className="mt-1 text-sm text-slate-500">전체 라인 평균 양품률</div>
            </div>
            <div className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
              <div className="text-xs text-slate-500">운영 안정성</div>
              <div className="mt-2 text-2xl font-bold text-slate-950">94/100</div>
              <div className="mt-1 text-sm text-slate-500">지연/알람 종합 점수</div>
            </div>
            <div className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
              <div className="text-xs text-slate-500">배포 후보</div>
              <div className="mt-2 text-2xl font-bold text-slate-950">1건</div>
              <div className="mt-1 text-sm text-slate-500">MODEL-2.2.0-rc1</div>
            </div>
          </div>
        </Card>

        <div className="grid gap-4">
          <Card className="p-5">
            <div className="mb-4 text-xl font-bold text-slate-950">핵심 리스크</div>
            <div className="space-y-3">
              <div className="rounded-2xl border-2 border-amber-200 bg-amber-50 p-4">
                <div className="font-semibold text-amber-900">LINE-B 지연 증가</div>
                <div className="mt-1 text-sm text-amber-800">배포보다는 입력 품질 영향 가능성이 큽니다.</div>
              </div>
              <div className="rounded-2xl border-2 border-blue-200 bg-blue-50 p-4">
                <div className="font-semibold text-blue-900">Staging 모델 성능 우위</div>
                <div className="mt-1 text-sm text-blue-800">F1/지연 모두 현행 운영 모델보다 소폭 우수합니다.</div>
              </div>
            </div>
          </Card>

          <Card className="p-5">
            <div className="mb-4 text-xl font-bold text-slate-950">오늘의 흐름</div>
            <div className="space-y-3 text-sm text-slate-700">
              <div className="rounded-2xl border-2 border-transparent bg-slate-50 p-4">현장 검사와 피드백 수집은 정상 작동 중입니다.</div>
              <div className="rounded-2xl border-2 border-transparent bg-slate-50 p-4">전산에서는 LINE-B 지연과 검수 대기 36건을 보고 있습니다.</div>
              <div className="rounded-2xl border-2 border-transparent bg-slate-50 p-4">다음 액션은 입력 품질 확인 후 Staging 배포 판단입니다.</div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [audience, setAudience] = useState('field');
  const [adminTab, setAdminTab] = useState('ops');
  const [selectedLine, setSelectedLine] = useState('LINE-A');

  return (
    <div
      className={cls(
        "h-screen w-screen overflow-hidden font-['Pretendard','Inter','system-ui','sans-serif']",
        audience === 'field' ? 'bg-[#030712] text-slate-100' : 'bg-slate-100 text-slate-900'
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
        <main className={cls('min-h-0 flex-1 overflow-hidden px-6 py-4', audience === 'field' ? 'bg-[#030712]' : '')}>
          {audience === 'field' && <FieldPage selectedLine={selectedLine} />}
          {audience === 'admin' && <AdminPage adminTab={adminTab} />}
          {audience === 'summary' && <SummaryPage />}
        </main>
      </div>
    </div>
  );
}