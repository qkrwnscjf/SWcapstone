import { AlertTriangle, Cpu, Database, Flame } from "lucide-react";
import type { DashboardResponse } from "../types/mlops";
import { baseLineStatus, LINES } from "../app/constants";
import { asset, formatDate, pickCanaryModel, pickProductionModel, pickStagingModel } from "../app/utils";
import { Badge, Card, EmptyState, Stat } from "../components/ui";

export function OpsView({ dashboard }: { dashboard: DashboardResponse }) {
  const productionModel = pickProductionModel(dashboard);
  const stagingModel = pickStagingModel(dashboard);
  const canaryModel = pickCanaryModel(dashboard);
  const activeDataset = dashboard.dataset_versions[0];
  const latestRun = dashboard.training_runs[0] ?? null;
  const activeRun = dashboard.training_runs.find((run) => ["preparing", "running", "stopping"].includes(run.status ?? ""));

  const lineStatus = LINES.map((lineId) => {
    const base = baseLineStatus[lineId];
    if (dashboard.deployment.canary_line === lineId) {
      return { lineId, ...base, state: "주의", alert: `Canary ${dashboard.deployment.canary_model_id ?? "-"}` };
    }
    return { lineId, ...base };
  });

  return (
    <div className="grid h-full min-h-0 grid-rows-[auto_1fr] gap-4">
      <div className="grid grid-cols-4 gap-4">
        <Stat label="운영 모델" value={productionModel?.id ?? "-"} sub={productionModel?.name ?? "Production"} icon={Cpu} tone="blue" />
        <Stat label="피드백" value={`${dashboard.feedback_items.length}건`} sub="학습 데이터 후보" icon={AlertTriangle} tone="amber" />
        <Stat label="데이터셋" value={activeDataset?.id ?? "-"} sub={`${activeDataset?.sample_count ?? 0} samples`} icon={Database} />
        <Stat label="학습 상태" value={activeRun ? "진행 중" : "대기"} sub={latestRun?.id ?? "최근 실행 없음"} icon={Flame} tone={activeRun ? "amber" : "green"} />
      </div>

      <div className="grid min-h-0 grid-cols-[1.15fr_.85fr] gap-4">
        <Card className="flex min-h-0 flex-col p-5">
          <div className="mb-4 flex items-center justify-between">
            <div className="text-xl font-black text-slate-950">라인 운영 상태</div>
            <Badge tone={dashboard.deployment.canary_model_id ? "amber" : "green"}>
              {dashboard.deployment.canary_model_id ? "Canary 검증" : "안정 운영"}
            </Badge>
          </div>
          <div className="min-h-0 overflow-auto rounded-2xl border-2 border-slate-200">
            <table className="min-w-full text-sm">
              <thead className="sticky top-0 bg-slate-50 text-slate-700">
                <tr>
                  {["라인", "상태", "수율", "응답", "카메라", "알림"].map((heading) => (
                    <th key={heading} className="px-4 py-3 text-left text-sm font-black">
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {lineStatus.map((line) => (
                  <tr key={line.lineId} className="border-t border-slate-200 bg-white">
                    <td className="px-4 py-3 font-bold text-slate-900">{line.lineId}</td>
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
          </div>
        </Card>

        <div className="grid min-h-0 grid-rows-[auto_1fr] gap-4">
          <Card className="p-5">
            <div className="mb-4 text-xl font-black text-slate-950">운영 체크</div>
            <div className="grid gap-3 text-base">
              <div className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="font-black text-slate-900">Runtime Artifact</div>
                <div className="mt-2 grid gap-1 text-slate-700">
                  <div>Gate: {dashboard.runtime_config.gate_file ?? "-"}</div>
                  <div>Heatmap: {dashboard.runtime_config.heatmap_file ?? "-"}</div>
                </div>
              </div>
              <div className="rounded-2xl border-2 border-slate-200 bg-slate-50 p-4">
                <div className="font-black text-slate-900">다음 결정</div>
                <div className="mt-2 text-slate-700">
                  {canaryModel
                    ? `${dashboard.deployment.canary_line ?? "라인"} Canary 결과 확인`
                    : stagingModel
                      ? `${stagingModel.id} 배포 승인 검토`
                      : "대기 중인 배포 후보 없음"}
                </div>
              </div>
            </div>
          </Card>

          <Card className="flex min-h-0 flex-col p-5">
            <div className="mb-4 text-xl font-black text-slate-950">최근 피드백</div>
            <div className="min-h-0 overflow-auto pr-1">
              <div className="space-y-3">
                {dashboard.feedback_items.map((item) => (
                  <div key={item.id} className="grid grid-cols-[72px_1fr] gap-3 rounded-2xl border-2 border-slate-200 p-3">
                    <img src={asset(item.image_url)} alt={item.id} className="h-16 w-16 rounded-2xl bg-slate-100 object-cover" />
                    <div className="min-w-0">
                      <div className="font-bold text-slate-900">{item.feedback_type}</div>
                      <div className="mt-1 truncate text-sm text-slate-600">{item.comment || "메모 없음"}</div>
                      <div className="mt-2 text-sm text-slate-500">{formatDate(item.created_at)}</div>
                    </div>
                  </div>
                ))}
                {!dashboard.feedback_items.length ? <EmptyState>최근 피드백이 없습니다.</EmptyState> : null}
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

