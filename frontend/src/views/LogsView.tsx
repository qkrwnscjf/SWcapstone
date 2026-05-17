import type { DashboardResponse } from "../types/mlops";
import { formatDate } from "../app/utils";
import { Badge, Card, EmptyState } from "../components/ui";

export function LogsView({ dashboard }: { dashboard: DashboardResponse }) {
  return (
    <div className="grid h-full min-h-0 grid-cols-[.95fr_1.05fr] gap-4">
      <Card className="flex min-h-0 flex-col p-5">
        <div className="mb-4 text-xl font-black text-slate-950">시스템 로그</div>
        <div className="min-h-0 overflow-auto pr-1">
          <div className="space-y-3">
            {dashboard.logs.map((log) => (
              <div key={log.id} className="rounded-2xl border-2 border-slate-200 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="font-bold text-slate-900">{log.message}</div>
                  <Badge tone={log.level === "warning" ? "amber" : log.level === "error" ? "red" : "blue"}>{log.level}</Badge>
                </div>
                <div className="mt-2 text-sm font-semibold text-slate-500">{formatDate(log.time)}</div>
              </div>
            ))}
            {!dashboard.logs.length ? <EmptyState>표시할 로그가 없습니다.</EmptyState> : null}
          </div>
        </div>
      </Card>

      <Card className="flex min-h-0 flex-col p-5">
        <div className="mb-4 text-xl font-black text-slate-950">인터페이스</div>
        <div className="min-h-0 overflow-auto pr-1">
          <div className="space-y-3">
            {Object.entries(dashboard.interfaces).map(([key, value]) => (
              <div key={key} className="rounded-2xl border-2 border-slate-200 p-4">
                <div className="text-sm font-black text-slate-500">{key}</div>
                <div className="mt-2 text-base font-semibold text-slate-800">{value}</div>
              </div>
            ))}
            {!Object.keys(dashboard.interfaces).length ? <EmptyState>등록된 인터페이스가 없습니다.</EmptyState> : null}
          </div>
        </div>
      </Card>
    </div>
  );
}

