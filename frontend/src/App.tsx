import { useEffect, useState } from "react";
import { fetchDashboard } from "./api/mlops";
import type { AdminTab, Audience, LineId } from "./app/types";
import { cls } from "./app/utils";
import { TopBar } from "./components/TopBar";
import { MessageBanner } from "./components/ui";
import type { DashboardResponse } from "./types/mlops";
import { AdminPage } from "./views/AdminPage";
import { FieldPage } from "./views/FieldPage";

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
    dashboard?.training_runs.some((run) => ["preparing", "running", "stopping"].includes(run.status ?? ""))
  );

  useEffect(() => {
    if (!hasActiveTraining) return;
    const timer = window.setInterval(() => {
      void refresh();
    }, 1000);
    return () => window.clearInterval(timer);
  }, [hasActiveTraining]);

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
          {loading ? <MessageBanner message="대시보드를 불러오는 중입니다." tone="slate" dark={audience === "field"} /> : null}
          {error ? <div className="mb-4"><MessageBanner message={error} tone="red" dark={audience === "field"} /></div> : null}
          {dashboard && audience === "field" ? <FieldPage selectedLine={selectedLine} dashboard={dashboard} onRefresh={refresh} /> : null}
          {dashboard && audience === "admin" ? <AdminPage adminTab={adminTab} dashboard={dashboard} onRefresh={refresh} /> : null}
        </main>
      </div>
    </div>
  );
}

