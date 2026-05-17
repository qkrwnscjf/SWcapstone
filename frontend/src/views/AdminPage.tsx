import type { AdminTab } from "../app/types";
import type { DashboardResponse } from "../types/mlops";
import { LogsView } from "./LogsView";
import { OpsView } from "./OpsView";
import { TrainDeployView } from "./TrainDeployView";
import { VersionView } from "./VersionView";

export function AdminPage({
  adminTab,
  dashboard,
  onRefresh,
}: {
  adminTab: AdminTab;
  dashboard: DashboardResponse;
  onRefresh: () => Promise<void>;
}) {
  if (adminTab === "ops") return <OpsView dashboard={dashboard} />;
  if (adminTab === "train") return <TrainDeployView dashboard={dashboard} onRefresh={onRefresh} />;
  if (adminTab === "version") return <VersionView dashboard={dashboard} onRefresh={onRefresh} />;
  return <LogsView dashboard={dashboard} />;
}

