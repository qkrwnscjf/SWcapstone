import { Factory, FileText, Layers, MonitorCog, Rocket } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { AdminTab, Audience, LineId } from "./types";

export const LINES: LineId[] = ["LINE-A", "LINE-B", "LINE-C"];

export const audienceItems: Array<{ key: Audience; label: string; icon: LucideIcon }> = [
  { key: "field", label: "현장", icon: Factory },
  { key: "admin", label: "관리", icon: MonitorCog },
];

export const adminTabs: Array<{ key: AdminTab; label: string; icon: LucideIcon }> = [
  { key: "ops", label: "운영", icon: MonitorCog },
  { key: "train", label: "학습", icon: Rocket },
  { key: "version", label: "버전", icon: Layers },
  { key: "logs", label: "로그", icon: FileText },
];

export const baseLineStatus: Record<
  LineId,
  { state: string; yieldRate: number; latency: number; camera: string; alert: string }
> = {
  "LINE-A": { state: "정상", yieldRate: 99.1, latency: 46, camera: "6/6", alert: "없음" },
  "LINE-B": { state: "주의", yieldRate: 97.8, latency: 92, camera: "5/6", alert: "지연 증가" },
  "LINE-C": { state: "정상", yieldRate: 98.7, latency: 51, camera: "4/4", alert: "없음" },
};

