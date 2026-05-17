import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import type { Tone } from "../app/types";
import { cls } from "../app/utils";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <section className={cls("rounded-2xl border-2 border-slate-200 bg-white shadow-sm", className)}>
      {children}
    </section>
  );
}

export function DarkCard({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <section className={cls("rounded-2xl border-2 border-slate-800 bg-[#08111f] shadow-lg", className)}>
      {children}
    </section>
  );
}

export function Badge({
  children,
  tone = "slate",
  dark = false,
}: {
  children: ReactNode;
  tone?: Tone;
  dark?: boolean;
}) {
  const toneMap = dark
    ? {
        slate: "border-slate-700 bg-slate-900 text-slate-200",
        blue: "border-sky-800 bg-sky-950 text-sky-200",
        green: "border-emerald-800 bg-emerald-950 text-emerald-200",
        amber: "border-amber-800 bg-amber-950 text-amber-200",
        red: "border-rose-800 bg-rose-950 text-rose-200",
      }
    : {
        slate: "border-slate-200 bg-slate-100 text-slate-800",
        blue: "border-blue-200 bg-blue-100 text-blue-800",
        green: "border-emerald-200 bg-emerald-100 text-emerald-800",
        amber: "border-amber-200 bg-amber-100 text-amber-800",
        red: "border-rose-200 bg-rose-100 text-rose-800",
      };

  return (
    <span className={cls("inline-flex items-center rounded-full border px-3 py-1 text-sm font-bold", toneMap[tone])}>
      {children}
    </span>
  );
}

export function Stat({
  label,
  value,
  sub,
  icon: Icon,
  tone = "slate",
  dark = false,
}: {
  label: string;
  value: string;
  sub?: string;
  icon?: LucideIcon;
  tone?: Tone;
  dark?: boolean;
}) {
  const toneMap = dark
    ? {
        slate: "border-slate-800 bg-slate-950 text-slate-100",
        blue: "border-sky-900 bg-sky-950 text-sky-100",
        green: "border-emerald-900 bg-emerald-950 text-emerald-100",
        amber: "border-amber-900 bg-amber-950 text-amber-100",
        red: "border-rose-900 bg-rose-950 text-rose-100",
      }
    : {
        slate: "border-slate-200 bg-white text-slate-950",
        blue: "border-blue-100 bg-blue-50 text-blue-950",
        green: "border-emerald-100 bg-emerald-50 text-emerald-950",
        amber: "border-amber-100 bg-amber-50 text-amber-950",
        red: "border-rose-100 bg-rose-50 text-rose-950",
      };

  return (
    <div className={cls("min-w-0 rounded-2xl border-2 p-4", toneMap[tone])}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-bold text-slate-500">{label}</div>
          <div className="mt-2 truncate text-2xl font-black leading-tight">{value}</div>
          {sub ? <div className={cls("mt-1 truncate text-sm", dark ? "text-slate-400" : "text-slate-600")}>{sub}</div> : null}
        </div>
        {Icon ? <Icon className="mt-1 h-5 w-5 shrink-0 text-slate-500" /> : null}
      </div>
    </div>
  );
}

export function MessageBanner({
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
        slate: "border-slate-700 bg-slate-950 text-slate-200",
        blue: "border-sky-900 bg-sky-950 text-sky-200",
        green: "border-emerald-900 bg-emerald-950 text-emerald-200",
        amber: "border-amber-900 bg-amber-950 text-amber-200",
        red: "border-rose-900 bg-rose-950 text-rose-200",
      }
    : {
        slate: "border-slate-200 bg-slate-50 text-slate-800",
        blue: "border-blue-200 bg-blue-50 text-blue-800",
        green: "border-emerald-200 bg-emerald-50 text-emerald-800",
        amber: "border-amber-200 bg-amber-50 text-amber-800",
        red: "border-rose-200 bg-rose-50 text-rose-800",
      };

  return <div className={cls("rounded-2xl border-2 px-4 py-3 text-sm font-semibold", toneMap[tone])}>{message}</div>;
}

export function EmptyState({ children, dark = false }: { children: ReactNode; dark?: boolean }) {
  return (
    <div className={cls("rounded-2xl border-2 p-5 text-center text-sm font-semibold", dark ? "border-slate-800 text-slate-400" : "border-slate-200 text-slate-500")}>
      {children}
    </div>
  );
}

