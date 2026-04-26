"use client";

import { cn } from "@/lib/utils";
import type { LucideIcon } from "lucide-react";

interface Props {
  label: string;
  value: string;
  sub?: string;
  icon: LucideIcon;
  color: "blue" | "red" | "amber" | "emerald";
}

const colorMap = {
  blue: { bg: "bg-blue-500/10", text: "text-blue-400", glow: "shadow-blue-500/5" },
  red: { bg: "bg-red-500/10", text: "text-red-400", glow: "shadow-red-500/5" },
  amber: { bg: "bg-amber-500/10", text: "text-amber-400", glow: "shadow-amber-500/5" },
  emerald: { bg: "bg-emerald-500/10", text: "text-emerald-400", glow: "shadow-emerald-500/5" },
};

export default function KpiCard({ label, value, sub, icon: Icon, color }: Props) {
  const c = colorMap[color];
  return (
    <div className={cn(
      "bg-bg-card border border-border rounded-xl p-5 flex items-center gap-4",
      "hover:border-gray-700 transition-all animate-fade-up",
      "hover:shadow-lg", c.glow
    )}>
      <div className={cn("w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0", c.bg, c.text)}>
        <Icon size={22} />
      </div>
      <div>
        <p className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider">{label}</p>
        <p className="text-2xl font-bold tracking-tight mt-0.5">{value}</p>
        {sub && <p className="text-xs text-gray-600 mt-0.5">{sub}</p>}
      </div>
    </div>
  );
}
