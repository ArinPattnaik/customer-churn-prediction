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
  blue: "bg-blue-500/10 text-blue-400",
  red: "bg-red-500/10 text-red-400",
  amber: "bg-amber-500/10 text-amber-400",
  emerald: "bg-emerald-500/10 text-emerald-400",
};

export default function KpiCard({ label, value, sub, icon: Icon, color }: Props) {
  return (
    <div className="bg-bg-card border border-border rounded-xl p-5 flex items-center gap-4 hover:border-gray-700 transition-colors animate-fade-up">
      <div className={cn("w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0", colorMap[color])}>
        <Icon size={22} />
      </div>
      <div>
        <p className="text-xs text-gray-500 font-medium">{label}</p>
        <p className="text-2xl font-bold tracking-tight">{value}</p>
        {sub && <p className="text-xs text-gray-600">{sub}</p>}
      </div>
    </div>
  );
}
