"use client";

import { LayoutDashboard, Users, Box, X, Activity } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Tab } from "@/app/page";
import type { ModelInfo, ScoringResult } from "@/lib/api";

const navItems: { id: Tab; label: string; icon: typeof LayoutDashboard }[] = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "customers", label: "Customers", icon: Users },
  { id: "explainability", label: "Explainability", icon: Box },
];

interface Props {
  tab: Tab;
  setTab: (t: Tab) => void;
  modelInfo: ModelInfo | null;
  data: ScoringResult | null;
  open: boolean;
  onClose: () => void;
}

export default function Sidebar({ tab, setTab, modelInfo, data, open, onClose }: Props) {
  const info = data?.model_info ?? modelInfo;

  return (
    <>
      {/* Mobile overlay */}
      {open && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 lg:hidden animate-fade-in" onClick={onClose} />
      )}

      <aside
        className={cn(
          "fixed top-0 left-0 bottom-0 w-60 bg-bg-card border-r border-border flex flex-col z-50 transition-transform duration-300",
          "lg:translate-x-0",
          open ? "translate-x-0" : "-translate-x-full"
        )}
      >
        {/* Brand */}
        <div className="flex items-center justify-between px-5 py-5 border-b border-border">
          <div className="flex items-center gap-2.5">
            <span className="text-2xl">🛡️</span>
            <span className="text-lg font-bold tracking-tight">ChurnGuard</span>
          </div>
          <button className="lg:hidden text-gray-400 hover:text-white transition-colors" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-3 space-y-0.5">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => { setTab(item.id); onClose(); }}
              className={cn(
                "w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all",
                tab === item.id
                  ? "bg-accent/10 text-accent-light shadow-sm shadow-accent/5"
                  : "text-gray-400 hover:bg-bg-hover hover:text-gray-200"
              )}
            >
              <item.icon size={18} />
              {item.label}
            </button>
          ))}
        </nav>

        {/* Model badge */}
        <div className="p-4 border-t border-border">
          <div className="flex items-center gap-2.5 text-xs text-gray-500">
            <span
              className={cn(
                "w-2 h-2 rounded-full flex-shrink-0 transition-colors",
                info ? "bg-emerald-500 shadow-sm shadow-emerald-500/50" : "bg-amber-500 animate-pulse"
              )}
            />
            <span className="truncate">
              {info ? `${info.model_type} v${info.version}` : "Loading model..."}
            </span>
          </div>
        </div>
      </aside>
    </>
  );
}
