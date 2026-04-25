"use client";

import { useRef } from "react";
import { Upload, Play, Menu } from "lucide-react";
import type { Tab } from "@/app/page";

const titles: Record<Tab, { title: string; sub: string }> = {
  dashboard: { title: "Dashboard", sub: "Overview of churn risk and revenue impact" },
  customers: { title: "Customers", sub: "Detailed scores and retention strategies" },
  explainability: { title: "Explainability", sub: "SHAP feature importance and individual explanations" },
};

interface Props {
  tab: Tab;
  onUpload: (file: File) => void;
  onDemo: () => void;
  onMenuToggle: () => void;
}

export default function TopBar({ tab, onUpload, onDemo, onMenuToggle }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const { title, sub } = titles[tab];

  return (
    <header className="sticky top-0 z-30 flex items-center gap-4 px-6 h-16 bg-bg-card/80 backdrop-blur-md border-b border-border">
      <button className="lg:hidden text-gray-400 hover:text-white" onClick={onMenuToggle}>
        <Menu size={22} />
      </button>

      <div className="flex-1 min-w-0">
        <h1 className="text-base font-semibold truncate">{title}</h1>
        <p className="text-xs text-gray-500 truncate">{sub}</p>
      </div>

      <div className="flex items-center gap-2.5">
        <label className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-medium border border-border text-gray-300 hover:border-gray-600 hover:bg-bg-hover cursor-pointer transition-all">
          <Upload size={14} />
          <span className="hidden sm:inline">Upload CSV</span>
          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onUpload(f);
              e.target.value = "";
            }}
          />
        </label>
        <button
          onClick={onDemo}
          className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-medium bg-accent hover:bg-accent-light text-white transition-all"
        >
          <Play size={14} />
          <span className="hidden sm:inline">Demo Mode</span>
        </button>
      </div>
    </header>
  );
}
