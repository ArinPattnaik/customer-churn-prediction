"use client";

import { useRef } from "react";
import { Upload, Play, Shield, BarChart3, Brain, DollarSign } from "lucide-react";

interface Props {
  onDemo: () => void;
  onUpload: (file: File) => void;
}

export default function Welcome({ onDemo, onUpload }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);

  const features = [
    { icon: BarChart3, title: "Risk Scoring", desc: "ML-powered churn probability for every customer" },
    { icon: Brain, title: "SHAP Explanations", desc: "Understand what drives each prediction" },
    { icon: DollarSign, title: "Revenue Impact", desc: "Quantify revenue at risk and projected savings" },
    { icon: Shield, title: "Retention Strategies", desc: "Targeted recommendations per customer segment" },
  ];

  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="max-w-2xl text-center animate-fade-up">
        <div className="text-6xl mb-6">🛡️</div>
        <h2 className="text-3xl font-bold tracking-tight mb-3">
          Welcome to <span className="text-accent-light">ChurnGuard</span>
        </h2>
        <p className="text-gray-400 mb-10 max-w-md mx-auto leading-relaxed">
          Upload customer data or try the demo to explore churn predictions,
          SHAP explanations, and retention strategies.
        </p>

        <div className="flex items-center justify-center gap-3 mb-12">
          <label className="flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-medium border border-border text-gray-200 hover:border-gray-500 hover:bg-bg-hover cursor-pointer transition-all">
            <Upload size={18} />
            Upload CSV
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
            className="flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-medium bg-accent hover:bg-accent-light text-white transition-all"
          >
            <Play size={18} />
            Try Demo
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {features.map((f) => (
            <div
              key={f.title}
              className="p-4 rounded-xl bg-bg-card border border-border text-left"
            >
              <f.icon size={20} className="text-accent-light mb-2" />
              <h3 className="text-xs font-semibold mb-1">{f.title}</h3>
              <p className="text-[11px] text-gray-500 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
