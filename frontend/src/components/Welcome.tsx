"use client";

import { useRef } from "react";
import { Upload, Play, Shield, BarChart3, Brain, DollarSign, Sparkles, ArrowRight } from "lucide-react";

interface Props {
  onDemo: () => void;
  onUpload: (file: File) => void;
}

export default function Welcome({ onDemo, onUpload }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);

  const features = [
    { icon: BarChart3, title: "Risk Scoring", desc: "ML-powered churn probability for every customer", color: "from-blue-500/20 to-blue-600/5" },
    { icon: Brain, title: "SHAP Explanations", desc: "Understand what drives each prediction", color: "from-purple-500/20 to-purple-600/5" },
    { icon: DollarSign, title: "Revenue Impact", desc: "Quantify revenue at risk and projected savings", color: "from-emerald-500/20 to-emerald-600/5" },
    { icon: Shield, title: "Retention Strategies", desc: "Targeted recommendations per customer segment", color: "from-amber-500/20 to-amber-600/5" },
  ];

  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="max-w-2xl text-center animate-fade-up">
        {/* Glowing icon */}
        <div className="relative inline-block mb-8">
          <div className="text-6xl animate-pulse-glow rounded-2xl p-4">🛡️</div>
          <Sparkles size={16} className="absolute -top-1 -right-1 text-accent-light animate-pulse" />
        </div>

        <h2 className="text-4xl font-extrabold tracking-tight mb-4">
          Welcome to <span className="text-gradient">ChurnGuard</span>
        </h2>
        <p className="text-gray-400 mb-10 max-w-lg mx-auto leading-relaxed text-[15px]">
          Upload customer data or try the demo to explore AI-powered churn predictions,
          SHAP explanations, and retention strategies.
        </p>

        <div className="flex items-center justify-center gap-3 mb-14">
          <label className="group flex items-center gap-2 px-6 py-3.5 rounded-xl text-sm font-medium border border-border text-gray-200 hover:border-gray-500 hover:bg-bg-hover cursor-pointer transition-all">
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
            className="group flex items-center gap-2 px-6 py-3.5 rounded-xl text-sm font-medium bg-accent hover:bg-accent-light text-white transition-all shadow-lg shadow-accent/20 hover:shadow-accent/30"
          >
            <Play size={18} />
            Try Demo
            <ArrowRight size={14} className="opacity-0 -ml-2 group-hover:opacity-100 group-hover:ml-0 transition-all" />
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {features.map((f, i) => (
            <div
              key={f.title}
              className="group p-5 rounded-xl bg-bg-card border border-border text-left hover:border-gray-700 transition-all"
              style={{ animationDelay: `${i * 80}ms` }}
            >
              <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${f.color} flex items-center justify-center mb-3`}>
                <f.icon size={18} className="text-accent-light" />
              </div>
              <h3 className="text-xs font-semibold mb-1">{f.title}</h3>
              <p className="text-[11px] text-gray-500 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
