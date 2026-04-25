"use client";

import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import type { ScoringResult } from "@/lib/api";
import { waterfallUrl } from "@/lib/api";
import { cleanFeatureName } from "@/lib/utils";
import Card from "./Card";

interface Props {
  data: ScoringResult;
}

const GRADIENT_COLORS = ["#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe"];

export default function Explainability({ data }: Props) {
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [imgLoading, setImgLoading] = useState(false);

  const barData = data.feature_importance.map((f) => ({
    name: cleanFeatureName(f.feature),
    value: f.importance,
  }));

  async function loadWaterfall(idx: number) {
    setSelectedIdx(idx);
    setImgLoading(true);
    setImgSrc(null);
    try {
      const res = await fetch(waterfallUrl(idx));
      if (!res.ok) throw new Error("Failed");
      const blob = await res.blob();
      setImgSrc(URL.createObjectURL(blob));
    } catch {
      setImgSrc(null);
    } finally {
      setImgLoading(false);
    }
  }

  return (
    <div className="p-6 space-y-5">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Global Feature Importance */}
        <Card title="Global Feature Importance">
          <div className="h-[420px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} layout="vertical" margin={{ left: 10, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2030" horizontal={false} />
                <XAxis type="number" tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={140}
                  tick={{ fill: "#d1d5db", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    background: "#12131c",
                    border: "1px solid #1e2030",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                  itemStyle={{ color: "#e5e7eb" }}
                  formatter={(v: number) => v.toFixed(6)}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={16}>
                  {barData.map((_, i) => {
                    const colorIdx = Math.min(Math.floor(i / 4), GRADIENT_COLORS.length - 1);
                    return <Cell key={i} fill={GRADIENT_COLORS[colorIdx]} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Waterfall */}
        <Card
          title="Individual SHAP Waterfall"
          action={
            <select
              value={selectedIdx ?? ""}
              onChange={(e) => {
                const v = e.target.value;
                if (v !== "") loadWaterfall(Number(v));
              }}
              className="px-3 py-1.5 bg-bg border border-border rounded-lg text-xs text-gray-300 outline-none focus:border-accent transition-colors max-w-[220px]"
            >
              <option value="">Select a customer...</option>
              {data.customers.map((c, i) => (
                <option key={c.customer_id} value={i}>
                  {c.customer_id} ({c.risk_segment} — {(c.churn_probability * 100).toFixed(1)}%)
                </option>
              ))}
            </select>
          }
        >
          <div className="min-h-[380px] flex items-center justify-center">
            {imgLoading && (
              <div className="text-center">
                <div className="w-10 h-10 border-[3px] border-border border-t-accent rounded-full animate-spin mx-auto mb-3" />
                <p className="text-xs text-gray-500">Generating plot...</p>
              </div>
            )}
            {!imgLoading && imgSrc && (
              <img
                src={imgSrc}
                alt="SHAP Waterfall Plot"
                className="max-w-full h-auto rounded-lg"
              />
            )}
            {!imgLoading && !imgSrc && (
              <p className="text-sm text-gray-600">Select a customer to view their SHAP waterfall plot</p>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
