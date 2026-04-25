"use client";

import { Users, AlertTriangle, AlertCircle, CheckCircle, Download } from "lucide-react";
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from "recharts";
import type { ScoringResult } from "@/lib/api";
import { exportCsvUrl } from "@/lib/api";
import { cleanFeatureName, formatCurrency, formatNumber } from "@/lib/utils";
import KpiCard from "./KpiCard";
import Card from "./Card";

interface Props {
  data: ScoringResult;
}

const RISK_COLORS = ["#ef4444", "#f59e0b", "#22c55e"];

export default function Dashboard({ data }: Props) {
  const { summary: s, impact: imp, feature_importance: fi, model_info: mi } = data;
  const total = s.total_customers;

  const pieData = [
    { name: "High Risk", value: s.high_risk },
    { name: "Medium Risk", value: s.medium_risk },
    { name: "Low Risk", value: s.low_risk },
  ];

  const barData = fi.slice(0, 8).map((f) => ({
    name: cleanFeatureName(f.feature),
    value: f.importance,
  }));

  return (
    <div className="p-6 space-y-5">
      {/* KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <KpiCard label="Total Customers" value={formatNumber(total)} icon={Users} color="blue" />
        <KpiCard
          label="High Risk"
          value={formatNumber(s.high_risk)}
          sub={`${((s.high_risk / total) * 100).toFixed(1)}%`}
          icon={AlertTriangle}
          color="red"
        />
        <KpiCard
          label="Medium Risk"
          value={formatNumber(s.medium_risk)}
          sub={`${((s.medium_risk / total) * 100).toFixed(1)}%`}
          icon={AlertCircle}
          color="amber"
        />
        <KpiCard
          label="Low Risk"
          value={formatNumber(s.low_risk)}
          sub={`${((s.low_risk / total) * 100).toFixed(1)}%`}
          icon={CheckCircle}
          color="emerald"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card title="Risk Distribution">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius="55%"
                  outerRadius="80%"
                  paddingAngle={3}
                  dataKey="value"
                  stroke="none"
                >
                  {pieData.map((_, i) => (
                    <Cell key={i} fill={RISK_COLORS[i]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "#12131c",
                    border: "1px solid #1e2030",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                  itemStyle={{ color: "#e5e7eb" }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-6 -mt-2">
            {pieData.map((d, i) => (
              <div key={d.name} className="flex items-center gap-2 text-xs text-gray-400">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: RISK_COLORS[i] }} />
                {d.name}
              </div>
            ))}
          </div>
        </Card>

        <Card title="Top Feature Importance (SHAP)">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} layout="vertical" margin={{ left: 10, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2030" horizontal={false} />
                <XAxis type="number" tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={130}
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
                  formatter={(v: number) => v.toFixed(5)}
                />
                <Bar dataKey="value" fill="#6366f1" radius={[0, 4, 4, 0]} barSize={18} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Revenue Impact */}
      <Card
        title="Revenue Impact Analysis"
        action={
          <a
            href={exportCsvUrl()}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-border text-gray-400 hover:text-gray-200 hover:border-gray-600 transition-all"
          >
            <Download size={14} />
            Export CSV
          </a>
        }
      >
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
          <div>
            <p className="text-xs text-gray-500 font-medium mb-1">Revenue at Risk</p>
            <p className="text-xl font-bold text-red-400">{formatCurrency(imp.total_revenue_at_risk)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 font-medium mb-1">Targeted Customers</p>
            <p className="text-xl font-bold">{formatNumber(imp.targeted_customer_count)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 font-medium mb-1">Projected Savings</p>
            <p className="text-xl font-bold text-emerald-400">{formatCurrency(imp.projected_revenue_saved)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 font-medium mb-1">Retention ROI</p>
            <p className="text-xl font-bold text-blue-400">{imp.retention_roi.toFixed(2)}x</p>
          </div>
        </div>
      </Card>

      {/* Model Info */}
      <Card title="Model Details">
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          {[
            { label: "Type", value: mi.model_type },
            { label: "Version", value: mi.version },
            { label: "Trained", value: new Date(mi.training_date).toLocaleDateString() },
            { label: "Rows", value: formatNumber(mi.dataset_row_count) },
            { label: "AUC-ROC", value: mi.auc_roc?.toFixed(4) ?? "N/A" },
            { label: "Accuracy", value: mi.accuracy ? `${(mi.accuracy * 100).toFixed(1)}%` : "N/A" },
          ].map((item) => (
            <div key={item.label}>
              <p className="text-[11px] text-gray-500 font-medium">{item.label}</p>
              <p className="text-sm font-semibold">{item.value}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
