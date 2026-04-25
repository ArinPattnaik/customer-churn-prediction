"use client";

import { useState, useMemo } from "react";
import { Search } from "lucide-react";
import type { ScoringResult, Customer } from "@/lib/api";
import { cleanFeatureName, formatCurrency } from "@/lib/utils";
import { cn } from "@/lib/utils";
import Card from "./Card";

const PAGE_SIZE = 20;

const riskBadge: Record<string, string> = {
  High: "bg-red-500/10 text-red-400",
  Medium: "bg-amber-500/10 text-amber-400",
  Low: "bg-emerald-500/10 text-emerald-400",
};

function ProbBar({ value }: { value: number }) {
  const color = value >= 0.7 ? "#ef4444" : value >= 0.4 ? "#f59e0b" : "#22c55e";
  return (
    <div className="flex items-center gap-2">
      <div className="w-14 h-1.5 bg-border rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${value * 100}%`, background: color }} />
      </div>
      <span className="text-xs tabular-nums">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

interface Props {
  data: ScoringResult;
}

export default function Customers({ data }: Props) {
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("");
  const [page, setPage] = useState(1);

  const filtered = useMemo(() => {
    let list = data.customers;
    if (search) {
      const q = search.toLowerCase();
      list = list.filter((c) => c.customer_id.toLowerCase().includes(q));
    }
    if (riskFilter) {
      list = list.filter((c) => c.risk_segment === riskFilter);
    }
    return list;
  }, [data.customers, search, riskFilter]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const pageData = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  return (
    <div className="p-6">
      <Card
        title="Customer Scores"
        action={
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input
                type="text"
                placeholder="Search ID..."
                value={search}
                onChange={(e) => { setSearch(e.target.value); setPage(1); }}
                className="pl-8 pr-3 py-1.5 w-44 bg-bg border border-border rounded-lg text-xs text-gray-200 placeholder-gray-600 outline-none focus:border-accent transition-colors"
              />
            </div>
            <select
              value={riskFilter}
              onChange={(e) => { setRiskFilter(e.target.value); setPage(1); }}
              className="px-3 py-1.5 bg-bg border border-border rounded-lg text-xs text-gray-300 outline-none focus:border-accent transition-colors"
            >
              <option value="">All Segments</option>
              <option value="High">High Risk</option>
              <option value="Medium">Medium Risk</option>
              <option value="Low">Low Risk</option>
            </select>
          </div>
        }
        bodyClassName="p-0"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-bg">
                <th className="px-5 py-3 text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Customer ID</th>
                <th className="px-5 py-3 text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Churn Prob.</th>
                <th className="px-5 py-3 text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Risk</th>
                <th className="px-5 py-3 text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Top Driver</th>
                <th className="px-5 py-3 text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Strategy</th>
                <th className="px-5 py-3 text-right text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Revenue</th>
              </tr>
            </thead>
            <tbody>
              {pageData.map((c) => (
                <tr key={c.customer_id} className="border-t border-border hover:bg-bg-hover transition-colors">
                  <td className="px-5 py-3 font-medium">{c.customer_id}</td>
                  <td className="px-5 py-3"><ProbBar value={c.churn_probability} /></td>
                  <td className="px-5 py-3">
                    <span className={cn("px-2.5 py-0.5 rounded-full text-[11px] font-semibold", riskBadge[c.risk_segment])}>
                      {c.risk_segment}
                    </span>
                  </td>
                  <td className="px-5 py-3 text-gray-400 text-xs">
                    {c.drivers[0] ? cleanFeatureName(c.drivers[0]) : "—"}
                  </td>
                  <td className="px-5 py-3 text-gray-400 text-xs max-w-[200px] truncate">
                    {c.retention_strategy || "—"}
                  </td>
                  <td className="px-5 py-3 text-right tabular-nums">{formatCurrency(c.annual_revenue)}</td>
                </tr>
              ))}
              {pageData.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-5 py-12 text-center text-gray-600">No customers match your filters</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-1 py-4 border-t border-border">
            <button
              disabled={safePage <= 1}
              onClick={() => setPage(safePage - 1)}
              className="px-3 py-1.5 text-xs rounded-md border border-border text-gray-400 hover:text-white hover:border-gray-600 disabled:opacity-30 disabled:cursor-default transition-all"
            >
              ‹ Prev
            </button>
            {Array.from({ length: Math.min(7, totalPages) }, (_, i) => {
              let p: number;
              if (totalPages <= 7) {
                p = i + 1;
              } else if (safePage <= 4) {
                p = i + 1;
              } else if (safePage >= totalPages - 3) {
                p = totalPages - 6 + i;
              } else {
                p = safePage - 3 + i;
              }
              return (
                <button
                  key={p}
                  onClick={() => setPage(p)}
                  className={cn(
                    "w-8 h-8 text-xs rounded-md border transition-all",
                    p === safePage
                      ? "bg-accent border-accent text-white"
                      : "border-border text-gray-500 hover:text-white hover:border-gray-600"
                  )}
                >
                  {p}
                </button>
              );
            })}
            <button
              disabled={safePage >= totalPages}
              onClick={() => setPage(safePage + 1)}
              className="px-3 py-1.5 text-xs rounded-md border border-border text-gray-400 hover:text-white hover:border-gray-600 disabled:opacity-30 disabled:cursor-default transition-all"
            >
              Next ›
            </button>
          </div>
        )}
      </Card>
    </div>
  );
}
