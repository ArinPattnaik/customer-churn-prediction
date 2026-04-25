"use client";

import { useState, useEffect, useCallback } from "react";
import {
  ScoringResult,
  ModelInfo,
  fetchModelInfo,
  scoreCustomers,
} from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import TopBar from "@/components/TopBar";
import Welcome from "@/components/Welcome";
import Dashboard from "@/components/Dashboard";
import Customers from "@/components/Customers";
import Explainability from "@/components/Explainability";
import LoadingOverlay from "@/components/LoadingOverlay";

export type Tab = "dashboard" | "customers" | "explainability";

export default function Home() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [data, setData] = useState<ScoringResult | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    fetchModelInfo()
      .then(setModelInfo)
      .catch(() => {});
  }, []);

  const handleScore = useCallback(async (file?: File) => {
    setLoading(true);
    setError(null);
    try {
      const result = await scoreCustomers(file);
      setData(result);
      setTab("dashboard");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Scoring failed");
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        tab={tab}
        setTab={setTab}
        modelInfo={modelInfo}
        data={data}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="flex-1 flex flex-col min-h-screen overflow-hidden ml-0 lg:ml-60">
        <TopBar
          tab={tab}
          onUpload={(f) => handleScore(f)}
          onDemo={() => handleScore()}
          onMenuToggle={() => setSidebarOpen(true)}
        />

        <div className="flex-1 overflow-y-auto">
          {loading && <LoadingOverlay />}

          {error && (
            <div className="m-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm">
              {error}
            </div>
          )}

          {!data && !loading && <Welcome onDemo={() => handleScore()} onUpload={(f) => handleScore(f)} />}

          {data && tab === "dashboard" && <Dashboard data={data} />}
          {data && tab === "customers" && <Customers data={data} />}
          {data && tab === "explainability" && <Explainability data={data} />}
        </div>
      </main>
    </div>
  );
}
