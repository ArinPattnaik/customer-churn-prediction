const API = process.env.NEXT_PUBLIC_API_URL || "";

export interface ModelInfo {
  model_type: string;
  training_date: string;
  dataset_row_count: number;
  evaluation_metrics: Record<string, number>;
  version: string;
}

export interface Customer {
  customer_id: string;
  churn_probability: number;
  risk_segment: "High" | "Medium" | "Low";
  drivers: string[];
  driver_values: number[];
  retention_strategy: string;
  annual_revenue: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ScoringResult {
  summary: {
    total_customers: number;
    high_risk: number;
    medium_risk: number;
    low_risk: number;
  };
  impact: {
    total_revenue_at_risk: number;
    targeted_customer_count: number;
    projected_revenue_saved: number;
    retention_roi: number;
  };
  customers: Customer[];
  feature_importance: FeatureImportance[];
  model_info: {
    model_type: string;
    training_date: string;
    dataset_row_count: number;
    auc_roc: number | null;
    accuracy: number | null;
    version: string;
  };
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API}/api/model-info`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Failed to load model info" }));
    throw new Error(err.error || "Failed to load model info");
  }
  return res.json();
}

export async function scoreCustomers(file?: File): Promise<ScoringResult> {
  const form = new FormData();
  if (file) form.append("file", file);
  const res = await fetch(`${API}/api/score`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Scoring failed" }));
    throw new Error(err.error || "Scoring failed");
  }
  return res.json();
}

export function waterfallUrl(index: number): string {
  return `${API}/api/waterfall/${index}`;
}

export function exportCsvUrl(): string {
  return `${API}/api/export/csv`;
}
