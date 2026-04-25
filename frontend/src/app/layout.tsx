import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ChurnGuard — Customer Churn Prediction",
  description:
    "AI-powered churn prediction dashboard with SHAP explanations and retention strategies.",
  icons: {
    icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🛡️</text></svg>",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-bg text-gray-100 antialiased">{children}</body>
    </html>
  );
}
