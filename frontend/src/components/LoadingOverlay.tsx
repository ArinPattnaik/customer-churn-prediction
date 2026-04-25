"use client";

export default function LoadingOverlay() {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="text-center">
        <div className="w-12 h-12 border-[3px] border-border border-t-accent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-sm text-gray-400">Scoring customers &amp; computing explanations...</p>
      </div>
    </div>
  );
}
