"use client";

import { cn } from "@/lib/utils";

interface Props {
  title?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
}

export default function Card({ title, action, children, className, bodyClassName }: Props) {
  return (
    <div className={cn(
      "bg-bg-card border border-border rounded-xl overflow-hidden animate-fade-up",
      "hover:border-gray-700/60 transition-all",
      className
    )}>
      {title && (
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-border">
          <h3 className="text-sm font-semibold">{title}</h3>
          {action}
        </div>
      )}
      <div className={cn("p-5", bodyClassName)}>{children}</div>
    </div>
  );
}
