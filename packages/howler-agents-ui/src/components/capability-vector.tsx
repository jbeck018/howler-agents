import * as React from "react";
import { cn } from "@/lib/utils";

interface CapabilityVectorProps {
  vector: number[];
  labels?: string[];
  className?: string;
}

export function CapabilityVector({ vector, labels, className }: CapabilityVectorProps) {
  return (
    <div className={cn("flex flex-wrap gap-1", className)}>
      {vector.map((val, i) => (
        <div
          key={i}
          className={cn(
            "h-6 w-6 rounded-sm border text-[10px] flex items-center justify-center",
            val > 0.5
              ? "bg-primary/20 border-primary text-primary"
              : "bg-muted/20 border-border text-muted-foreground"
          )}
          title={labels?.[i] ?? `Probe ${i + 1}`}
        >
          {val > 0.5 ? "1" : "0"}
        </div>
      ))}
    </div>
  );
}
