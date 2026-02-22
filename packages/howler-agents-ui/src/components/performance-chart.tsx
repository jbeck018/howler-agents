import * as React from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

interface DataPoint {
  generation: number;
  bestScore: number;
  meanScore: number;
}

interface PerformanceChartProps {
  data: DataPoint[];
  title?: string;
}

export function PerformanceChart({ data, title = "Performance Over Generations" }: PerformanceChartProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.285 0 0)" />
            <XAxis dataKey="generation" stroke="oklch(0.708 0 0)" fontSize={12} />
            <YAxis stroke="oklch(0.708 0 0)" fontSize={12} />
            <Tooltip
              contentStyle={{
                backgroundColor: "oklch(0.218 0 0)",
                border: "1px solid oklch(0.285 0 0)",
                borderRadius: "0.5rem",
                color: "oklch(0.984 0 0)",
                fontFamily: "var(--font-mono)",
              }}
            />
            <Line type="monotone" dataKey="bestScore" stroke="oklch(0.752 0.147 84)" name="Best" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="meanScore" stroke="oklch(0.708 0 0)" name="Mean" strokeWidth={1} strokeDasharray="5 5" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
