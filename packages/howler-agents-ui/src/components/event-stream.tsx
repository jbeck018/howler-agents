import * as React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";

interface StreamEvent {
  type: string;
  data: Record<string, unknown>;
  timestamp: string;
}

interface EventStreamProps {
  events: StreamEvent[];
}

const eventColors: Record<string, "default" | "secondary" | "success" | "warning" | "destructive"> = {
  generation_started: "default",
  agent_evaluated: "secondary",
  selection_completed: "warning",
  reproduction_completed: "default",
  generation_completed: "success",
  run_completed: "success",
  run_failed: "destructive",
};

export function EventStream({ events }: EventStreamProps) {
  const containerRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [events]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Event Stream</CardTitle>
      </CardHeader>
      <CardContent>
        <div ref={containerRef} className="h-80 space-y-2 overflow-y-auto pr-2">
          {events.length === 0 && (
            <p className="text-sm text-muted-foreground">No events yet. Start an evolution run to see events.</p>
          )}
          {events.map((event, i) => (
            <div key={i} className="flex items-start gap-2 rounded-[var(--radius)] border p-2 text-xs">
              <Badge variant={eventColors[event.type] ?? "default"} className="shrink-0">
                {event.type}
              </Badge>
              <pre className="flex-1 overflow-x-auto text-muted-foreground">
                {JSON.stringify(event.data, null, 2)}
              </pre>
              <span className="shrink-0 text-muted-foreground">
                {new Date(event.timestamp).toLocaleTimeString()}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
