import * as React from "react";
import { Moon, Sun } from "lucide-react";
import { Button } from "./ui/button";

export function AppHeader() {
  const [dark, setDark] = React.useState(true);

  const toggleTheme = () => {
    setDark(!dark);
    document.documentElement.classList.toggle("light", !dark);
  };

  return (
    <header className="flex h-14 items-center justify-between border-b bg-card px-6">
      <h1 className="text-sm font-medium text-muted-foreground">
        Group-Evolving Agents Dashboard
      </h1>
      <Button variant="ghost" size="icon" onClick={toggleTheme}>
        {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
      </Button>
    </header>
  );
}
