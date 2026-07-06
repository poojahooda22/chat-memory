import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes, useLocation } from "react-router";

import { AppShell } from "@/components/layout/AppShell";
import { Header } from "@/components/layout/Header";
import { Sidebar } from "@/components/layout/Sidebar";
import { ThemeProvider } from "@/components/theme-provider";
import { ConversationsProvider } from "@/lib/conversations";
import { queryClient } from "@/lib/query";
import { Home } from "@/pages/Home";
import { Memory } from "@/pages/Memory";
import { Sources } from "@/pages/Sources";

const TITLES: Record<string, string> = {
  "/": "Chat",
  "/memory": "Memory",
  "/sources": "Sources",
};

function Shell() {
  const { pathname } = useLocation();
  return (
    <AppShell sidebar={<Sidebar />} header={<Header title={TITLES[pathname] ?? "chat-memory"} />}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/memory" element={<Memory />} />
        <Route path="/sources" element={<Sources />} />
      </Routes>
    </AppShell>
  );
}

export function App() {
  return (
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <ConversationsProvider>
          <BrowserRouter>
            <Shell />
          </BrowserRouter>
        </ConversationsProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

export default App;