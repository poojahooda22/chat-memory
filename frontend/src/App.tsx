import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes, useLocation } from "react-router";

import { AppShell } from "@/components/layout/AppShell";
import { Header } from "@/components/layout/Header";
import { Sidebar } from "@/components/layout/Sidebar";
import { ThemeProvider } from "@/components/theme-provider";
import { AuthProvider, useAuth } from "@/lib/auth";
import { ConversationsProvider } from "@/lib/conversations";
import { queryClient } from "@/lib/query";
import { Home } from "@/pages/Home";
import { Login } from "@/pages/Login";
import { Memory } from "@/pages/Memory";
import { Moments } from "@/pages/Moments";
import { Sources } from "@/pages/Sources";

const TITLES: Record<string, string> = {
  "/": "Chat",
  "/memory": "Memory",
  "/moments": "Moments",
  "/sources": "Sources",
};

function Shell() {
  const { pathname } = useLocation();
  return (
    <AppShell sidebar={<Sidebar />} header={<Header title={TITLES[pathname] ?? "chat-memory"} />}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/memory" element={<Memory />} />
        <Route path="/moments" element={<Moments />} />
        <Route path="/sources" element={<Sources />} />
      </Routes>
    </AppShell>
  );
}

/** The gate: show the login page until there's a session; the app only mounts when signed in. */
function Guarded() {
  const { session, loading } = useAuth();
  if (loading) {
    return (
      <div className="bg-background text-muted-foreground flex min-h-screen items-center justify-center text-sm">
        …
      </div>
    );
  }
  if (!session) return <Login />;
  return (
    <ConversationsProvider>
      <BrowserRouter>
        <Shell />
      </BrowserRouter>
    </ConversationsProvider>
  );
}

export function App() {
  return (
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <Guarded />
        </AuthProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

export default App;