import { QueryClientProvider } from "@tanstack/react-query";

import { Dashboard } from "@/components/Dashboard";
import { queryClient } from "@/lib/query";

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Dashboard />
    </QueryClientProvider>
  );
}

export default App;