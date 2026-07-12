import { useState } from "react";
import { BrainCircuit } from "lucide-react";

import { useAuth } from "@/lib/auth";

/** GitHub mark as inline SVG — lucide has been dropping brand icons, so don't depend on one. */
function GitHubIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden="true">
      <path d="M12 .5C5.7.5.5 5.7.5 12c0 5.1 3.3 9.4 7.9 10.9.6.1.8-.2.8-.5v-1.8c-3.2.7-3.9-1.5-3.9-1.5-.5-1.3-1.3-1.7-1.3-1.7-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.7 1.3 3.4 1 .1-.7.4-1.3.7-1.6-2.5-.3-5.2-1.3-5.2-5.7 0-1.3.4-2.3 1.2-3.1-.1-.3-.5-1.5.1-3.1 0 0 1-.3 3.3 1.2a11.4 11.4 0 0 1 6 0C17 4.3 18 4.6 18 4.6c.6 1.6.2 2.8.1 3.1.8.8 1.2 1.8 1.2 3.1 0 4.4-2.7 5.4-5.3 5.7.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.5A11.5 11.5 0 0 0 23.5 12C23.5 5.7 18.3.5 12 .5z" />
    </svg>
  );
}

/** The front door — sign in or create an account. Everything behind it is per-user private. */
export function Login() {
  const { signIn, signUp, signInWithGitHub } = useAuth();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setNotice(null);
    setBusy(true);
    if (mode === "login") {
      const { error } = await signIn(email, password);
      if (error) setError(error);
    } else {
      const { error, needsConfirm } = await signUp(email, password);
      if (error) setError(error);
      else if (needsConfirm) setNotice("Check your email to confirm, then sign in.");
    }
    setBusy(false);
  }

  const inputCls =
    "border-input bg-background focus:border-ring/60 w-full rounded-lg border px-3 py-2 text-sm focus:outline-none";

  return (
    <div className="bg-background text-foreground flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex flex-col items-center gap-2 text-center">
          <BrainCircuit className="size-8" />
          <h1 className="text-xl font-semibold tracking-tight">chat-memory</h1>
          <p className="text-muted-foreground text-sm">
            an assistant that remembers you — your own private memory
          </p>
        </div>

        <form onSubmit={submit} className="bg-card space-y-3 rounded-2xl border p-5">
          <input
            type="email"
            required
            autoComplete="email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className={inputCls}
          />
          <input
            type="password"
            required
            minLength={6}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            placeholder="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className={inputCls}
          />
          {error && <p className="text-destructive text-xs">{error}</p>}
          {notice && <p className="text-xs text-emerald-500">{notice}</p>}
          <button
            type="submit"
            disabled={busy}
            className="bg-primary text-primary-foreground w-full rounded-lg py-2 text-sm font-medium transition-opacity hover:opacity-90 disabled:opacity-60"
          >
            {busy ? "…" : mode === "login" ? "Sign in" : "Create account"}
          </button>
          <button
            type="button"
            onClick={() => signInWithGitHub()}
            className="border-input hover:bg-secondary flex w-full items-center justify-center gap-2 rounded-lg border py-2 text-sm font-medium transition-colors"
          >
            <GitHubIcon className="size-4" /> Continue with GitHub
          </button>
        </form>

        <button
          onClick={() => {
            setMode((m) => (m === "login" ? "signup" : "login"));
            setError(null);
            setNotice(null);
          }}
          className="text-muted-foreground hover:text-foreground mt-4 w-full text-center text-xs"
        >
          {mode === "login" ? "New here? Create an account" : "Have an account? Sign in"}
        </button>
      </div>
    </div>
  );
}