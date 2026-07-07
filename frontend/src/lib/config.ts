// Backend base URL. Set BUN_PUBLIC_BACKEND_URL in Vercel (prod); Bun inlines the literal
// value into the browser bundle at build time. Falls back to the local dev backend.
export const BACKEND_URL =
  process.env.BUN_PUBLIC_BACKEND_URL || "http://localhost:8005/api/v1";

// user identity now comes from the signed-in Supabase session (see lib/auth), never a constant