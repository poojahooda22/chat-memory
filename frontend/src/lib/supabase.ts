import { createClient } from "@supabase/supabase-js";

// The URL + anon (publishable) key are safe in the browser bundle — the anon key is designed to
// be public; access is protected by the backend verifying the signed JWT. Bun inlines
// BUN_PUBLIC_* at build time (set them in frontend/.env locally, and in Vercel for prod).
const url = process.env.BUN_PUBLIC_SUPABASE_URL;
const anon = process.env.BUN_PUBLIC_SUPABASE_ANON_KEY;

if (!url || !anon) {
  // fail loud in dev rather than a cryptic runtime error deep in a request later
  console.error("Missing BUN_PUBLIC_SUPABASE_URL / BUN_PUBLIC_SUPABASE_ANON_KEY");
}

export const supabase = createClient(url ?? "", anon ?? "");