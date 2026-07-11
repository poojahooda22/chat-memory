# Deploy — chat-memory (free tier)

Stack: **Supabase Postgres** (data + auth) · **Render** free Docker web service (backend) ·
**Vercel** static (frontend). Cost: $0 (Render free spins down after ~15 min idle and cold-starts
on the next request — the first hit after idle takes ~30–60s; the UI is static so it stays instant).

Config lives in the repo: `render.yaml` (backend blueprint), `frontend/vercel.json` (static build),
`backend/Dockerfile` (runs `alembic upgrade head` then uvicorn on `$PORT`).

## Order matters (there are two cross-dependencies)
The frontend inlines the backend URL **at build time**, and the backend's CORS must name the
frontend URL. So: DB → backend → frontend → tighten CORS.

### 1. Database — Supabase (your existing project)
- Dashboard → **Database → Extensions** → enable `vector` (the migration also runs `CREATE
  EXTENSION vector`, but enabling it first avoids a permissions surprise).
- **Project Settings → Database → Connection string → "Session pooler"** (NOT "Transaction
  pooler"). Copy it and reshape to the app's driver:
  ```
  postgresql+psycopg://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
  ```
  Why session pooler (5432), not transaction pooler (6543): psycopg uses prepared statements and
  the recall path sets a per-transaction GUC (`SET LOCAL hnsw.iterative_scan`) — transaction-mode
  pgbouncer breaks both.

### 2. Backend — Render
- **New + → Blueprint →** pick `poojahooda22/chat-memory`. Render reads `render.yaml`.
- Set the four env vars (all `sync:false`, you paste them):
  - `DATABASE_URL` = the session-pooler string from step 1
  - `AI_GATEWAY_API_KEY` = your Vercel AI Gateway key
  - `SUPABASE_URL` = `https://<ref>.supabase.co`
  - `CORS_ORIGINS` = `["*"]` for now (tighten in step 4)
- Deploy. On boot it migrates to head (incl. the `0010` FTS index) then serves. Confirm
  `https://<service>.onrender.com/health` returns `{"status":"ok"}`. Note the service URL.

### 3. Frontend — Vercel
- **New Project → import the repo → Root Directory = `frontend`.** Vercel reads `frontend/vercel.json`.
- Env vars (inlined at build — a later change needs a redeploy):
  - `BUN_PUBLIC_BACKEND_URL` = `https://<service>.onrender.com/api/v1`
  - `BUN_PUBLIC_SUPABASE_URL` = `https://<ref>.supabase.co`
  - `BUN_PUBLIC_SUPABASE_ANON_KEY` = your Supabase anon key
- Deploy. Note the URL, e.g. `https://chat-memory.vercel.app`.

### 4. Lock CORS + auth redirect
- Render → the service → Environment → `CORS_ORIGINS` = `["https://chat-memory.vercel.app"]` → redeploy.
- Supabase → **Authentication → URL Configuration** → add the Vercel URL to Site URL + the redirect
  allowlist, so sign-in redirects resolve.

### 5. Verify
Open the Vercel URL, sign in, ask "did we talk about TanStack?" — it should recall.

## Honest caveats
- **Prod starts with an EMPTY memory DB.** Your local data (the Monty photos, the TanStack turns)
  lives in the local dockerized Postgres on `:5434`, not Supabase. A fresh sign-up on prod has a
  blank memory. Migrating the local data across is a separate optional step (pg_dump the local
  `public` tables → restore into Supabase) — ask if you want it.
- **Free Render cold start:** the first request after idle is slow (~30–60s). Fine for a portfolio;
  upgrade to a paid instance ($7/mo) for always-warm.
- **Multi-tenant hardening deferred:** a couple of per-user query filters and `CORS *` were flagged
  in review; fine for a demo you drive, worth closing before opening sign-ups widely.
- **The leaked GitHub OAuth secret** remains in git history (deferred by choice); rotate it if the
  repo's exposure grows.
