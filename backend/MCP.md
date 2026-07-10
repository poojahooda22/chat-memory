# chat-memory as an MCP server

`mcp_server.py` exposes the memory as a local **stdio MCP server** — so any MCP host (Claude
Desktop, Cursor, or your own agent) can read and write your memory as tools. The REST API stays
the source of truth; this is a second surface over the same engine.

It's the **single-user local wrapper**: it acts for one user (`MCP_USER_ID`). Multi-tenant remote
(per-user OAuth token propagation over Streamable HTTP) is the Tier-2 upgrade.

## Tools

| Tool | Kind | What it does |
|---|---|---|
| `recall(query)` | read-only | Returns the relevant memory as a **bundle**: facts (each with `source` + a 0-1 `confidence` + provenance episode ids), photo memories, and one overall confidence. |
| `remember(text)` | write | Distils a durable fact from a sentence and deduplicates it against what's known. |
| `graph_neighbors(entity)` | read-only | Given a named person/pet/thing, returns who it co-occurs with in photos + edge weight. |

Tool results are **structured** (typed), and annotations are honest (`readOnlyHint` etc.) so a
host can gate confirmation UX correctly.

## Run it

Prereqs: the DB is up (`docker compose up -d` → `chat-memory-db` on 5434) and `backend/.env` has
`AI_GATEWAY_API_KEY` + `DATABASE_URL`. Find your user id (the `sub` of your Supabase session).

```bash
# from backend/
MCP_USER_ID='<your-user-id>' uv run python mcp_server.py     # stdio; logs to stderr
```

### Claude Desktop / Cursor config

```json
{
  "mcpServers": {
    "chat-memory": {
      "command": "uv",
      "args": ["run", "python", "mcp_server.py"],
      "cwd": "<absolute path to>/chat-memory/backend",
      "env": { "MCP_USER_ID": "<your-user-id>" }
    }
  }
}
```

## Notes

- **stdio discipline:** stdout is the JSON-RPC stream — the server logs only to stderr.
- **Verify:** `npx @modelcontextprotocol/inspector uv run python mcp_server.py` opens the Inspector
  to browse + call the tools.
- **Not built yet:** remote Streamable-HTTP transport + OAuth 2.1 (audience-validated, no token
  passthrough) for multi-user hosting; an evaluation harness of ≥10 tasks.