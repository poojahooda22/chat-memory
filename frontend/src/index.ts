/** Dev/prod static server for the SPA. Serves index.html for all routes.
 * The app calls the FastAPI backend directly (see src/lib/config.ts) — this server
 * only hosts the frontend bundle. Port 3100 avoids other local projects. */
import { serve } from "bun";

import index from "./index.html";

const server = serve({
  port: Number(process.env.PORT) || 3100,
  routes: {
    "/*": index,
  },
  development: process.env.NODE_ENV !== "production" && {
    hmr: true,
    console: true,
  },
});

console.log(`chat-memory frontend running at ${server.url}`);