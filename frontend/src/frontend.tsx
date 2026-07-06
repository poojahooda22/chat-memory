/** React entry point — mounts <App/> into #root. Included from src/index.html. */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import "./index.css";

const elem = document.getElementById("root")!;
const app = (
  <StrictMode>
    <App />
  </StrictMode>
);

// Bun HMR: reuse the same root across hot reloads
(import.meta.hot.data.root ??= createRoot(elem)).render(app);