import { Database } from "lucide-react";

/** Placeholder — Phase 7 will connect the user's own data (Google Drive / exports) here,
 * the sources that feed episodic memory. */
export function Sources() {
  return (
    <div className="mx-auto flex max-w-md flex-1 flex-col items-center justify-center gap-3 px-4 text-center">
      <Database className="text-muted-foreground size-8" />
      <h2 className="text-lg font-semibold">Sources</h2>
      <p className="text-muted-foreground text-sm">
        Connect the data that feeds your memory — Google Drive, photo exports, and more. Coming in
        a later phase; for now, memory is built from your chat.
      </p>
    </div>
  );
}