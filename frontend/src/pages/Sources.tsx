import { useCallback, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Clock3, ImagePlus, Loader2, RotateCcw, XCircle } from "lucide-react";

import { listUploads, retryUpload, uploadImages, uploadImageUrl, type IngestJob } from "@/lib/api";
import { cn } from "@/lib/utils";

const STATUS: Record<IngestJob["status"], { icon: typeof Clock3; label: string; cls: string }> = {
  queued: { icon: Clock3, label: "queued", cls: "text-muted-foreground" },
  processing: { icon: Loader2, label: "processing", cls: "text-muted-foreground animate-spin" },
  done: { icon: CheckCircle2, label: "remembered", cls: "text-emerald-500" },
  failed: { icon: XCircle, label: "failed", cls: "text-destructive" },
};

/** The mouth of the memory: feed it your photos and screenshots. Each upload is captioned,
 * timestamped from its EXIF when possible, and distilled into memories. */
export function Sources() {
  const queryClient = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const uploads = useQuery({
    queryKey: ["uploads"],
    queryFn: listUploads,
    // poll while anything is still working its way into memory
    refetchInterval: (query) =>
      query.state.data?.some((j) => j.status === "queued" || j.status === "processing")
        ? 2500
        : false,
  });

  const upload = useMutation({
    mutationFn: uploadImages,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["uploads"] }),
  });
  const retry = useMutation({
    mutationFn: retryUpload,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["uploads"] }),
  });

  const addFiles = useCallback(
    (list: FileList | null) => {
      if (!list?.length) return;
      const images = Array.from(list).filter((f) => f.type.startsWith("image/"));
      if (images.length) upload.mutate(images);
    },
    [upload],
  );

  return (
    <div className="min-h-0 w-full flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl px-4 py-8">
        <h2 className="text-lg font-semibold">Sources</h2>
        <p className="text-muted-foreground mt-1 text-sm">
          Feed your memory. Photos keep their capture time (and location, when the file has it);
          screenshots are remembered by the text on them. Everything lands as an episode with
          receipts.
        </p>

        {/* drop zone — hands the RAW files to the uploader (no canvas, EXIF survives) */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            addFiles(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
          className={cn(
            "mt-6 flex cursor-pointer flex-col items-center gap-2 rounded-2xl border border-dashed px-6 py-10 text-center transition-colors",
            dragging ? "border-ring bg-card" : "border-border hover:bg-card/60",
          )}
        >
          <ImagePlus className="text-muted-foreground size-7" />
          <div className="text-sm font-medium">
            {upload.isPending ? "Uploading…" : "Drop photos or screenshots here"}
          </div>
          <div className="text-muted-foreground text-xs">or click to choose files</div>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={(e) => {
              addFiles(e.target.files);
              e.target.value = ""; // same file can be re-picked
            }}
          />
        </div>

        {upload.isError && (
          <div className="text-destructive mt-3 text-xs">
            Upload failed: {(upload.error as Error).message}
          </div>
        )}

        {/* ingest status list */}
        <div className="mt-8 space-y-2">
          {(uploads.data ?? []).map((job) => {
            const s = STATUS[job.status];
            const Icon = s.icon;
            return (
              <div key={job.id} className="bg-card flex items-start gap-3 rounded-xl border p-3">
                <img
                  src={uploadImageUrl(job.id)}
                  alt={job.filename}
                  className="size-14 shrink-0 rounded-lg object-cover"
                  loading="lazy"
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 text-sm">
                    <span className="truncate font-medium">{job.filename || job.kind}</span>
                    <span className="text-muted-foreground shrink-0 text-[11px] uppercase">
                      {job.kind}
                    </span>
                  </div>
                  <div className="text-muted-foreground mt-0.5 text-xs">
                    {job.captured_at && job.time_source === "exif"
                      ? `captured ${new Date(job.captured_at).toLocaleString()}`
                      : `uploaded ${new Date(job.created_at).toLocaleString()}`}
                  </div>
                  {job.status === "done" && job.caption && (
                    <p className="mt-1.5 line-clamp-2 text-xs">{job.caption}</p>
                  )}
                  {job.status === "failed" && (
                    <p className="text-destructive mt-1.5 line-clamp-2 text-xs">{job.error}</p>
                  )}
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <span className={cn("flex items-center gap-1 text-xs", s.cls)}>
                    <Icon className="size-3.5" /> {s.label}
                  </span>
                  {(job.status === "failed" || job.status === "queued") && (
                    <button
                      onClick={() => retry.mutate(job.id)}
                      title="Retry"
                      className="text-muted-foreground hover:bg-secondary hover:text-foreground rounded-md p-1 transition-colors"
                    >
                      <RotateCcw className="size-3.5" />
                    </button>
                  )}
                </div>
              </div>
            );
          })}
          {uploads.data?.length === 0 && (
            <p className="text-muted-foreground text-xs">Nothing fed yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}