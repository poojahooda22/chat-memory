import { useCallback, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  Clock3,
  ImagePlus,
  Loader2,
  PawPrint,
  Pencil,
  RotateCcw,
  Tag,
  Trash2,
  User,
  XCircle,
} from "lucide-react";

import {
  deleteUpload,
  labelEntity,
  listUploads,
  renameUpload,
  retryUpload,
  unlabelEntity,
  uploadImages,
  uploadImageUrl,
  type EntityChip,
  type IngestJob,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const STATUS: Record<IngestJob["status"], { icon: typeof Clock3; label: string; cls: string }> = {
  queued: { icon: Clock3, label: "queued", cls: "text-muted-foreground" },
  processing: { icon: Loader2, label: "processing", cls: "text-muted-foreground animate-spin" },
  done: { icon: CheckCircle2, label: "remembered", cls: "text-emerald-500" },
  failed: { icon: XCircle, label: "failed", cls: "text-destructive" },
};

/** Detected people/pets on a photo: labeled ones show their name; unlabeled ones ask for it.
 * Naming an entity is how "a golden retriever" becomes Monty everywhere. */
function EntityChips({ job }: { job: IngestJob }) {
  const queryClient = useQueryClient();
  const [naming, setNaming] = useState<number | null>(null);
  const [value, setValue] = useState("");

  const label = useMutation({
    mutationFn: ({ index, name }: { index: number; name: string }) =>
      labelEntity(job.episode_id!, index, name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["uploads"] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
    },
  });
  const unlabel = useMutation({
    mutationFn: (index: number) => unlabelEntity(job.episode_id!, index),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["uploads"] }),
  });

  const nameable = job.entities.filter((e) => e.type === "person" || e.type === "pet");
  if (!job.episode_id || nameable.length === 0) return null;

  function commit(chip: EntityChip) {
    const name = value.trim();
    if (name) label.mutate({ index: chip.index, name });
    setNaming(null);
    setValue("");
  }

  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5">
      {nameable.map((chip) => {
        const Icon = chip.type === "pet" ? PawPrint : User;
        if (naming === chip.index) {
          return (
            <input
              key={chip.index}
              autoFocus
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") { e.preventDefault(); commit(chip); }
                else if (e.key === "Escape") { e.preventDefault(); setNaming(null); setValue(""); }
              }}
              onBlur={() => commit(chip)}
              placeholder={`name ${chip.description}…`}
              className="border-ring/60 bg-background w-44 rounded-full border px-2.5 py-0.5 text-xs focus:outline-none"
            />
          );
        }
        if (chip.label) {
          const auto = chip.labeled_by === "memory";
          return (
            <span
              key={chip.index}
              title={auto ? `Recognized by your memory: ${chip.description}` : chip.description}
              className="bg-secondary text-secondary-foreground inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium"
            >
              <Icon className="size-3" /> {chip.label}
              {auto && (
                <>
                  <span className="text-muted-foreground text-[10px] font-normal">auto</span>
                  <button
                    onClick={() => unlabel.mutate(chip.index)}
                    title={`Not ${chip.label}? Remove this label`}
                    className="text-muted-foreground hover:text-destructive"
                  >
                    <XCircle className="size-3" />
                  </button>
                </>
              )}
            </span>
          );
        }
        if (chip.suggested_name) {
          // recognition proposal: one click confirms; the pencil names it differently
          return (
            <span
              key={chip.index}
              className="border-ring/50 inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs"
            >
              <Icon className="size-3" />
              <span className="text-muted-foreground">{chip.description} —</span>
              <button
                onClick={() => label.mutate({ index: chip.index, name: chip.suggested_name! })}
                title={`Confirm this is ${chip.suggested_name}`}
                className="hover:text-foreground font-medium underline-offset-2 hover:underline"
              >
                {chip.suggested_name}? ✓
              </button>
              <button
                onClick={() => { setNaming(chip.index); setValue(""); }}
                title="No — name it differently"
                className="text-muted-foreground hover:text-foreground"
              >
                <Tag className="size-3" />
              </button>
            </span>
          );
        }
        return (
          <button
            key={chip.index}
            onClick={() => { setNaming(chip.index); setValue(""); }}
            title="Give this a name so your memory knows who it is"
            className="border-border text-muted-foreground hover:border-ring/60 hover:text-foreground inline-flex items-center gap-1 rounded-full border border-dashed px-2.5 py-0.5 text-xs transition-colors"
          >
            <Tag className="size-3" /> {chip.description} — name this
          </button>
        );
      })}
      {label.isError && (
        <span className="text-destructive text-xs">{(label.error as Error).message}</span>
      )}
    </div>
  );
}

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
  const rename = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => renameUpload(id, name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["uploads"] }),
  });
  const remove = useMutation({
    mutationFn: deleteUpload,
    onSuccess: () => {
      // forgetting a photo can also forget single-source memories — refresh both
      queryClient.invalidateQueries({ queryKey: ["uploads"] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
    },
  });
  const [renamingJob, setRenamingJob] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  function commitRename(id: string) {
    const name = renameValue.trim();
    if (name) rename.mutate({ id, name });
    setRenamingJob(null);
    setRenameValue("");
  }

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
                  <div className="group/name flex items-center gap-2 text-sm">
                    {renamingJob === job.id ? (
                      <input
                        autoFocus
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onFocus={(e) => e.currentTarget.select()}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") { e.preventDefault(); commitRename(job.id); }
                          else if (e.key === "Escape") { e.preventDefault(); setRenamingJob(null); }
                        }}
                        onBlur={() => commitRename(job.id)}
                        className="border-ring/60 bg-background w-56 rounded-md border px-2 py-0.5 text-sm focus:outline-none"
                      />
                    ) : (
                      <>
                        <span className="truncate font-medium">{job.filename || job.kind}</span>
                        <button
                          onClick={() => { setRenamingJob(job.id); setRenameValue(job.filename); }}
                          title="Rename"
                          className="text-muted-foreground hover:text-foreground opacity-0 transition-opacity group-hover/name:opacity-100"
                        >
                          <Pencil className="size-3" />
                        </button>
                      </>
                    )}
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
                  {job.status === "done" && <EntityChips job={job} />}
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
                  <button
                    onClick={() => remove.mutate(job.id)}
                    title="Forget this photo (removes its episode; memories it alone supported are forgotten)"
                    className="text-muted-foreground hover:bg-destructive/10 hover:text-destructive rounded-md p-1 transition-colors"
                  >
                    <Trash2 className="size-3.5" />
                  </button>
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