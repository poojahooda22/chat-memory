import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { CalendarDays, Orbit, PawPrint, User, X } from "lucide-react";

import { getGraph, listUploads, uploadImageUrl, type IngestJob } from "@/lib/api";
import { cn } from "@/lib/utils";

/** Preset bubble slots (percent + size) so a handful of photos scatter pleasantly. */
const SLOTS: { x: number; y: number; size: number }[] = [
  { x: 16, y: 26, size: 132 },
  { x: 84, y: 22, size: 116 },
  { x: 12, y: 70, size: 108 },
  { x: 88, y: 72, size: 120 },
  { x: 26, y: 90, size: 92 },
  { x: 74, y: 92, size: 96 },
  { x: 6, y: 46, size: 88 },
  { x: 94, y: 48, size: 88 },
  { x: 40, y: 8, size: 84 },
  { x: 60, y: 8, size: 84 },
];

/** Lay the entity nodes on a ring around the centre (an ellipse to fit the wide canvas). */
function ringPosition(i: number, n: number): { x: number; y: number } {
  if (n <= 1) return { x: 50, y: 48 };
  const angle = (-90 + (360 * i) / n) * (Math.PI / 180);
  return { x: 50 + 26 * Math.cos(angle), y: 48 + 30 * Math.sin(angle) };
}

const NODE_ICON = { pet: PawPrint, person: User, object: Orbit } as const;

/** Your memories as a canvas: photos float as bubbles, and the people & pets you've named
 * connect by lines whose thickness is how often they appear together. */
export function Moments() {
  const [openId, setOpenId] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const uploads = useQuery({ queryKey: ["uploads"], queryFn: listUploads });
  const graph = useQuery({ queryKey: ["graph"], queryFn: getGraph });

  const photos = useMemo(
    () =>
      (uploads.data ?? [])
        .filter((j) => j.status === "done" && j.episode_id)
        .sort((a, b) =>
          (a.captured_at ?? a.created_at).localeCompare(b.captured_at ?? b.created_at),
        ),
    [uploads.data],
  );

  const nodes = graph.data?.nodes ?? [];
  const edges = graph.data?.edges ?? [];
  const nodePos = useMemo(() => {
    const map = new Map<string, { x: number; y: number }>();
    nodes.forEach((node, i) => map.set(node.id, ringPosition(i, nodes.length)));
    return map;
  }, [nodes]);
  const nodeById = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);

  const open = photos.find((j) => j.id === openId) ?? null;
  const selected = selectedNodeId ? (nodeById.get(selectedNodeId) ?? null) : null;
  const selectedNeighbours = useMemo(() => {
    if (!selectedNodeId) return [];
    return edges
      .filter((e) => e.src === selectedNodeId || e.dst === selectedNodeId)
      .map((e) => ({
        other: e.src === selectedNodeId ? e.dst : e.src,
        weight: e.weight,
        count: e.cooccur_count,
        learning: e.is_learning,
      }))
      .sort((a, b) => b.weight - a.weight);
  }, [edges, selectedNodeId]);

  if (photos.length === 0 && nodes.length === 0) {
    return (
      <div className="mx-auto flex max-w-md flex-1 flex-col items-center justify-center gap-3 px-4 text-center">
        <Orbit className="text-muted-foreground size-8" />
        <h2 className="text-lg font-semibold">Moments</h2>
        <p className="text-muted-foreground text-sm">
          Feed photos in Sources and name who's in them — they'll float here and connect.
        </p>
      </div>
    );
  }

  return (
    <div className="relative min-h-0 w-full flex-1 overflow-hidden">
      {/* hint */}
      {nodes.length > 0 && (
        <div className="text-muted-foreground pointer-events-none absolute top-3 left-1/2 z-20 -translate-x-1/2 text-center text-[11px]">
          people &amp; pets you've named — thicker lines appear together more often
        </div>
      )}

      {/* photo bubbles (dimmed a touch when the graph is present so the connections read) */}
      {photos.map((job, i) => {
        const slot = SLOTS[i % SLOTS.length]!;
        const wrap = Math.floor(i / SLOTS.length) * 4;
        return (
          <motion.button
            key={job.id}
            onClick={() => setOpenId(job.id)}
            className={cn(
              "border-border/60 absolute overflow-hidden rounded-full border shadow-lg",
              nodes.length > 0 && "opacity-70",
            )}
            style={{
              left: `${slot.x + wrap}%`,
              top: `${slot.y + wrap}%`,
              width: slot.size,
              height: slot.size,
              transform: "translate(-50%, -50%)",
            }}
            initial={{ opacity: 0, scale: 0.6 }}
            animate={{ opacity: nodes.length > 0 ? 0.7 : 1, scale: 1, y: [0, -8, 0] }}
            transition={{
              opacity: { duration: 0.5, delay: i * 0.08 },
              scale: { type: "spring", stiffness: 200, damping: 18, delay: i * 0.08 },
              y: { duration: 4 + (i % 3), repeat: Infinity, ease: "easeInOut", delay: i * 0.4 },
            }}
            whileHover={{ scale: 1.06, opacity: 1 }}
            title={job.caption ?? job.filename}
          >
            <img
              src={uploadImageUrl(job.id)}
              alt={job.filename}
              className="size-full object-cover"
              loading="lazy"
            />
          </motion.button>
        );
      })}

      {/* edges — drawn in a 0..100 space matching the node percent positions */}
      <svg
        className="pointer-events-none absolute inset-0 z-[5] size-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >
        {edges.map((e, i) => {
          const a = nodePos.get(e.src);
          const b = nodePos.get(e.dst);
          if (!a || !b) return null;
          return (
            <line
              key={i}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              className={e.is_learning ? "stroke-muted-foreground/30" : "stroke-foreground/50"}
              strokeWidth={1 + e.weight * 6}
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </svg>

      {/* entity nodes */}
      {nodes.map((node) => {
        const p = nodePos.get(node.id)!;
        const size = Math.min(120, 54 + node.photo_count * 10);
        const Icon = NODE_ICON[node.type] ?? Orbit;
        return (
          <motion.button
            key={node.id}
            onClick={() => setSelectedNodeId(node.id)}
            className="border-background bg-secondary absolute z-10 rounded-full border-2 shadow-xl"
            style={{
              left: `${p.x}%`,
              top: `${p.y}%`,
              width: size,
              height: size,
              transform: "translate(-50%, -50%)",
            }}
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ type: "spring", stiffness: 240, damping: 20 }}
            whileHover={{ scale: 1.08 }}
          >
            {node.representative_job_id ? (
              <img
                src={uploadImageUrl(node.representative_job_id)}
                alt={node.name}
                className="size-full rounded-full object-cover"
              />
            ) : (
              <div className="text-muted-foreground flex size-full items-center justify-center">
                <Icon className="size-6" />
              </div>
            )}
            <span className="bg-background/90 text-foreground absolute -bottom-2 left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium whitespace-nowrap shadow">
              <Icon className="size-3" /> {node.name}
            </span>
          </motion.button>
        );
      })}

      {/* selected entity → its connections */}
      <AnimatePresence>
        {selected && (
          <motion.div
            className="bg-card absolute top-4 left-4 z-20 w-64 rounded-xl border p-3 shadow-xl"
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -12 }}
          >
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-sm font-semibold">
                {(() => {
                  const Icon = NODE_ICON[selected.type] ?? Orbit;
                  return <Icon className="size-4" />;
                })()}
                {selected.name}
              </span>
              <button onClick={() => setSelectedNodeId(null)} aria-label="Close">
                <X className="text-muted-foreground hover:text-foreground size-4" />
              </button>
            </div>
            <p className="text-muted-foreground mt-0.5 text-xs">in {selected.photo_count} photos</p>
            <div className="mt-2 space-y-1.5">
              {selectedNeighbours.length === 0 ? (
                <p className="text-muted-foreground text-xs">
                  No connections yet — appears alone so far.
                </p>
              ) : (
                selectedNeighbours.map((nb) => (
                  <div key={nb.other} className="flex items-center gap-2 text-xs">
                    <span className="w-16 shrink-0 truncate">
                      {nodeById.get(nb.other)?.name ?? "?"}
                    </span>
                    <div className="bg-secondary h-1.5 flex-1 overflow-hidden rounded-full">
                      <div
                        className={cn("h-full rounded-full", nb.learning ? "bg-muted-foreground/40" : "bg-foreground")}
                        style={{ width: `${Math.round(nb.weight * 100)}%` }}
                      />
                    </div>
                    <span className="text-muted-foreground shrink-0">{nb.count}📷</span>
                  </div>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* step inside a photo bubble */}
      <AnimatePresence>
        {open && (
          <motion.div
            className="bg-background/80 absolute inset-0 z-30 flex items-center justify-center p-6 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setOpenId(null)}
          >
            <motion.div
              className="bg-card w-full max-w-lg overflow-hidden rounded-2xl border shadow-xl"
              initial={{ scale: 0.85, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ type: "spring", stiffness: 260, damping: 24 }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="relative">
                <img
                  src={uploadImageUrl(open.id)}
                  alt={open.filename}
                  className="max-h-80 w-full object-cover"
                />
                <button
                  onClick={() => setOpenId(null)}
                  aria-label="Close"
                  className="bg-background/70 text-foreground absolute top-2 right-2 rounded-full p-1.5 backdrop-blur"
                >
                  <X className="size-4" />
                </button>
              </div>
              <div className="space-y-2 p-4">
                <div className="text-muted-foreground flex items-center gap-1.5 text-xs">
                  <CalendarDays className="size-3.5" />
                  {open.captured_at && open.time_source === "exif"
                    ? `captured ${new Date(open.captured_at).toLocaleDateString()}`
                    : `uploaded ${new Date(open.created_at).toLocaleDateString()}`}
                </div>
                {open.caption && <p className="text-sm">{open.caption}</p>}
                <MomentEntities job={open} />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function MomentEntities({ job }: { job: IngestJob }) {
  const named = job.entities.filter((e) => e.label);
  if (named.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5 pt-1">
      {named.map((e) => (
        <span
          key={e.index}
          className="bg-secondary text-secondary-foreground inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium"
        >
          {e.type === "pet" ? <PawPrint className="size-3" /> : <User className="size-3" />}
          {e.label}
        </span>
      ))}
    </div>
  );
}