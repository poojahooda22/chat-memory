import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { CalendarDays, Orbit, PawPrint, User, X } from "lucide-react";

import { listUploads, uploadImageUrl, type IngestJob } from "@/lib/api";

/** Preset bubble slots (percent coordinates + size), hand-tuned so a handful of photos
 * scatter pleasantly. More photos than slots wrap around with a small drift. */
const SLOTS: { x: number; y: number; size: number }[] = [
  { x: 24, y: 28, size: 168 },
  { x: 58, y: 18, size: 132 },
  { x: 76, y: 46, size: 152 },
  { x: 38, y: 62, size: 140 },
  { x: 12, y: 58, size: 116 },
  { x: 62, y: 74, size: 124 },
  { x: 86, y: 16, size: 100 },
  { x: 44, y: 12, size: 96 },
  { x: 88, y: 74, size: 108 },
  { x: 20, y: 84, size: 100 },
  { x: 70, y: 34, size: 92 },
  { x: 8, y: 22, size: 92 },
];

/** Your photos as floating memory bubbles — click one to step inside it. */
export function Moments() {
  const [openId, setOpenId] = useState<string | null>(null);
  const uploads = useQuery({ queryKey: ["uploads"], queryFn: listUploads });

  const photos = useMemo(
    () =>
      (uploads.data ?? [])
        .filter((j) => j.status === "done" && j.episode_id)
        .sort((a, b) => (a.captured_at ?? a.created_at).localeCompare(b.captured_at ?? b.created_at)),
    [uploads.data],
  );
  const open = photos.find((j) => j.id === openId) ?? null;

  if (photos.length === 0) {
    return (
      <div className="mx-auto flex max-w-md flex-1 flex-col items-center justify-center gap-3 px-4 text-center">
        <Orbit className="text-muted-foreground size-8" />
        <h2 className="text-lg font-semibold">Moments</h2>
        <p className="text-muted-foreground text-sm">
          Feed photos in Sources and they'll float here as memory bubbles.
        </p>
      </div>
    );
  }

  return (
    <div className="relative min-h-0 w-full flex-1 overflow-hidden">
      {photos.map((job, i) => {
        const slot = SLOTS[i % SLOTS.length];
        const wrap = Math.floor(i / SLOTS.length) * 4; // later laps drift slightly
        return (
          <motion.button
            key={job.id}
            onClick={() => setOpenId(job.id)}
            className="border-border/60 absolute overflow-hidden rounded-full border shadow-lg"
            style={{
              left: `${slot.x + wrap}%`,
              top: `${slot.y + wrap}%`,
              width: slot.size,
              height: slot.size,
              transform: "translate(-50%, -50%)",
            }}
            initial={{ opacity: 0, scale: 0.6 }}
            animate={{ opacity: 1, scale: 1, y: [0, -8, 0] }}
            transition={{
              opacity: { duration: 0.5, delay: i * 0.08 },
              scale: { type: "spring", stiffness: 200, damping: 18, delay: i * 0.08 },
              // each bubble floats on its own rhythm
              y: { duration: 4 + (i % 3), repeat: Infinity, ease: "easeInOut", delay: i * 0.4 },
            }}
            whileHover={{ scale: 1.07 }}
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

      {/* step inside a bubble */}
      <AnimatePresence>
        {open && (
          <motion.div
            className="bg-background/80 absolute inset-0 z-10 flex items-center justify-center p-6 backdrop-blur-sm"
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