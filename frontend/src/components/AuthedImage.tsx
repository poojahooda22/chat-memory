import { useEffect, useState } from "react";

import { fetchImageBlobUrl } from "@/lib/api";
import { cn } from "@/lib/utils";

/** A protected upload image. The auth token can't ride on a plain <img src>, so we fetch the
 * bytes through the authenticated client and show them as an object URL (revoked on unmount). */
export function AuthedImage({
  jobId,
  alt,
  className,
}: {
  jobId: string;
  alt?: string;
  className?: string;
}) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;
    fetchImageBlobUrl(jobId)
      .then((u) => {
        if (cancelled) {
          URL.revokeObjectURL(u);
          return;
        }
        objectUrl = u;
        setUrl(u);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [jobId]);

  if (!url) return <div className={cn("bg-secondary animate-pulse", className)} />;
  return <img src={url} alt={alt} className={className} />;
}