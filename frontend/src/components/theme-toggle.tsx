import { useCallback, useRef } from "react";
import { flushSync } from "react-dom";
import { Moon, Sun } from "lucide-react";

import { useTheme } from "@/components/theme-provider";
import { cn } from "@/lib/utils";

// Circular clip-path reveal expanding from the button center via the View Transitions API,
// ported from the Perplexity app. Falls back to an instant switch where the API is
// unavailable or reduced motion is requested.
export function ThemeToggle({ className, duration = 450 }: { className?: string; duration?: number }) {
  const { theme, setTheme } = useTheme();
  const isDark = theme === "dark";
  const ref = useRef<HTMLButtonElement>(null);
  const label = isDark ? "Switch to light theme" : "Switch to dark theme";

  const toggle = useCallback(() => {
    const next = isDark ? "light" : "dark";
    // Apply synchronously so the View Transition snapshots the new theme.
    const apply = () =>
      flushSync(() => {
        document.documentElement.classList.toggle("dark", next === "dark");
        setTheme(next);
      });

    const doc = document as Document & {
      startViewTransition?: (cb: () => void) => { ready: Promise<void> };
    };
    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (prefersReduced || typeof doc.startViewTransition !== "function" || !ref.current) {
      apply();
      return;
    }

    const { top, left, width, height } = ref.current.getBoundingClientRect();
    const x = left + width / 2;
    const y = top + height / 2;
    const maxRadius = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));

    const transition = doc.startViewTransition(apply);
    transition.ready.then(() => {
      document.documentElement.animate(
        { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${maxRadius}px at ${x}px ${y}px)`] },
        { duration, easing: "ease-in-out", pseudoElement: "::view-transition-new(root)" },
      );
    });
  }, [isDark, setTheme, duration]);

  return (
    <button
      ref={ref}
      type="button"
      onClick={toggle}
      aria-label={label}
      className={cn(
        "text-muted-foreground hover:bg-accent hover:text-foreground focus-visible:ring-ring/50 inline-flex size-8 items-center justify-center rounded-lg transition-colors focus-visible:outline-none focus-visible:ring-[3px]",
        className,
      )}
    >
      {isDark ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </button>
  );
}