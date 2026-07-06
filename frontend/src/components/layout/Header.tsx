import { ThemeToggle } from "@/components/theme-toggle";

/** Top row: page title on the left, theme toggle on the right. */
export function Header({ title }: { title: string }) {
  return (
    <header className="flex items-center justify-between border-b px-5 py-3">
      <span className="text-sm font-medium">{title}</span>
      <ThemeToggle />
    </header>
  );
}