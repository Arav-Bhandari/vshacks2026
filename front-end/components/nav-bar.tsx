"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Search, Upload, History } from "lucide-react";
import { cn } from "@/lib/utils";

const links = [
  { href: "/", label: "Search", icon: Search },
  { href: "/protocol/upload", label: "Upload Protocol", icon: Upload },
  { href: "/sessions", label: "Sessions", icon: History },
];

export function NavBar() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-canvas/85 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-2 group">
          <Activity
            className="h-5 w-5 text-accent transition-transform duration-200 group-hover:scale-110"
            strokeWidth={2.25}
          />
          <span className="font-display text-lg tracking-tight">
            TrialScope <span className="italic text-accent">AI</span>
          </span>
        </Link>
        <nav className="flex items-center gap-1">
          {links.map(({ href, label, icon: Icon }) => {
            const active =
              href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-3 py-2 text-sm font-medium transition-colors duration-200",
                  active
                    ? "bg-accent-soft text-accent"
                    : "text-ink-muted hover:text-ink hover:bg-accent-soft/50",
                )}
              >
                <Icon className="h-4 w-4" strokeWidth={2} />
                <span className="hidden sm:inline">{label}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
