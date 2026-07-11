"use client";

import { cn } from "@/lib/utils";
import {
  createContext,
  useContext,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

type TabsContextValue = {
  value: string;
  setValue: (v: string) => void;
  registerTrigger: (value: string, el: HTMLButtonElement | null) => void;
};

const TabsContext = createContext<TabsContextValue | null>(null);

function useTabsContext() {
  const ctx = useContext(TabsContext);
  if (!ctx) throw new Error("Tabs components must be used within <Tabs>");
  return ctx;
}

export function Tabs({
  defaultValue,
  value: controlledValue,
  onValueChange,
  children,
  className,
}: {
  defaultValue?: string;
  value?: string;
  onValueChange?: (v: string) => void;
  children: ReactNode;
  className?: string;
}) {
  const [internal, setInternal] = useState(defaultValue ?? "");
  const value = controlledValue ?? internal;
  const triggers = useRef(new Map<string, HTMLButtonElement>());
  const setValue = (v: string) => {
    setInternal(v);
    onValueChange?.(v);
  };
  const registerTrigger = (v: string, el: HTMLButtonElement | null) => {
    if (el) triggers.current.set(v, el);
    else triggers.current.delete(v);
  };
  return (
    <TabsContext.Provider
      value={{ value, setValue, registerTrigger }}
    >
      <TabsTriggerMapContext.Provider value={triggers}>
        <div className={className}>{children}</div>
      </TabsTriggerMapContext.Provider>
    </TabsContext.Provider>
  );
}

const TabsTriggerMapContext = createContext<{
  current: Map<string, HTMLButtonElement>;
} | null>(null);

export function TabsList({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  const { value } = useTabsContext();
  const triggers = useContext(TabsTriggerMapContext);
  const listRef = useRef<HTMLDivElement>(null);
  const [indicator, setIndicator] = useState<{ x: number; w: number } | null>(
    null,
  );

  useLayoutEffect(() => {
    const el = triggers?.current.get(value);
    const list = listRef.current;
    if (!el || !list) return;
    const update = () =>
      setIndicator({ x: el.offsetLeft, w: el.offsetWidth });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    ro.observe(list);
    return () => ro.disconnect();
  }, [value, triggers]);

  return (
    <div
      ref={listRef}
      role="tablist"
      className={cn(
        "relative flex gap-1 overflow-x-auto border-b border-border",
        className,
      )}
    >
      {children}
      {indicator && (
        <span
          aria-hidden
          className="pointer-events-none absolute bottom-[-1px] left-0 h-0.5 bg-accent rounded-full transition-transform duration-200"
          style={{
            width: indicator.w,
            transform: `translateX(${indicator.x}px)`,
            transitionTimingFunction: "var(--ease-out)",
          }}
        />
      )}
    </div>
  );
}

export function TabsTrigger({
  value,
  children,
  className,
}: {
  value: string;
  children: ReactNode;
  className?: string;
}) {
  const { value: active, setValue, registerTrigger } = useTabsContext();
  const isActive = active === value;
  return (
    <button
      ref={(el) => registerTrigger(value, el)}
      role="tab"
      type="button"
      aria-selected={isActive}
      onClick={() => setValue(value)}
      className={cn(
        "whitespace-nowrap px-4 py-3 text-sm font-medium transition-colors duration-150",
        isActive ? "text-accent" : "text-ink-muted hover:text-ink",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function TabsContent({
  value,
  children,
  className,
}: {
  value: string;
  children: ReactNode;
  className?: string;
}) {
  const { value: active } = useTabsContext();
  if (active !== value) return null;
  return (
    <div role="tabpanel" className={cn("animate-fade-up", className)}>
      {children}
    </div>
  );
}
