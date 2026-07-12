"use client";

import { useCallback, useRef, useState, type DragEvent } from "react";
import { useRouter } from "next/navigation";
import { FileText, UploadCloud, AlertCircle, Loader2 } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState<"idle" | "uploading" | "error">(
    "idle",
  );
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const acceptFile = useCallback((f: File | undefined) => {
    if (!f) return;
    if (f.type !== "application/pdf") {
      setError("Please upload a PDF file.");
      return;
    }
    setError(null);
    setFile(f);
  }, []);

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    acceptFile(e.dataTransfer.files?.[0]);
  }

  async function startAnalysis() {
    if (!file) return;
    setStatus("uploading");
    setError(null);
    try {
      const { session_id } = await api.uploadProtocol(file);
      await api.analyzeProtocol(session_id);
      router.push(`/protocol/${session_id}/analysis`);
    } catch (err) {
      setStatus("error");
      setError(
        err instanceof ApiError
          ? err.message
          : "Upload failed. Is the backend running?",
      );
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-6 py-16">
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-accent">
        Protocol analysis
      </p>
      <h1 className="mt-3 font-display text-3xl tracking-tight">
        Upload a draft protocol
      </h1>
      <p className="mt-3 max-w-xl text-ink-muted">
        We&apos;ll convert it to USDM, benchmark it against similar trials,
        score patient burden, predict duration risk, and check FDA
        compliance.
      </p>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        className={cn(
          "mt-10 flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed px-6 py-16 text-center",
          "transition-[border-color,background-color,transform] duration-150 ease-out",
          dragging
            ? "scale-[1.01] border-accent bg-accent-soft"
            : "border-border bg-surface hover:border-accent/50",
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(e) => acceptFile(e.target.files?.[0])}
        />
        {file ? (
          <>
            <FileText className="h-8 w-8 text-accent" />
            <p className="font-medium text-ink">{file.name}</p>
            <p className="text-xs text-ink-muted">
              {(file.size / 1024 / 1024).toFixed(2)} MB &mdash; click to
              replace
            </p>
          </>
        ) : (
          <>
            <UploadCloud className="h-8 w-8 text-ink-muted" />
            <p className="font-medium text-ink">
              Drag and drop a protocol PDF here
            </p>
            <p className="text-xs text-ink-muted">or click to browse</p>
          </>
        )}
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-2 rounded-md border border-red/30 bg-red-soft px-4 py-3 text-sm text-red">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {error}
        </div>
      )}

      <div className="mt-6 flex justify-end">
        <Button
          size="lg"
          disabled={!file || status === "uploading"}
          onClick={startAnalysis}
        >
          {status === "uploading" ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin [animation-duration:600ms]" /> Starting
              analysis…
            </>
          ) : (
            "Analyze protocol"
          )}
        </Button>
      </div>
    </div>
  );
}
