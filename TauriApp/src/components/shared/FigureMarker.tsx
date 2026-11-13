/* ──────────────────────────────────────────────────────────
   FigureMarker — the numbered dot that ties an image tile to the
   figure tabs that use it. The figure tab shows the same number,
   so "2" on a tile means "used in the second tab". The figure on
   screen is filled green (the same green as a tile used in this
   figure); every other figure is an outlined, neutral dot.
   ────────────────────────────────────────────────────────── */

interface FigureMarkerProps {
  /** 1-based tab position, or a label such as "+3". */
  label: string | number;
  current: boolean;
  size?: number;
  title?: string;
}

export function FigureMarker({ label, current, size = 12, title }: FigureMarkerProps) {
  return (
    <span
      title={title}
      className="inline-flex items-center justify-center rounded-full font-bold leading-none flex-none"
      style={{
        minWidth: size,
        height: size,
        padding: "0 2px",
        fontSize: Math.max(7, Math.round(size * 0.66)),
        boxSizing: "border-box",
        ...(current
          ? { backgroundColor: "var(--c-green)", color: "#ffffff", border: "1px solid var(--c-green)" }
          : { backgroundColor: "var(--c-surface)", color: "var(--c-text)", border: "1px solid var(--c-text-dim)" }),
      }}
    >
      {label}
    </span>
  );
}
