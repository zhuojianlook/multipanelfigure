/**
 * Shared symbol definitions for SVG rendering.
 * Matches python-sidecar/symbol_defs.py EXACTLY.
 *
 * TIP-BASED symbols (Arrow, NarrowTriangle, Arrowhead):
 *   (0,0) = the TIP point. Tip points in rotation direction.
 *
 * CENTER-BASED symbols (Triangle, Star, Rectangle, Ellipse, Cross):
 *   (0,0) = the CENTER of the shape.
 */

type Point = [number, number];
type SymbolData = {
  fill: Point[][];
  stroke: Point[][];
  filled: boolean;
};

function rot(points: Point[], angleDeg: number): Point[] {
  if (Math.abs(angleDeg) < 0.1) return points;
  const rad = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  return points.map(([x, y]) => [x * cos - y * sin, x * sin + y * cos]);
}

/** `width` is a cross-axis thickness multiplier used ONLY by the
 *  direction-based symbols (Arrow, NarrowTriangle) — it scales the shape
 *  perpendicular to its pointing direction. All other shapes ignore it. */
export function getSymbolPolys(shape: string, rotation: number = 0, width: number = 1.0): SymbolData {
  let fill: Point[][] = [];
  let stroke: Point[][] = [];
  let filled = false;

  // ── TIP-BASED: (0,0) = tip ──

  if (shape === "Arrow") {
    // Single filled polygon — a clean arrow with a sharp tip. See
    // symbol_defs.py for the full rationale: rendering the arrow as ONE
    // filled polygon (not a stroked shaft + filled head) makes it look
    // identical in the edit-panel SVG overlay and the final figure,
    // because stroke width has no shared unit across the render paths.
    // head_len >> head_hw gives the pointed tip.
    const headLen = 0.38;            // length (unaffected by width)
    const headHW = 0.11 * width;     // head base half-width
    const shaftLen = 0.6;            // total length (unaffected by width)
    const shaftHW = 0.03 * width;    // shaft half-thickness
    const arrow: Point[] = [
      [0.0, 0.0],               // tip
      [-headLen, -headHW],      // head back, upper
      [-headLen, -shaftHW],     // head -> shaft junction, upper
      [-shaftLen, -shaftHW],    // shaft tail, upper
      [-shaftLen, shaftHW],     // shaft tail, lower
      [-headLen, shaftHW],      // head -> shaft junction, lower
      [-headLen, headHW],       // head back, lower
    ];
    fill = [rot(arrow, rotation)];
    filled = true;
  } else if (shape === "NarrowTriangle") {
    // Tip at origin pointing RIGHT (same direction as Arrow at rot=0).
    // `width` scales the base half-width.
    const hw = 0.12 * width;
    const pts: Point[] = [[0, 0], [-1.0, -hw], [-1.0, hw]];
    fill = [rot(pts, rotation)];
    filled = true;
  } else if (shape === "Arrowhead") {
    const pts: Point[] = [[0, 0], [-0.35, -0.2], [-0.2, 0], [-0.35, 0.2]];
    fill = [rot(pts, rotation)];
    filled = true;

  // ── CENTER-BASED: (0,0) = center ──

  } else if (shape === "Triangle") {
    const pts: Point[] = [[0, -0.45], [-0.39, 0.22], [0.39, 0.22]];
    fill = [rot(pts, rotation)];
    filled = false;
  } else if (shape === "Star") {
    const pts: Point[] = [];
    for (let i = 0; i < 10; i++) {
      const angle = ((i * 36 - 90) * Math.PI) / 180;
      const r = i % 2 === 0 ? 0.48 : 0.2;
      pts.push([r * Math.cos(angle), r * Math.sin(angle)]);
    }
    fill = [rot(pts, rotation)];
    filled = false;
  } else if (shape === "Asterisk") {
    const arms: Point[][] = [];
    for (let i = 0; i < 6; i++) {
      const angle = (i * 30 * Math.PI) / 180;
      arms.push([[0, 0], [0.45 * Math.cos(angle), 0.45 * Math.sin(angle)]]);
    }
    stroke = arms.map((arm) => rot(arm, rotation));
  } else if (shape === "Rectangle") {
    const pts: Point[] = [[-0.4, -0.4], [0.4, -0.4], [0.4, 0.4], [-0.4, 0.4]];
    fill = [rot(pts, rotation)];
    filled = false;
  } else if (shape === "Ellipse") {
    const pts: Point[] = [];
    for (let i = 0; i < 32; i++) {
      const angle = (i * 360 / 32 * Math.PI) / 180;
      pts.push([0.4 * Math.cos(angle), 0.4 * Math.sin(angle)]);
    }
    fill = [rot(pts, rotation)];
    filled = false;
  } else if (shape === "Cross") {
    const h: Point[] = [[-0.45, 0], [0.45, 0]];
    const v: Point[] = [[0, -0.45], [0, 0.45]];
    stroke = [rot(h, rotation), rot(v, rotation)];
  } else {
    // Fallback circle
    const pts: Point[] = [];
    for (let i = 0; i < 12; i++) {
      const angle = (i * 30 * Math.PI) / 180;
      pts.push([0.3 * Math.cos(angle), 0.3 * Math.sin(angle)]);
    }
    fill = [pts];
  }

  return { fill, stroke, filled };
}

export function symbolToSvgPoints(
  shape: string, cx: number, cy: number, sz: number, rotation: number = 0, width: number = 1.0
): { fillPolys: string[]; strokePolys: string[]; filled: boolean } {
  const data = getSymbolPolys(shape, rotation, width);
  const fillPolys = data.fill.map((poly) =>
    poly.map(([x, y]) => `${cx + x * sz},${cy + y * sz}`).join(" ")
  );
  const strokePolys = data.stroke.map((poly) =>
    poly.map(([x, y]) => `${cx + x * sz},${cy + y * sz}`).join(" ")
  );
  return { fillPolys, strokePolys, filled: data.filled };
}
