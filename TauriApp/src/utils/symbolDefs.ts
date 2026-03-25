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

export function getSymbolPolys(shape: string, rotation: number = 0): SymbolData {
  let fill: Point[][] = [];
  let stroke: Point[][] = [];
  let filled = false;

  // ── TIP-BASED: (0,0) = tip ──

  if (shape === "Arrow") {
    const shaft: Point[] = [[0, 0], [-0.6, 0]];
    const head: Point[] = [[0, 0], [-0.22, -0.1], [-0.22, 0.1]];
    stroke = [rot(shaft, rotation)];
    fill = [rot(head, rotation)];
    filled = true;
  } else if (shape === "NarrowTriangle") {
    // Tip at origin pointing RIGHT (same direction as Arrow at rot=0)
    const pts: Point[] = [[0, 0], [-1.0, -0.12], [-1.0, 0.12]];
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
  shape: string, cx: number, cy: number, sz: number, rotation: number = 0
): { fillPolys: string[]; strokePolys: string[]; filled: boolean } {
  const data = getSymbolPolys(shape, rotation);
  const fillPolys = data.fill.map((poly) =>
    poly.map(([x, y]) => `${cx + x * sz},${cy + y * sz}`).join(" ")
  );
  const strokePolys = data.stroke.map((poly) =>
    poly.map(([x, y]) => `${cx + x * sz},${cy + y * sz}`).join(" ")
  );
  return { fillPolys, strokePolys, filled: data.filled };
}
