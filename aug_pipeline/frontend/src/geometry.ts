export type Point = [number, number];
export function imagePoint(clientX: number, clientY: number, rect: {left: number; top: number; width: number; height: number}, width: number, height: number): Point {
  return [Math.max(0, Math.min(width - 1, (clientX - rect.left) / rect.width * width)),
          Math.max(0, Math.min(height - 1, (clientY - rect.top) / rect.height * height))];
}
export function boxFrom(a: Point, b: Point): [number, number, number, number] | null {
  const box: [number, number, number, number] = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
  return box[2] - box[0] >= 2 && box[3] - box[1] >= 2 ? box : null;
}
