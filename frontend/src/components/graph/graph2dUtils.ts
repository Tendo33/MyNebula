export const GRAPH_2D_COLORS = {
  NODE_DEFAULT: '#6B7280',
  NODE_HOVER: '#8B5CF6',
  NODE_SELECTED: '#3B82F6',
  NODE_NEIGHBOR: '#60A5FA',
  NODE_DIM: 'rgba(107, 114, 128, 0.3)',
  LINK_DEFAULT: 'rgba(156, 163, 175, 0.4)',
  LINK_ACTIVE: 'rgba(139, 92, 246, 0.6)',
  LINK_DIM: 'rgba(156, 163, 175, 0.1)',
  CLUSTER_BG: 'rgba(0, 0, 0, 0.03)',
  LABEL_BG: 'rgba(255, 255, 255, 0.9)',
  LABEL_TEXT: '#1F2937',
} as const;

const NODE_BASE_SIZE = 5;
const NODE_MAX_SIZE = 30;

export const getNodeId = (node: number | { id: number }): number =>
  typeof node === 'object' ? node.id : node;

export const calculateNodeRadius = (stars: number): number => {
  const base = Math.log10(Math.max(stars, 1) + 1) * NODE_BASE_SIZE;
  return Math.min(Math.max(base, NODE_BASE_SIZE), NODE_MAX_SIZE);
};

export const computeConvexHull = (
  points: { x: number; y: number }[]
): { x: number; y: number }[] => {
  if (points.length < 3) return points;

  let start = 0;
  for (let index = 1; index < points.length; index += 1) {
    if (
      points[index].y < points[start].y ||
      (points[index].y === points[start].y && points[index].x < points[start].x)
    ) {
      start = index;
    }
  }

  [points[0], points[start]] = [points[start], points[0]];
  const pivot = points[0];

  points.sort((left, right) => {
    if (left === pivot) return -1;
    if (right === pivot) return 1;

    const angleLeft = Math.atan2(left.y - pivot.y, left.x - pivot.x);
    const angleRight = Math.atan2(right.y - pivot.y, right.x - pivot.x);

    if (angleLeft !== angleRight) return angleLeft - angleRight;

    const distanceLeft = (left.x - pivot.x) ** 2 + (left.y - pivot.y) ** 2;
    const distanceRight = (right.x - pivot.x) ** 2 + (right.y - pivot.y) ** 2;
    return distanceLeft - distanceRight;
  });

  const hull: { x: number; y: number }[] = [];
  for (const point of points) {
    while (hull.length >= 2) {
      const left = hull[hull.length - 2];
      const right = hull[hull.length - 1];
      const cross =
        (right.x - left.x) * (point.y - left.y) -
        (right.y - left.y) * (point.x - left.x);
      if (cross <= 0) hull.pop();
      else break;
    }
    hull.push(point);
  }

  return hull;
};
