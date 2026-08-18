import type { ForceGraphMethods, LinkObject, NodeObject } from 'react-force-graph-2d';

export interface ProcessedNode extends NodeObject {
  id: number;
  name: string;
  full_name: string;
  description?: string;
  language?: string;
  cluster_id: number | null;
  color: string;
  size: number;
  stargazers_count: number;
  // Owner info for avatar display
  owner?: string;
  owner_avatar_url?: string;
  // AI-generated content
  ai_summary?: string;
  ai_tags?: string[];
  // Force-graph will add x, y, vx, vy
  x?: number;
  y?: number;
  fx?: number; // Fixed position
  fy?: number;
}

export interface ProcessedLink extends LinkObject {
  source: number | ProcessedNode;
  target: number | ProcessedNode;
  weight: number;
}

export interface ProcessedData {
  nodes: ProcessedNode[];
  links: ProcessedLink[];
}

export interface ClusterCenter {
  x: number;
  y: number;
  count: number;
}

export interface ClusterLayoutData {
  centers: Map<number, ClusterCenter>;
  clusterNodes: Map<number, ProcessedNode[]>;
}

export type RegisteredForce = NonNullable<Parameters<ForceGraphMethods['d3Force']>[1]>;

export type ImageCache = Map<string, HTMLImageElement | 'loading' | 'error'>;

export type HullCache = Map<number, { signature: string; hull: { x: number; y: number }[] }>;

export const ZOOM_TO_FIT_PADDING = 80;

// Layout tuning: larger = more spaced out initial view
export const POSITION_SCALE = 22;
export const MIN_CLUSTER_DISTANCE = 85;
export const CENTER_PULL_STRENGTH = 0.01;
