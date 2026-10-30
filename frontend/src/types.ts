/** Shapes of the static JSON written by `python -m nebula`. */

export interface GraphNode {
  id: number;
  full_name: string;
  name: string;
  owner: string;
  owner_avatar_url?: string | null;
  description?: string | null;
  language?: string | null;
  html_url: string;
  stargazers_count: number;
  ai_summary?: string | null;
  ai_tags?: string[] | null;
  topics?: string[] | null;
  starred_at?: string | null;
  last_commit_time?: string | null;
  cluster_id: number | null;
  star_list_id: number | null;
}

/** [source id, target id, weight] */
export type GraphEdge = [number, number, number];

export interface ClusterInfo {
  id: number;
  name: string;
  description?: string | null;
  keywords: string[];
  color: string;
  repo_count: number;
}

export interface StarListInfo {
  id: number;
  name: string;
  description?: string | null;
  repo_count: number;
}

export interface TimelinePoint {
  date: string;
  count: number;
  repos: string[];
  top_languages: string[];
  top_topics: string[];
}

export interface GraphPayload {
  version: string;
  generated_at: string;
  total_repos: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  clusters: ClusterInfo[];
  star_lists: StarListInfo[];
  timeline: TimelinePoint[];
}

/** [id, score, semantic, tag_overlap, flags(bit0 same list, bit1 same language)] */
export type RelatedItem = [number, number, number, number, number];
export type RelatedPayload = Record<string, RelatedItem[]>;
