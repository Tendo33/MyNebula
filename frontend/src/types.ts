/** Shapes of the static JSON written by `pnpm sync`. List ids are GitHub list ids. */

/** Repos in no star list use this id once the owner has at least one list. */
export const UNCATEGORIZED_LIST_ID = 'uncategorized';

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
  star_list_ids: string[];
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
  id: string;
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

