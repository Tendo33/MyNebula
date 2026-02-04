export interface GraphNode {
  id: number;
  github_id: number;
  full_name: string;
  name: string;
  description?: string;
  language?: string;
  html_url: string;
  // Owner info
  owner: string;
  owner_avatar_url?: string;
  // Position
  x: number;
  y: number;
  z: number;
  // Clustering
  cluster_id?: number;
  color: string;
  size: number;
  // User's star list (GitHub user-defined category)
  star_list_id?: number;
  star_list_name?: string;
  // Stats
  stargazers_count: number;
  ai_summary?: string;
  ai_tags?: string[];
  topics?: string[];
  starred_at?: string;
}

export interface GraphEdge {
  source: number | GraphNode; // force-graph can mutate this to object
  target: number | GraphNode;
  weight: number;
}

export interface ClusterInfo {
  id: number;
  name?: string;
  description?: string;
  keywords: string[];
  color: string;
  repo_count: number;
  center_x?: number;
  center_y?: number;
  center_z?: number;
}

export interface StarListInfo {
  id: number;
  name: string;
  description?: string;
  repo_count: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  clusters: ClusterInfo[];
  star_lists: StarListInfo[];
  total_nodes: number;
  total_edges: number;
  total_clusters: number;
  total_star_lists: number;
}

export interface TimelinePoint {
  date: string;
  count: number;
  repos: string[];
  top_languages: string[];
  top_topics: string[];
}

export interface TimelineData {
  points: TimelinePoint[];
  total_stars: number;
  date_range: [string, string];
}
