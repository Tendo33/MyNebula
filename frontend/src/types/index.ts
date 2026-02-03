export interface GraphNode {
  id: number;
  github_id: number;
  full_name: string;
  name: string;
  description?: string;
  language?: string;
  html_url: string;
  x: number;
  y: number;
  z: number;
  cluster_id?: number;
  color: string;
  size: number;
  stargazers_count: number;
  ai_summary?: string;
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

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  clusters: ClusterInfo[];
  total_nodes: number;
  total_edges: number;
  total_clusters: number;
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
