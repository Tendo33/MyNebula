import client from '../client';

export interface DashboardSummary {
  total_repos: number;
  embedded_repos: number;
  total_topics: number;
  total_clusters: number;
  total_edges: number;
}

export interface DashboardLanguageStat {
  language: string;
  count: number;
}

export interface DashboardTopicStat {
  topic: string;
  count: number;
}

export interface DashboardCluster {
  id: number;
  name?: string;
  repo_count: number;
  color?: string;
  keywords: string[];
}

export interface DashboardResponse {
  summary: DashboardSummary;
  top_languages: DashboardLanguageStat[];
  top_topics: DashboardTopicStat[];
  top_clusters: DashboardCluster[];
  version?: string;
  generated_at?: string;
  request_id?: string;
}

export const getDashboardV2 = async (): Promise<DashboardResponse> => {
  const response = await client.get<DashboardResponse>('/v2/dashboard');
  return response.data;
};
