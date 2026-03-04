import client from '../client';

export interface DataRepoItem {
  id: number;
  full_name: string;
  name: string;
  owner: string;
  owner_avatar_url?: string;
  description?: string;
  ai_summary?: string;
  topics: string[];
  language?: string;
  stargazers_count: number;
  html_url: string;
  cluster_id: number | null;
  star_list_id: number | null;
  starred_at?: string | null;
  last_commit_time?: string | null;
}

export interface DataReposResponse {
  items: DataRepoItem[];
  count: number;
  limit: number;
  offset: number;
  version?: string;
  generated_at?: string;
  request_id?: string;
}

export const getDataReposV2 = async (params?: {
  cluster_id?: number;
  language?: string;
  min_stars?: number;
  q?: string;
  limit?: number;
  offset?: number;
}): Promise<DataReposResponse> => {
  const response = await client.get<DataReposResponse>('/v2/data/repos', { params });
  return response.data;
};
