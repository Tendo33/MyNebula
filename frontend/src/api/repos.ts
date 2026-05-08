import client from './client';

export interface RepoRelatedItem {
  id: number;
  github_repo_id: number;
  full_name: string;
  owner: string;
  name: string;
  description?: string;
  language?: string;
  html_url: string;
  stargazers_count: number;
  ai_summary?: string;
  topics: string[];
}

export interface RelatedScoreComponents {
  semantic: number;
  tag_overlap: number;
  same_star_list: number;
  same_language: number;
}

export interface RelatedRepoResponse {
  repo: RepoRelatedItem;
  score: number;
  reasons: string[];
  components: RelatedScoreComponents;
}

export interface RelatedFeedbackPayload {
  candidate_repo_id: number;
  feedback: 'helpful' | 'not_helpful';
  score_snapshot?: number;
  model_version?: string;
}

export interface RepoSearchPayload {
  query: string;
  limit?: number;
  language?: string;
  cluster_id?: number;
  min_stars?: number;
}

export interface RepoSearchItem {
  repo: RepoRelatedItem & {
    forks_count: number;
    watchers_count: number;
    open_issues_count: number;
    cluster_id: number | null;
    coord_x: number | null;
    coord_y: number | null;
    coord_z: number | null;
    starred_at: string | null;
    repo_updated_at: string | null;
    is_embedded: boolean;
    is_summarized: boolean;
  };
  score: number;
  highlight?: string | null;
}

export const getRelatedRepos = async (
  repoId: number,
  params?: {
    limit?: number;
    min_score?: number;
    min_semantic?: number;
  }
): Promise<RelatedRepoResponse[]> => {
  const response = await client.get<RelatedRepoResponse[]>(`/v2/repos/${repoId}/related`, {
    params: {
      limit: params?.limit ?? 20,
      min_score: params?.min_score ?? 0.4,
      min_semantic: params?.min_semantic ?? 0.65,
    },
  });
  return response.data;
};

export const submitRelatedFeedback = async (
  repoId: number,
  payload: RelatedFeedbackPayload
): Promise<void> => {
  await client.post(`/v2/repos/${repoId}/related-feedback`, payload);
};

export const searchRepos = async (
  payload: RepoSearchPayload,
  options?: { signal?: AbortSignal }
): Promise<RepoSearchItem[]> => {
  const response = await client.post<RepoSearchItem[]>('/v2/repos/search', payload, {
    signal: options?.signal,
  });
  return response.data;
};
