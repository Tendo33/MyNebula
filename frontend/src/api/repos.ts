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

export const getRelatedRepos = async (
  repoId: number,
  params?: {
    limit?: number;
    min_score?: number;
  }
): Promise<RelatedRepoResponse[]> => {
  const response = await client.get<RelatedRepoResponse[]>(`/repos/${repoId}/related`, {
    params: {
      limit: params?.limit ?? 20,
      min_score: params?.min_score ?? 0.35,
    },
  });
  return response.data;
};

export const submitRelatedFeedback = async (
  repoId: number,
  payload: RelatedFeedbackPayload
): Promise<void> => {
  await client.post(`/repos/${repoId}/related-feedback`, payload);
};
