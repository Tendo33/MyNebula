import client from '../client';

export interface PipelineStartResponse {
  pipeline_run_id: number;
  status: string;
  phase: string;
  message: string;
  version?: string;
  generated_at?: string;
  request_id?: string;
}

export interface PipelineStatusResponse {
  pipeline_run_id: number;
  user_id: number;
  status: string;
  phase: string;
  last_error?: string | null;
  created_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  version?: string;
  generated_at?: string;
  request_id?: string;
}

export const startSyncPipelineV2 = async (params?: {
  mode?: 'incremental' | 'full';
  use_llm?: boolean;
  max_clusters?: number;
  min_clusters?: number;
}): Promise<PipelineStartResponse> => {
  const response = await client.post<PipelineStartResponse>('/v2/sync/start', null, {
    params: {
      mode: params?.mode ?? 'incremental',
      use_llm: params?.use_llm ?? true,
      max_clusters: params?.max_clusters ?? 8,
      min_clusters: params?.min_clusters ?? 3,
    },
  });
  return response.data;
};

export const getPipelineStatusV2 = async (
  runId: number
): Promise<PipelineStatusResponse> => {
  const response = await client.get<PipelineStatusResponse>(`/v2/sync/jobs/${runId}`);
  return response.data;
};
