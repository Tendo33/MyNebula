import client from './client';
import {
  getPipelineStatusV2,
  startSyncPipelineV2,
  type PipelineStatusResponse,
} from './v2/sync';

export interface SyncStartResponse {
  task_id: number;
  message: string;
  status: string;
}

export interface SyncStatusResponse {
  task_id: number | null;
  status: string;
  task_type: string;
  total_items: number;
  processed_items: number;
  failed_items: number;
  progress_percent: number;
  error_message: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface SyncJobStatusResponse {
  task_id: number;
  task_type: string;
  status: string;
  phase: string;
  progress_percent: number;
  eta_seconds: number | null;
  last_error: string | null;
  retryable: boolean;
  started_at: string | null;
  completed_at: string | null;
}

const toLegacySyncStartResponse = (
  run: Awaited<ReturnType<typeof startSyncPipelineV2>>
): SyncStartResponse => ({
  task_id: run.pipeline_run_id,
  message: run.message,
  status: run.status,
});

const toLegacySyncStatusResponse = (
  pipeline: PipelineStatusResponse,
  taskType: string
): SyncStatusResponse => {
  const statusToProgress: Record<string, number> = {
    pending: 0,
    running: 50,
    partial_failed: 95,
    completed: 100,
    failed: 100,
  };
  return {
    task_id: pipeline.pipeline_run_id,
    status: pipeline.status,
    task_type: taskType,
    total_items: 100,
    processed_items: statusToProgress[pipeline.status] ?? 0,
    failed_items: pipeline.status === 'failed' || pipeline.status === 'partial_failed' ? 1 : 0,
    progress_percent: statusToProgress[pipeline.status] ?? 0,
    error_message: pipeline.last_error ?? null,
    started_at: pipeline.started_at ?? null,
    completed_at: pipeline.completed_at ?? null,
  };
};

const toLegacySyncJobStatusResponse = (pipeline: PipelineStatusResponse): SyncJobStatusResponse => ({
  task_id: pipeline.pipeline_run_id,
  task_type: 'pipeline',
  status: pipeline.status,
  phase: pipeline.phase,
  progress_percent:
    pipeline.status === 'completed'
      ? 100
      : pipeline.status === 'failed'
        ? 100
        : pipeline.status === 'running'
          ? 50
          : 0,
  eta_seconds: null,
  last_error: pipeline.last_error ?? null,
  retryable: pipeline.status === 'failed' || pipeline.status === 'partial_failed',
  started_at: pipeline.started_at ?? null,
  completed_at: pipeline.completed_at ?? null,
});

/**
 * @deprecated Use `frontend/src/api/v2/sync.ts` instead.
 */
export const startStarSync = async (mode: 'incremental' | 'full' = 'incremental') => {
  const response = await startSyncPipelineV2({ mode });
  return toLegacySyncStartResponse(response);
};

/**
 * @deprecated Use `frontend/src/api/v2/sync.ts` instead.
 */
export const getSyncStatus = async (taskId: number) => {
  const response = await getPipelineStatusV2(taskId);
  return toLegacySyncStatusResponse(response, 'pipeline');
};

/**
 * @deprecated Keep legacy fallback for call-sites that still need list status.
 */
export const getAllSyncStatus = async () => {
  const response = await client.get<SyncStatusResponse[]>('/sync/status');
  return response.data;
};

/**
 * @deprecated Use `frontend/src/api/v2/sync.ts` instead.
 */
export const getSyncJobStatus = async (taskId: number) => {
  const response = await getPipelineStatusV2(taskId);
  return toLegacySyncJobStatusResponse(response);
};

/**
 * @deprecated Use `frontend/src/api/v2/sync.ts` instead.
 */
export const startEmbedding = async () => {
  const response = await startSyncPipelineV2({ mode: 'incremental' });
  return toLegacySyncStartResponse(response);
};

/**
 * @deprecated Use `frontend/src/api/v2/sync.ts` instead.
 */
export const startSummaries = async () => {
  const response = await startSyncPipelineV2({ mode: 'incremental' });
  return toLegacySyncStartResponse(response);
};

/**
 * @deprecated Use `frontend/src/api/v2/sync.ts` instead.
 */
export const startClustering = async (
  useLlm: boolean = true,
  maxClusters: number = 8,
  minClusters: number = 2
) => {
  const response = await startSyncPipelineV2({
    mode: 'incremental',
    use_llm: useLlm,
    max_clusters: maxClusters,
    min_clusters: minClusters,
  });
  return toLegacySyncStartResponse(response);
};
