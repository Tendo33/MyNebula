import client from '../client';

export interface ScheduleConfig {
  is_enabled: boolean;
  schedule_hour: number;
  schedule_minute: number;
  timezone: string;
}

export interface ScheduleResponse extends ScheduleConfig {
  last_run_at: string | null;
  last_run_status: 'success' | 'failed' | 'running' | null;
  last_run_error: string | null;
  next_run_at: string | null;
}

export interface SyncInfoResponse {
  last_sync_at: string | null;
  github_token_configured: boolean;
  single_user_mode: boolean;
  total_repos: number;
  synced_repos: number;
  embedded_repos: number;
  summarized_repos: number;
  schedule: ScheduleResponse | null;
}

export interface GraphDefaults {
  max_clusters: number;
  min_clusters: number;
  related_min_semantic: number;
  hq_rendering: boolean;
  show_trajectories: boolean;
}

export interface SettingsResponse {
  schedule: ScheduleResponse;
  sync_info: SyncInfoResponse;
  graph_defaults: GraphDefaults;
  version?: string;
  generated_at?: string;
  request_id?: string;
}

export interface ScheduleUpdateResponse {
  schedule: ScheduleResponse;
  version?: string;
  generated_at?: string;
  request_id?: string;
}

export interface FullRefreshTask {
  task_id: number;
  message: string;
  reset_count: number;
}

export interface FullRefreshStartResponse {
  task: FullRefreshTask;
  version?: string;
  generated_at?: string;
  request_id?: string;
}

export interface FullRefreshJobStatus {
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

export interface FullRefreshJobResponse {
  job: FullRefreshJobStatus;
  version?: string;
  generated_at?: string;
  request_id?: string;
}

export const getSettingsV2 = async (): Promise<SettingsResponse> => {
  const response = await client.get<SettingsResponse>('/v2/settings');
  return response.data;
};

export const updateScheduleV2 = async (
  config: ScheduleConfig
): Promise<ScheduleUpdateResponse> => {
  const response = await client.post<ScheduleUpdateResponse>('/v2/settings/schedule', config);
  return response.data;
};

export const triggerFullRefreshV2 = async (): Promise<FullRefreshStartResponse> => {
  const response = await client.post<FullRefreshStartResponse>('/v2/settings/full-refresh', {
    confirm: true,
  });
  return response.data;
};

export const getFullRefreshJobStatusV2 = async (
  taskId: number
): Promise<FullRefreshJobResponse> => {
  const response = await client.get<FullRefreshJobResponse>(
    `/v2/settings/full-refresh/jobs/${taskId}`
  );
  return response.data;
};
