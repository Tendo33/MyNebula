import client from './client';

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

/**
 * 开始同步星标仓库
 * @param mode 同步模式: 'incremental' (增量) 或 'full' (全量)
 */
export const startStarSync = async (mode: 'incremental' | 'full' = 'incremental') => {
  const token = localStorage.getItem("token");
  const response = await client.post<SyncStartResponse>("/sync/stars", null, {
    params: { token, mode },
  });
  return response.data;
};

/**
 * 获取同步任务状态
 * @param taskId 任务 ID
 */
export const getSyncStatus = async (taskId: number) => {
  const token = localStorage.getItem("token");
  const response = await client.get<SyncStatusResponse>(`/sync/status/${taskId}`, {
    params: { token },
  });
  return response.data;
};

/**
 * 获取所有同步任务状态
 */
export const getAllSyncStatus = async () => {
  const token = localStorage.getItem("token");
  const response = await client.get<SyncStatusResponse[]>("/sync/status", {
    params: { token },
  });
  return response.data;
};
