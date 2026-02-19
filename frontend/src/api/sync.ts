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

/**
 * 开始同步星标仓库
 * @param mode 同步模式: 'incremental' (增量) 或 'full' (全量)
 */
export const startStarSync = async (mode: "incremental" | "full" = "incremental") => {
	const response = await client.post<SyncStartResponse>("/sync/stars", null, {
		params: { mode },
	});
	return response.data;
};

/**
 * 获取同步任务状态
 * @param taskId 任务 ID
 */
export const getSyncStatus = async (taskId: number) => {
	const response = await client.get<SyncStatusResponse>(`/sync/status/${taskId}`);
	return response.data;
};

/**
 * 获取所有同步任务状态
 */
export const getAllSyncStatus = async () => {
	const response = await client.get<SyncStatusResponse[]>("/sync/status");
	return response.data;
};

/**
 * 获取聚合任务状态（包含阶段和 ETA）
 */
export const getSyncJobStatus = async (taskId: number) => {
  const response = await client.get<SyncJobStatusResponse>(`/sync/jobs/${taskId}`);
  return response.data;
};

/**
 * 开始计算向量嵌入（增量：只处理未嵌入的仓库）
 */
export const startEmbedding = async () => {
	const response = await client.post<SyncStartResponse>("/sync/embeddings");
	return response.data;
};

/**
 * 开始生成 AI 摘要和标签
 */
export const startSummaries = async () => {
	const response = await client.post<SyncStartResponse>("/sync/summaries");
	return response.data;
};

/**
 * 开始运行聚类（生成3D坐标和分类）
 */
export const startClustering = async (
  useLlm: boolean = true,
  maxClusters: number = 8,
  minClusters: number = 2
) => {
	const response = await client.post<SyncStartResponse>("/sync/clustering", null, {
		params: {
			use_llm: useLlm,
			max_clusters: maxClusters,
			min_clusters: minClusters,
		},
	});
	return response.data;
};
