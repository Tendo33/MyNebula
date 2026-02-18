import client from './client';

// ==================== Type Definitions ====================

/**
 * 调度配置请求/响应基础类型
 */
export interface ScheduleConfig {
	is_enabled: boolean;
	schedule_hour: number;
	schedule_minute: number;
	timezone: string;
}

/**
 * 调度配置完整响应类型
 */
export interface ScheduleResponse extends ScheduleConfig {
	last_run_at: string | null;
	last_run_status: 'success' | 'failed' | 'running' | null;
	last_run_error: string | null;
	next_run_at: string | null;
}

/**
 * 同步状态信息响应类型
 */
export interface SyncInfoResponse {
	last_sync_at: string | null;
	github_token_configured: boolean;
	total_repos: number;
	synced_repos: number;
	embedded_repos: number;
	summarized_repos: number;
	schedule: ScheduleResponse | null;
}

/**
 * 全量刷新响应类型
 */
export interface FullRefreshResponse {
	task_id: number;
	message: string;
	reset_count: number;
}

// ==================== API Functions ====================

/**
 * 获取当前调度配置
 */
export const getSchedule = async (): Promise<ScheduleResponse> => {
	const response = await client.get<ScheduleResponse>('/sync/schedule');
	return response.data;
};

/**
 * 更新调度配置
 * @param config 新的调度配置
 */
export const updateSchedule = async (config: ScheduleConfig): Promise<ScheduleResponse> => {
	const response = await client.post<ScheduleResponse>('/sync/schedule', config);
	return response.data;
};

/**
 * 获取完整的同步状态信息
 */
export const getSyncInfo = async (): Promise<SyncInfoResponse> => {
	const response = await client.get<SyncInfoResponse>('/sync/info');
	return response.data;
};

/**
 * 触发全量刷新
 * 警告：这是一个耗时操作，会重置所有仓库的处理状态
 */
export const triggerFullRefresh = async (): Promise<FullRefreshResponse> => {
	const response = await client.post<FullRefreshResponse>('/sync/full-refresh');
	return response.data;
};

// ==================== Helper Functions ====================

/**
 * 格式化下次运行时间为用户友好的字符串
 * @param nextRunAt ISO 时间字符串
 * @param timezone 用户时区
 * @param t 翻译函数
 */
export const formatNextRunTime = (nextRunAt: string | null, timezone: string, t: any): string => {
	if (!nextRunAt) return t("time.not_set");

	try {
		const date = new Date(nextRunAt);
		return date.toLocaleString(undefined, {
			timeZone: timezone,
			year: "numeric",
			month: "2-digit",
			day: "2-digit",
			hour: "2-digit",
			minute: "2-digit",
		});
	} catch {
		return nextRunAt;
	}
};

/**
 * 格式化上次运行时间
 * @param lastRunAt ISO 时间字符串
 * @param t 翻译函数
 */
export const formatLastRunTime = (lastRunAt: string | null, t: any): string => {
	if (!lastRunAt) return t("time.never_run");

	try {
		const date = new Date(lastRunAt);
		const now = new Date();
		const diffMs = now.getTime() - date.getTime();
		const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
		const diffDays = Math.floor(diffHours / 24);

		if (diffDays > 0) {
			return t("time.days_ago", { count: diffDays });
		} else if (diffHours > 0) {
			return t("time.hours_ago", { count: diffHours });
		} else {
			const diffMinutes = Math.floor(diffMs / (1000 * 60));
			return diffMinutes > 0 ? t("time.minutes_ago", { count: diffMinutes }) : t("time.just_now");
		}
	} catch {
		return lastRunAt;
	}
};

/**
 * 获取运行状态的显示文本和颜色
 * @param status 运行状态
 * @param t 翻译函数
 */
export const getStatusDisplay = (status: string | null, t: any): { text: string; color: string } => {
	switch (status) {
		case "success":
			return { text: t("time.status.success"), color: "text-green-600" };
		case "failed":
			return { text: t("time.status.failed"), color: "text-red-600" };
		case "running":
			return { text: t("time.status.running"), color: "text-blue-600" };
		default:
			return { text: t("time.status.unknown"), color: "text-gray-500" };
	}
};
