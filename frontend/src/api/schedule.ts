import {
  getSettingsV2,
  getFullRefreshJobStatusV2,
  triggerFullRefreshV2,
  updateScheduleV2,
  type FullRefreshJobResponse,
  type ScheduleConfig,
  type ScheduleResponse,
  type SyncInfoResponse,
} from './v2/settings';
import {
  formatLastRunTime,
  formatNextRunTime,
  getStatusDisplay,
} from '../utils/scheduleFormat';

export type { ScheduleConfig, ScheduleResponse, SyncInfoResponse };

export interface FullRefreshResponse {
  task_id: number;
  message: string;
  reset_count: number;
}

export interface FullRefreshRequest {
  confirm: boolean;
}

const toLegacyFullRefreshResponse = (
  payload: Awaited<ReturnType<typeof triggerFullRefreshV2>>
): FullRefreshResponse => ({
  task_id: payload.task.task_id,
  message: payload.task.message,
  reset_count: payload.task.reset_count,
});

/**
 * @deprecated Use `frontend/src/api/v2/settings.ts` instead.
 */
export const getSchedule = async (): Promise<ScheduleResponse> => {
  const response = await getSettingsV2();
  return response.schedule;
};

/**
 * @deprecated Use `frontend/src/api/v2/settings.ts` instead.
 */
export const updateSchedule = async (config: ScheduleConfig): Promise<ScheduleResponse> => {
  const response = await updateScheduleV2(config);
  return response.schedule;
};

/**
 * @deprecated Use `frontend/src/api/v2/settings.ts` instead.
 */
export const getSyncInfo = async (): Promise<SyncInfoResponse> => {
  const response = await getSettingsV2();
  return response.sync_info;
};

/**
 * @deprecated Use `frontend/src/api/v2/settings.ts` instead.
 */
export const triggerFullRefresh = async (): Promise<FullRefreshResponse> => {
  return toLegacyFullRefreshResponse(await triggerFullRefreshV2());
};

/**
 * @deprecated Use `frontend/src/api/v2/settings.ts` instead.
 */
export const getFullRefreshJobStatus = async (
  taskId: number
): Promise<FullRefreshJobResponse> => {
  return getFullRefreshJobStatusV2(taskId);
};

export { formatLastRunTime, formatNextRunTime, getStatusDisplay };
