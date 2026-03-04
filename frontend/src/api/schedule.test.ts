import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./v2/settings', () => ({
  getSettingsV2: vi.fn(),
  updateScheduleV2: vi.fn(),
  triggerFullRefreshV2: vi.fn(),
  getFullRefreshJobStatusV2: vi.fn(),
}));

import {
  getSchedule,
  getSyncInfo,
  triggerFullRefresh,
  updateSchedule,
} from './schedule';
import {
  getSettingsV2,
  triggerFullRefreshV2,
  updateScheduleV2,
} from './v2/settings';

describe('api/schedule compatibility wrappers', () => {
  beforeEach(() => {
    vi.mocked(getSettingsV2).mockReset();
    vi.mocked(updateScheduleV2).mockReset();
    vi.mocked(triggerFullRefreshV2).mockReset();
  });

  it('reads schedule and sync info from v2 settings payload', async () => {
    vi.mocked(getSettingsV2).mockResolvedValue({
      schedule: {
        is_enabled: true,
        schedule_hour: 9,
        schedule_minute: 0,
        timezone: 'Asia/Shanghai',
        last_run_at: null,
        last_run_status: null,
        last_run_error: null,
        next_run_at: null,
      },
      sync_info: {
        last_sync_at: null,
        github_token_configured: true,
        single_user_mode: true,
        total_repos: 10,
        synced_repos: 10,
        embedded_repos: 10,
        summarized_repos: 10,
        schedule: null,
      },
      graph_defaults: {
        max_clusters: 8,
        min_clusters: 3,
        related_min_semantic: 0.65,
        hq_rendering: true,
        show_trajectories: true,
      },
    });

    const schedule = await getSchedule();
    const syncInfo = await getSyncInfo();

    expect(schedule.schedule_hour).toBe(9);
    expect(syncInfo.total_repos).toBe(10);
  });

  it('updates schedule through v2 endpoint', async () => {
    vi.mocked(updateScheduleV2).mockResolvedValue({
      schedule: {
        is_enabled: true,
        schedule_hour: 10,
        schedule_minute: 30,
        timezone: 'Asia/Shanghai',
        last_run_at: null,
        last_run_status: null,
        last_run_error: null,
        next_run_at: null,
      },
    });
    const payload = await updateSchedule({
      is_enabled: true,
      schedule_hour: 10,
      schedule_minute: 30,
      timezone: 'Asia/Shanghai',
    });

    expect(payload.schedule_hour).toBe(10);
    expect(updateScheduleV2).toHaveBeenCalledTimes(1);
  });

  it('maps full refresh response to legacy shape', async () => {
    vi.mocked(triggerFullRefreshV2).mockResolvedValue({
      task: {
        task_id: 100,
        message: 'started',
        reset_count: 66,
      },
    });

    const payload = await triggerFullRefresh();
    expect(payload).toEqual({
      task_id: 100,
      message: 'started',
      reset_count: 66,
    });
  });
});
