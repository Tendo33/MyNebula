import { beforeEach, describe, expect, it, vi } from 'vitest';

const { getMock, postMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
}));

vi.mock('../client', () => ({
  default: {
    get: getMock,
    post: postMock,
  },
}));

import {
  getFullRefreshJobStatusV2,
  getSettingsV2,
  triggerFullRefreshV2,
  updateGraphDefaultsV2,
  updateScheduleV2,
} from './settings';

describe('api/v2/settings client', () => {
  beforeEach(() => {
    getMock.mockReset();
    postMock.mockReset();
  });

  it('loads consolidated settings payload', async () => {
    getMock.mockResolvedValueOnce({ data: { schedule: {}, sync_info: {}, graph_defaults: {} } });
    await getSettingsV2();
    expect(getMock).toHaveBeenCalledWith('/v2/settings');
  });

  it('updates schedule via v2 endpoint', async () => {
    postMock.mockResolvedValueOnce({ data: { schedule: { is_enabled: true } } });
    await updateScheduleV2({
      is_enabled: true,
      schedule_hour: 9,
      schedule_minute: 0,
      timezone: 'Asia/Shanghai',
    });
    expect(postMock).toHaveBeenCalledWith('/v2/settings/schedule', {
      is_enabled: true,
      schedule_hour: 9,
      schedule_minute: 0,
      timezone: 'Asia/Shanghai',
    });
  });

  it('triggers full refresh and polls status through v2 endpoints', async () => {
    postMock.mockResolvedValueOnce({ data: { task: { task_id: 1 } } });
    getMock.mockResolvedValueOnce({ data: { job: { status: 'running' } } });

    await triggerFullRefreshV2();
    await getFullRefreshJobStatusV2(1);

    expect(postMock).toHaveBeenCalledWith('/v2/settings/full-refresh', { confirm: true });
    expect(getMock).toHaveBeenCalledWith('/v2/settings/full-refresh/jobs/1');
  });

  it('updates graph defaults through v2 endpoint', async () => {
    postMock.mockResolvedValueOnce({ data: { graph_defaults: { max_clusters: 10, min_clusters: 4 } } });
    await updateGraphDefaultsV2({ max_clusters: 10, min_clusters: 4 });
    expect(postMock).toHaveBeenCalledWith('/v2/settings/graph-defaults', {
      max_clusters: 10,
      min_clusters: 4,
    });
  });
});
