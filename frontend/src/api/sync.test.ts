import { beforeEach, describe, expect, it, vi } from 'vitest';

const { getMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
}));

vi.mock('./client', () => ({
  default: {
    get: getMock,
  },
}));

vi.mock('./v2/sync', () => ({
  startSyncPipelineV2: vi.fn(),
  getPipelineStatusV2: vi.fn(),
}));

import {
  getAllSyncStatus,
  getSyncJobStatus,
  getSyncStatus,
  startStarSync,
} from './sync';
import {
  getPipelineStatusV2,
  startSyncPipelineV2,
} from './v2/sync';

describe('api/sync compatibility wrappers', () => {
  beforeEach(() => {
    getMock.mockReset();
    vi.mocked(startSyncPipelineV2).mockReset();
    vi.mocked(getPipelineStatusV2).mockReset();
  });

  it('starts star sync through v2 pipeline endpoint', async () => {
    vi.mocked(startSyncPipelineV2).mockResolvedValue({
      pipeline_run_id: 7,
      status: 'pending',
      phase: 'pending',
      message: 'Pipeline started in background',
    });

    const payload = await startStarSync('incremental');
    expect(payload.task_id).toBe(7);
    expect(startSyncPipelineV2).toHaveBeenCalledWith({ mode: 'incremental' });
  });

  it('maps v2 pipeline status to legacy sync status', async () => {
    vi.mocked(getPipelineStatusV2).mockResolvedValue({
      pipeline_run_id: 8,
      user_id: 1,
      status: 'running',
      phase: 'embedding',
      last_error: null,
      created_at: null,
      started_at: null,
      completed_at: null,
    });

    const syncStatus = await getSyncStatus(8);
    const syncJobStatus = await getSyncJobStatus(8);

    expect(syncStatus.task_id).toBe(8);
    expect(syncJobStatus.phase).toBe('embedding');
  });

  it('keeps legacy list-status fallback endpoint for compatibility', async () => {
    getMock.mockResolvedValue({ data: [{ task_id: 1, status: 'running' }] });
    const payload = await getAllSyncStatus();
    expect(getMock).toHaveBeenCalledWith('/sync/status');
    expect(payload).toHaveLength(1);
  });
});
