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

import { getPipelineStatusV2, startReclusterV2, startSyncPipelineV2 } from './sync';

describe('api/v2/sync client', () => {
  beforeEach(() => {
    getMock.mockReset();
    postMock.mockReset();
  });

  it('starts regular sync pipeline', async () => {
    postMock.mockResolvedValueOnce({ data: { pipeline_run_id: 1 } });
    await startSyncPipelineV2({
      mode: 'incremental',
      use_llm: true,
      max_clusters: 8,
      min_clusters: 3,
    });
    expect(postMock).toHaveBeenCalledWith('/v2/sync/start', null, {
      params: {
        mode: 'incremental',
        use_llm: true,
        max_clusters: 8,
        min_clusters: 3,
      },
    });
  });

  it('starts full recluster pipeline', async () => {
    postMock.mockResolvedValueOnce({ data: { pipeline_run_id: 2 } });
    await startReclusterV2({ max_clusters: 12, min_clusters: 5 });
    expect(postMock).toHaveBeenCalledWith('/v2/sync/recluster', null, {
      params: {
        max_clusters: 12,
        min_clusters: 5,
      },
    });
  });

  it('loads pipeline status', async () => {
    getMock.mockResolvedValueOnce({ data: { pipeline_run_id: 3 } });
    await getPipelineStatusV2(3);
    expect(getMock).toHaveBeenCalledWith('/v2/sync/jobs/3');
  });
});
