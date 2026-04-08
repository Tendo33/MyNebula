import { beforeEach, describe, expect, it, vi } from 'vitest';

import { pollUntilComplete, waitForPollDelay } from '../settings/polling';

describe('settings polling', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it('stops waiting when aborted during delay', async () => {
    const controller = new AbortController();
    const promise = waitForPollDelay(2000, controller.signal);
    const expectation = expect(promise).rejects.toThrow(/aborted/i);

    controller.abort();
    await vi.runAllTimersAsync();

    await expectation;
  });

  it('returns cancelled when signal is aborted between poll iterations', async () => {
    const controller = new AbortController();
    const poll = vi
      .fn<() => Promise<{ status: string; last_error: string | null }>>()
      .mockResolvedValue({ status: 'running', last_error: null });

    const resultPromise = pollUntilComplete({
      signal: controller.signal,
      poll,
      isSuccess: (value) => value.status === 'completed',
      isFailure: (value) => value.status === 'failed',
      getFailureError: (value) => value.last_error,
      getPollError: () => 'poll_failed',
      intervalMs: 2000,
    });

    await Promise.resolve();
    controller.abort();
    await vi.runAllTimersAsync();

    await expect(resultPromise).resolves.toEqual({
      success: false,
      error: null,
      cancelled: true,
    });
    expect(poll).toHaveBeenCalledTimes(1);
  });

  it('allows a replacement controller to cancel an earlier polling loop', async () => {
    const firstController = new AbortController();
    const secondController = new AbortController();
    const firstPoll = vi
      .fn<() => Promise<{ status: string; last_error: string | null }>>()
      .mockResolvedValue({ status: 'running', last_error: null });
    const secondPoll = vi
      .fn<() => Promise<{ status: string; last_error: string | null }>>()
      .mockResolvedValueOnce({ status: 'completed', last_error: null });

    const firstPromise = pollUntilComplete({
      signal: firstController.signal,
      poll: firstPoll,
      isSuccess: (value) => value.status === 'completed',
      isFailure: (value) => value.status === 'failed',
      getFailureError: (value) => value.last_error,
      getPollError: () => 'poll_failed',
      intervalMs: 2000,
    });

    await Promise.resolve();
    firstController.abort();

    const secondPromise = pollUntilComplete({
      signal: secondController.signal,
      poll: secondPoll,
      isSuccess: (value) => value.status === 'completed',
      isFailure: (value) => value.status === 'failed',
      getFailureError: (value) => value.last_error,
      getPollError: () => 'poll_failed',
      intervalMs: 2000,
    });

    await vi.runAllTimersAsync();

    await expect(firstPromise).resolves.toEqual({
      success: false,
      error: null,
      cancelled: true,
    });
    await expect(secondPromise).resolves.toEqual({
      success: true,
      error: null,
      cancelled: false,
    });
  });
});
