export interface TaskCompletionResult {
  success: boolean;
  error: string | null;
  cancelled: boolean;
}

const createCancelledResult = (): TaskCompletionResult => ({
  success: false,
  error: null,
  cancelled: true,
});

const isAbortError = (error: unknown): boolean =>
  error instanceof DOMException && error.name === 'AbortError';

export const waitForPollDelay = (delayMs: number, signal: AbortSignal): Promise<void> =>
  new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('Polling aborted', 'AbortError'));
      return;
    }

    const timer = window.setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, delayMs);

    const onAbort = () => {
      window.clearTimeout(timer);
      signal.removeEventListener('abort', onAbort);
      reject(new DOMException('Polling aborted', 'AbortError'));
    };

    signal.addEventListener('abort', onAbort, { once: true });
  });

interface PollUntilCompleteOptions<T> {
  signal: AbortSignal;
  poll: () => Promise<T>;
  onProgress?: (value: T) => void;
  isSuccess: (value: T) => boolean;
  isFailure: (value: T) => boolean;
  getFailureError: (value: T) => string | null;
  getPollError: (error: unknown) => string;
  intervalMs?: number;
}

export const pollUntilComplete = async <T>({
  signal,
  poll,
  onProgress,
  isSuccess,
  isFailure,
  getFailureError,
  getPollError,
  intervalMs = 2000,
}: PollUntilCompleteOptions<T>): Promise<TaskCompletionResult> => {
  while (!signal.aborted) {
    try {
      const value = await poll();
      onProgress?.(value);

      if (isSuccess(value)) {
        return { success: true, error: null, cancelled: false };
      }

      if (isFailure(value)) {
        return {
          success: false,
          error: getFailureError(value),
          cancelled: false,
        };
      }
    } catch (error) {
      if (signal.aborted || isAbortError(error)) {
        return createCancelledResult();
      }

      return {
        success: false,
        error: getPollError(error),
        cancelled: false,
      };
    }

    try {
      await waitForPollDelay(intervalMs, signal);
    } catch (error) {
      if (signal.aborted || isAbortError(error)) {
        return createCancelledResult();
      }

      return {
        success: false,
        error: getPollError(error),
        cancelled: false,
      };
    }
  }

  return createCancelledResult();
};
