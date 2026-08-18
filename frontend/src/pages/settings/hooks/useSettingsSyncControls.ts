import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { TFunction } from 'i18next';

import type { SyncStepStatus } from '../../../components/ui/SyncProgress';
import {
  getPipelineStatusV2,
  startReclusterV2,
  startSyncPipelineV2,
  type PipelineStatusResponse,
} from '../../../api/v2/sync';
import {
  getFullRefreshJobStatusV2,
  triggerFullRefreshV2,
  type FullRefreshJobStatus,
} from '../../../api/v2/settings';
import { logClientError } from '../../../utils/debug';
import {
  buildTranslatedSteps,
  createFullRefreshSteps,
  createPipelineSyncSteps,
  finalizeCompletedSteps,
  mapFullRefreshProgressSteps,
  mapPipelineProgressSteps,
  normalizeFullRefreshPhase,
  normalizePipelinePhase,
} from '../progress';
import { pollUntilComplete, type TaskCompletionResult } from '../polling';

export const isPartialFailureResult = (result: TaskCompletionResult): boolean =>
  !result.success &&
  !result.cancelled &&
  typeof result.error === 'string' &&
  result.error.includes('partial failure');

/**
 * Sync, re-cluster, and full-refresh orchestration plus the progress polling
 * lifecycle.
 *
 * Every long operation runs under one `AbortController` held in
 * `activePollControllerRef`: starting a new operation aborts the previous poll,
 * and unmount aborts whatever is in flight. Without that, a poll outlives the
 * page and writes into unmounted state.
 */
export const useSettingsSyncControls = ({
  maxClusters,
  minClusters,
  setSyncing,
  setSyncStep,
  refreshData,
  reloadSettings,
  setError,
  setWarning,
  t,
}: {
  maxClusters: number;
  minClusters: number;
  setSyncing: (syncing: boolean) => void;
  setSyncStep: (step: string) => void;
  refreshData: () => Promise<void>;
  reloadSettings: () => Promise<void>;
  setError: (message: string | null) => void;
  setWarning: (message: string | null) => void;
  t: TFunction;
}) => {
  const [refreshLoading, setRefreshLoading] = useState(false);
  const [reclusterLoading, setReclusterLoading] = useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [progressTitle, setProgressTitle] = useState('');
  const [showSyncProgress, setShowSyncProgress] = useState(false);
  const [syncSteps, setSyncSteps] = useState<
    { id: string; status: SyncStepStatus; progress?: number; error?: string }[]
  >(createPipelineSyncSteps);

  const activePollControllerRef = useRef<AbortController | null>(null);
  const latestFullRefreshJobRef = useRef<FullRefreshJobStatus | null>(null);

  const translatedSteps = useMemo(() => buildTranslatedSteps(syncSteps, t), [syncSteps, t]);

  const resetSteps = useCallback(() => {
    setSyncSteps(createPipelineSyncSteps());
  }, []);

  const resetFullRefreshStepState = useCallback(() => {
    setSyncSteps(createFullRefreshSteps());
  }, []);

  const resetSyncUi = useCallback(() => {
    activePollControllerRef.current?.abort();
    activePollControllerRef.current = null;
    latestFullRefreshJobRef.current = null;
    setSyncing(false);
    setSyncStep('');
    setShowSyncProgress(false);
    resetSteps();
  }, [resetSteps, setSyncStep, setSyncing]);

  useEffect(() => {
    setProgressTitle(t('sync.title', 'Syncing Data'));
  }, [t]);

  const beginPollingOperation = useCallback(() => {
    activePollControllerRef.current?.abort();
    const controller = new AbortController();
    activePollControllerRef.current = controller;
    return controller;
  }, []);

  const finishPollingOperation = useCallback((controller: AbortController) => {
    if (activePollControllerRef.current === controller) {
      activePollControllerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      activePollControllerRef.current?.abort();
      activePollControllerRef.current = null;
    };
  }, []);

  const waitForPipelineComplete = async (
    runId: number,
    signal: AbortSignal,
    onProgress?: (pipeline: PipelineStatusResponse) => void
  ): Promise<TaskCompletionResult> => {
    return pollUntilComplete({
      signal,
      poll: () => getPipelineStatusV2(runId),
      onProgress,
      isSuccess: (pipeline) => pipeline.status === 'completed',
      isFailure: (pipeline) =>
        pipeline.status === 'failed' ||
        pipeline.status === 'partial_failed' ||
        pipeline.status === 'interrupted',
      getFailureError: (pipeline) => pipeline.last_error || null,
      getPollError: (err) => (err instanceof Error ? err.message : 'poll_pipeline_status_failed'),
    });
  };

  const waitForFullRefreshJobComplete = async (
    taskId: number,
    signal: AbortSignal,
    onProgress?: (job: FullRefreshJobStatus) => void
  ): Promise<TaskCompletionResult> => {
    return pollUntilComplete({
      signal,
      poll: async () => (await getFullRefreshJobStatusV2(taskId)).job,
      onProgress,
      isSuccess: (job) => job.status === 'completed' || job.status === 'partial_failed',
      isFailure: (job) => job.status === 'failed' || job.status === 'interrupted',
      getFailureError: (job) => job.last_error,
      getPollError: (err) => (err instanceof Error ? err.message : 'poll_job_status_failed'),
    });
  };

  const updatePipelineProgress = (pipeline: PipelineStatusResponse) => {
    const phaseOrder = ['stars', 'embeddings', 'clustering', 'snapshot'] as const;
    const normalizedPhase = normalizePipelinePhase(pipeline.phase);
    const phaseIndex =
      normalizedPhase !== null ? phaseOrder.findIndex((phase) => phase === normalizedPhase) : -1;

    if (phaseIndex >= 0) {
      setSyncStep(t(`sync.step_${phaseOrder[phaseIndex]}_label`, phaseOrder[phaseIndex]));
    }

    setSyncSteps((previous) => mapPipelineProgressSteps(previous, pipeline));
  };

  const updateFullRefreshProgress = (job: FullRefreshJobStatus) => {
    latestFullRefreshJobRef.current = job;
    const phaseOrder = ['reset', 'stars', 'embeddings', 'clustering', 'snapshot'] as const;
    const normalizedPhase = normalizeFullRefreshPhase(job.phase);
    const stepProgressSpan = 100 / phaseOrder.length;
    const phaseIndex =
      normalizedPhase !== null ? phaseOrder.findIndex((phase) => phase === normalizedPhase) : -1;
    const inferredIndex = Math.min(
      phaseOrder.length - 1,
      Math.max(0, Math.floor(Math.max(0, Math.min(99.999, job.progress_percent)) / stepProgressSpan))
    );
    const currentIndex =
      job.status === 'running' || job.status === 'pending'
        ? Math.max(phaseIndex, inferredIndex)
        : phaseIndex;

    if (currentIndex >= 0) {
      setSyncStep(t(`sync.step_${phaseOrder[currentIndex]}_label`, phaseOrder[currentIndex]));
    }

    setSyncSteps((previous) => mapFullRefreshProgressSteps(previous, job, t));
  };

  const handleSyncStars = async () => {
    try {
      setError(null);
      setSyncing(true);
      setShowSyncProgress(true);
      setProgressTitle(t('sync.title', 'Syncing Data'));
      resetSteps();

      const started = await startSyncPipelineV2({
        mode: 'incremental',
        use_llm: true,
        max_clusters: maxClusters,
        min_clusters: minClusters,
      });
      const controller = beginPollingOperation();
      const pipelineResult = await waitForPipelineComplete(
        started.pipeline_run_id,
        controller.signal,
        updatePipelineProgress
      );
      finishPollingOperation(controller);
      if (pipelineResult.cancelled) {
        return;
      }
      if (isPartialFailureResult(pipelineResult)) {
        await refreshData();
        await reloadSettings();
        setWarning(
          t(
            'sync.partial_failed_warning',
            'Sync completed with warnings. Check the latest run details before retrying.'
          )
        );
        return;
      }
      if (!pipelineResult.success) {
        throw new Error(pipelineResult.error || t('errors.sync_failed'));
      }

      setSyncSteps((prev) =>
        prev.map((step) =>
          step.status === 'warning'
            ? { ...step, progress: 100 }
            : { ...step, status: 'completed', progress: 100, error: undefined }
        )
      );

      await refreshData();
      await reloadSettings();
    } catch (err) {
      setError(t('errors.sync_failed'));
      logClientError('Sync failed:', err);
    } finally {
      activePollControllerRef.current?.abort();
      setSyncing(false);
      setSyncStep('');
    }
  };

  const handleRecluster = async () => {
    setWarning(null);
    setReclusterLoading(true);
    setError(null);
    setSyncing(true);
    setShowSyncProgress(true);
    setProgressTitle(t('graph.recluster'));
    resetSteps();

    try {
      const started = await startReclusterV2({
        max_clusters: maxClusters,
        min_clusters: minClusters,
      });
      const controller = beginPollingOperation();
      const pipelineResult = await waitForPipelineComplete(
        started.pipeline_run_id,
        controller.signal,
        updatePipelineProgress
      );
      finishPollingOperation(controller);
      if (pipelineResult.cancelled) {
        return;
      }
      if (isPartialFailureResult(pipelineResult)) {
        await refreshData();
        await reloadSettings();
        setWarning(
          t(
            'graph.recluster_partial_failed',
            'Re-cluster completed with warnings. Review the latest pipeline details.'
          )
        );
        return;
      }
      if (!pipelineResult.success) {
        throw new Error(pipelineResult.error || t('errors.clustering_failed'));
      }

      await refreshData();
      await reloadSettings();
    } catch (err) {
      setError(t('errors.clustering_failed'));
      logClientError('Re-cluster failed:', err);
    } finally {
      activePollControllerRef.current?.abort();
      setSyncing(false);
      setSyncStep('');
      setReclusterLoading(false);
    }
  };

  const handleFullRefresh = async () => {
    setShowConfirmDialog(false);
    activePollControllerRef.current?.abort();
    latestFullRefreshJobRef.current = null;
    setRefreshLoading(true);
    setError(null);
    setWarning(null);
    setSyncing(true);
    setShowSyncProgress(true);
    setProgressTitle(t('settings.full_refresh'));
    resetFullRefreshStepState();

    try {
      const started = await triggerFullRefreshV2();
      const controller = beginPollingOperation();
      const fullRefreshStatus = await waitForFullRefreshJobComplete(
        started.task.task_id,
        controller.signal,
        updateFullRefreshProgress
      );
      finishPollingOperation(controller);
      if (fullRefreshStatus.cancelled) {
        return;
      }
      if (!fullRefreshStatus.success) {
        throw new Error(fullRefreshStatus.error || t('settings.full_refresh_error'));
      }

      const latestJob = latestFullRefreshJobRef.current as FullRefreshJobStatus | null;
      const partialFailures = (latestJob?.error_details?.partial_failures ?? []) as Array<{
        phase: string;
        task_id: number;
        failed_items: number;
      }>;
      if (latestJob?.status === 'partial_failed') {
        const failedItems = partialFailures.reduce(
          (sum: number, entry) => sum + (entry.failed_items ?? 0),
          0
        );
        const warningMessage = t(
          'settings.full_refresh_partial_failed',
          `Full refresh completed with warnings (${partialFailures.length} stages, ${failedItems} failed items).`
        );

        setSyncSteps((prev) => finalizeCompletedSteps(prev, partialFailures, t));

        await refreshData();
        await reloadSettings();
        setWarning(warningMessage);
        return;
      }

      setSyncSteps((prev) => finalizeCompletedSteps(prev, partialFailures, t));

      await refreshData();
      await reloadSettings();
    } catch (err) {
      setError(t('settings.full_refresh_error'));
      logClientError('Failed to trigger full refresh:', err);
    } finally {
      setRefreshLoading(false);
      setSyncing(false);
      setSyncStep('');
    }
  };

  const handleSyncProgressClose = (syncing: boolean) => {
    activePollControllerRef.current?.abort();
    activePollControllerRef.current = null;
    setShowSyncProgress(false);
    if (!syncing) {
      resetSteps();
    }
  };

  return {
    refreshLoading,
    reclusterLoading,
    showConfirmDialog,
    setShowConfirmDialog,
    progressTitle,
    showSyncProgress,
    translatedSteps,
    resetSyncUi,
    handleSyncStars,
    handleRecluster,
    handleFullRefresh,
    handleSyncProgressClose,
  };
};
