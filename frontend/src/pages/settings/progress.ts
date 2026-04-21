import type { TFunction } from 'i18next';

import type { FullRefreshJobStatus } from '../../api/v2/settings';
import type { PipelineStatusResponse } from '../../api/v2/sync';
import type { SyncStepStatus } from '../../components/ui/SyncProgress';

export interface ProgressStepState {
  id: string;
  status: SyncStepStatus;
  progress?: number;
  error?: string;
}

export const createPipelineSyncSteps = (): ProgressStepState[] => ([
  { id: 'stars', status: 'pending', progress: 0 },
  { id: 'embeddings', status: 'pending', progress: 0 },
  { id: 'clustering', status: 'pending', progress: 0 },
  { id: 'snapshot', status: 'pending', progress: 0 },
]);

export const createFullRefreshSteps = (): ProgressStepState[] => ([
  { id: 'reset', status: 'pending', progress: 0 },
  { id: 'stars', status: 'pending', progress: 0 },
  { id: 'embeddings', status: 'pending', progress: 0 },
  { id: 'clustering', status: 'pending', progress: 0 },
  { id: 'snapshot', status: 'pending', progress: 0 },
]);

export const buildTranslatedSteps = (
  syncSteps: ProgressStepState[],
  t: TFunction
) =>
  syncSteps.map((step) => ({
    ...step,
    label: t(
      `sync.step_${step.id}_label`,
      step.id === 'reset' ? t('settings.confirm_step_fetch') : step.id
    ),
    description: t(
      `sync.step_${step.id}_desc`,
      step.id === 'reset' ? t('settings.full_refresh_desc') : ''
    ),
  }));

export const normalizePipelinePhase = (
  phase: string
): 'stars' | 'embeddings' | 'clustering' | 'snapshot' | null => {
  switch (phase) {
    case 'stars':
    case 'star':
      return 'stars';
    case 'embedding':
    case 'embeddings':
      return 'embeddings';
    case 'cluster':
    case 'clustering':
      return 'clustering';
    case 'snapshot':
      return 'snapshot';
    default:
      return null;
  }
};

export const normalizeFullRefreshPhase = (
  phase: string
): 'reset' | 'stars' | 'embeddings' | 'clustering' | 'snapshot' | null => {
  switch (phase) {
    case 'reset':
      return 'reset';
    case 'stars':
    case 'star':
      return 'stars';
    case 'embedding':
    case 'embeddings':
    case 'summary':
    case 'summaries':
      return 'embeddings';
    case 'cluster':
    case 'clustering':
      return 'clustering';
    case 'snapshot':
    case 'complete':
    case 'completed':
      return 'snapshot';
    case 'full_refresh':
    default:
      return null;
  }
};

export const getPartialFailureMap = (
  job: FullRefreshJobStatus
): Map<string, { failed_items: number; task_id: number }> => {
  const partialFailures = job.error_details?.partial_failures ?? [];
  return new Map(
    partialFailures.map((entry) => [
      entry.phase,
      { failed_items: entry.failed_items, task_id: entry.task_id },
    ])
  );
};

export const mapPipelineProgressSteps = (
  previous: ProgressStepState[],
  pipeline: PipelineStatusResponse
): ProgressStepState[] => {
  const phaseOrder = ['stars', 'embeddings', 'clustering', 'snapshot'] as const;
  const normalizedPhase = normalizePipelinePhase(pipeline.phase);
  const phaseIndex =
    normalizedPhase !== null ? phaseOrder.findIndex((phase) => phase === normalizedPhase) : -1;

  return previous.map((step, index) => {
    if (pipeline.status === 'completed') {
      return { ...step, status: 'completed', progress: 100, error: undefined };
    }

    if (pipeline.status === 'failed' || pipeline.status === 'partial_failed') {
      if (phaseIndex >= 0 && index < phaseIndex) {
        return { ...step, status: 'completed', progress: 100, error: undefined };
      }
      if (
        step.id === normalizedPhase ||
        (normalizedPhase === null && index === previous.length - 1)
      ) {
        return {
          ...step,
          status: 'failed',
          error: pipeline.last_error || undefined,
        };
      }
      return { ...step, status: 'pending', progress: 0, error: undefined };
    }

    if (phaseIndex >= 0 && index < phaseIndex) {
      return { ...step, status: 'completed', progress: 100, error: undefined };
    }
    if (phaseIndex >= 0 && index === phaseIndex) {
      return {
        ...step,
        status: 'running',
        progress: 40,
        error: undefined,
      };
    }
    return { ...step, status: 'pending', progress: 0, error: undefined };
  });
};

export const mapFullRefreshProgressSteps = (
  previous: ProgressStepState[],
  job: FullRefreshJobStatus,
  t: TFunction
): ProgressStepState[] => {
  const phaseOrder = ['reset', 'stars', 'embeddings', 'clustering', 'snapshot'] as const;
  const normalizedPhase = normalizeFullRefreshPhase(job.phase);
  const partialFailureMap = getPartialFailureMap(job);
  const stepProgressSpan = 100 / phaseOrder.length;
  const phaseIndex =
    normalizedPhase !== null ? phaseOrder.findIndex((phase) => phase === normalizedPhase) : -1;
  const inferredIndex = Math.min(
    phaseOrder.length - 1,
    Math.max(
      0,
      Math.floor(Math.max(0, Math.min(99.999, job.progress_percent)) / stepProgressSpan)
    )
  );
  const currentIndex =
    job.status === 'running' || job.status === 'pending'
      ? Math.max(phaseIndex, inferredIndex)
      : phaseIndex;

  return previous.map((step, index) => {
    if (job.status === 'completed' || job.status === 'partial_failed') {
      const partialFailure = partialFailureMap.get(step.id);
      if (partialFailure) {
        return {
          ...step,
          status: 'warning',
          progress: 100,
          error: t('settings.full_refresh_partial_phase_detail', {
            count: partialFailure.failed_items,
            defaultValue: `${partialFailure.failed_items} items failed in this stage.`,
          }),
        };
      }
      return { ...step, status: 'completed', progress: 100, error: undefined };
    }

    if (job.status === 'failed') {
      if (currentIndex >= 0 && index < currentIndex) {
        return { ...step, status: 'completed', progress: 100, error: undefined };
      }
      if (
        step.id === normalizedPhase ||
        (normalizedPhase === null && currentIndex >= 0 && step.id === phaseOrder[currentIndex]) ||
        (currentIndex < 0 && index === previous.length - 1)
      ) {
        return {
          ...step,
          status: 'failed',
          error: job.last_error || undefined,
        };
      }
      return { ...step, status: 'pending', progress: 0, error: undefined };
    }

    if (currentIndex >= 0 && index < currentIndex) {
      return { ...step, status: 'completed', progress: 100, error: undefined };
    }
    if (currentIndex >= 0 && index === currentIndex) {
      const perStepProgress = Math.max(
        0,
        Math.min(
          100,
          ((job.progress_percent - currentIndex * stepProgressSpan) / stepProgressSpan) * 100
        )
      );
      return {
        ...step,
        status: 'running',
        progress: Math.round(perStepProgress),
        error: undefined,
      };
    }
    return { ...step, status: 'pending', progress: 0, error: undefined };
  });
};

export const finalizeCompletedSteps = (
  previous: ProgressStepState[],
  partialFailures: Array<{ phase: string; failed_items: number }>,
  t: TFunction
): ProgressStepState[] => {
  const partialFailureMap = new Map(
    partialFailures.map((entry) => [normalizeFullRefreshPhase(entry.phase), entry.failed_items])
  );
  return previous.map((step) => {
    const failedItems = partialFailureMap.get(
      step.id as 'reset' | 'stars' | 'embeddings' | 'clustering' | 'snapshot' | null
    );
    if (failedItems) {
      return {
        ...step,
        status: 'warning',
        progress: 100,
        error: t('settings.full_refresh_partial_phase_detail', {
          count: failedItems,
          defaultValue: `${failedItems} items failed in this stage.`,
        }),
      };
    }
    return { ...step, status: 'completed', progress: 100, error: undefined };
  });
};
