import React from 'react';
import { createPortal } from 'react-dom';
import { clsx } from 'clsx';
import { useTranslation } from 'react-i18next';
import {
  Loader2,
  Check,
  Circle,
  AlertCircle,
  X,
  Star,
  Sparkles,
  Brain,
  Layers,
  Database,
} from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

export type SyncStepStatus = 'pending' | 'running' | 'completed' | 'warning' | 'failed';

export interface SyncStep {
  id: string;
  label: string;
  description?: string;
  status: SyncStepStatus;
  progress?: number; // 0-100
  error?: string;
}

interface SyncProgressProps {
  isOpen: boolean;
  onClose?: () => void;
  steps: SyncStep[];
  currentStep?: string;
  title?: string;
  canClose?: boolean;
}

// ============================================================================
// Step Icon Component
// ============================================================================

const getStepIcon = (stepId: string): React.ReactNode => {
  switch (stepId) {
    case 'stars':
      return <Star className="w-4 h-4" />;
    case 'reset':
      return <Database className="w-4 h-4" />;
    case 'summaries':
      return <Sparkles className="w-4 h-4" />;
    case 'embeddings':
      return <Brain className="w-4 h-4" />;
    case 'clustering':
      return <Layers className="w-4 h-4" />;
    default:
      return <Circle className="w-4 h-4" />;
  }
};

const getStatusIcon = (status: SyncStepStatus): React.ReactNode => {
  switch (status) {
    case 'completed':
      return <Check className="w-4 h-4 text-green-500" />;
    case 'running':
      return <Loader2 className="w-4 h-4 text-action-primary animate-spin" />;
    case 'failed':
      return <AlertCircle className="w-4 h-4 text-red-500" />;
    case 'warning':
      return <AlertCircle className="w-4 h-4 text-amber-500" />;
    default:
      return <Circle className="w-4 h-4 text-text-dim" />;
  }
};

// ============================================================================
// Component
// ============================================================================

export const SyncProgress: React.FC<SyncProgressProps> = ({
  isOpen,
  onClose,
  steps,
  title,
  canClose = false,
}) => {
  const { t } = useTranslation();
  const displayTitle = title || t('sync.title', 'Syncing Data');

  // Calculate overall progress
  const completedSteps = steps.filter(s => s.status === 'completed').length;
  const warningSteps = steps.filter(s => s.status === 'warning').length;
  const totalSteps = steps.length;
  const overallProgress = totalSteps > 0 ? ((completedSteps + warningSteps) / totalSteps) * 100 : 0;

  // Check if all done
  const allCompleted = steps.every(s => s.status === 'completed' || s.status === 'warning');
  const hasFailed = steps.some(s => s.status === 'failed');
  const hasWarnings = steps.some(s => s.status === 'warning');

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-[90] flex items-center justify-center px-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-slate-950/42 backdrop-blur-sm" />

      {/* Modal */}
      <div className="panel-surface-strong relative w-full max-w-md overflow-hidden rounded-[1.35rem] animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border-light px-5 py-4 dark:border-dark-border">
          <div className="flex items-center gap-3">
            {allCompleted && hasWarnings ? (
              <div className="w-10 h-10 rounded-full bg-amber-100 flex items-center justify-center">
                <AlertCircle className="w-5 h-5 text-amber-600" />
              </div>
            ) : allCompleted ? (
              <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center">
                <Check className="w-5 h-5 text-green-600" />
              </div>
            ) : hasFailed ? (
              <div className="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center">
                <AlertCircle className="w-5 h-5 text-red-600" />
              </div>
            ) : (
              <div className="w-10 h-10 rounded-full bg-action-primary/10 flex items-center justify-center">
                <Loader2 className="w-5 h-5 text-action-primary animate-spin" />
              </div>
            )}
            <div>
              <h3 className="text-base font-semibold text-text-main dark:text-dark-text-main">
                {allCompleted
                  ? hasWarnings
                    ? t('sync.completed_with_warnings', 'Completed With Warnings')
                    : t('sync.completed', 'Sync Completed')
                  : hasFailed
                  ? t('sync.failed', 'Sync Failed')
                  : displayTitle}
              </h3>
              <p className="text-sm text-text-muted dark:text-dark-text-main/70">
                {allCompleted
                  ? hasWarnings
                    ? t('sync.partial_done', 'Completed, but some steps need attention')
                    : t('sync.allDone', 'All tasks completed successfully')
                  : `${completedSteps}/${totalSteps} ${t('sync.steps', 'steps')}`}
              </p>
            </div>
          </div>

          {canClose && (
            <button
              onClick={onClose}
              className="rounded-xl p-2 transition-colors hover:bg-bg-hover dark:hover:bg-dark-bg-sidebar/70"
            >
              <X className="w-5 h-5 text-text-muted dark:text-dark-text-main/70" />
            </button>
          )}
        </div>

        {/* Overall Progress Bar */}
        <div className="px-5 py-3 bg-bg-sidebar/50 dark:bg-dark-bg-sidebar/60">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-text-muted dark:text-dark-text-main/70">
              {t('sync.overallProgress', 'Overall Progress')}
            </span>
            <span className="text-xs font-medium text-text-main dark:text-dark-text-main">
              {Math.round(overallProgress)}%
            </span>
          </div>
          <div className="h-2 bg-border-light rounded-full overflow-hidden dark:bg-dark-border">
            <div
              className={clsx(
                'h-full rounded-full transition-all duration-500',
                hasFailed
                  ? 'bg-red-500'
                  : hasWarnings
                  ? 'bg-amber-500'
                  : allCompleted
                  ? 'bg-green-500'
                  : 'bg-action-primary'
              )}
              style={{ width: `${overallProgress}%` }}
            />
          </div>
        </div>

        {/* Steps List */}
        <div className="px-5 py-4 space-y-3">
          {steps.map((step, index) => (
            <div
              key={step.id}
              className={clsx(
                'relative flex items-start gap-3 rounded-xl p-3 transition-colors',
                step.status === 'running' && 'bg-action-primary/5 ring-1 ring-action-primary/20',
                step.status === 'warning' && 'bg-amber-50 ring-1 ring-amber-200',
                step.status === 'failed' && 'bg-red-50 ring-1 ring-red-200'
              )}
            >
              {/* Step Icon */}
              <div
                className={clsx(
                  'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
                  step.status === 'completed' && 'bg-green-100 text-green-600',
                  step.status === 'running' && 'bg-action-primary/20 text-action-primary',
                  step.status === 'warning' && 'bg-amber-100 text-amber-600',
                  step.status === 'failed' && 'bg-red-100 text-red-600',
                  step.status === 'pending' && 'bg-bg-hover text-text-dim dark:bg-dark-bg-sidebar/70 dark:text-dark-text-main/60'
                )}
              >
                {getStepIcon(step.id)}
              </div>

              {/* Step Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span
                    className={clsx(
                      'text-sm font-medium',
                      step.status === 'pending'
                        ? 'text-text-muted dark:text-dark-text-main/70'
                        : 'text-text-main dark:text-dark-text-main'
                    )}
                  >
                    {step.label}
                  </span>
                  {getStatusIcon(step.status)}
                </div>

                {step.description && (
                  <p className="text-xs text-text-muted mt-0.5 dark:text-dark-text-main/70">{step.description}</p>
                )}

                {/* Progress bar for running step */}
                {step.status === 'running' && step.progress !== undefined && (
                  <div className="mt-2">
                    <div className="h-1.5 bg-border-light rounded-full overflow-hidden dark:bg-dark-border">
                      <div
                        className="h-full bg-action-primary rounded-full transition-all duration-300"
                        style={{ width: `${step.progress}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-text-dim mt-1 dark:text-dark-text-main/60">
                      {step.progress}%
                    </span>
                  </div>
                )}

                {/* Error message */}
                {(step.status === 'failed' || step.status === 'warning') && step.error && (
                  <p className={clsx('text-xs mt-1', step.status === 'failed' ? 'text-red-600' : 'text-amber-700')}>
                    {step.error}
                  </p>
                )}
              </div>

              {/* Connector line */}
              {index < steps.length - 1 && (
                <div className="absolute left-[2.15rem] top-[3.5rem] w-0.5 h-8 bg-border-light dark:bg-dark-border" />
              )}
            </div>
          ))}
        </div>

        {/* Footer */}
        {(allCompleted || hasFailed) && (
          <div className="border-t border-border-light bg-bg-sidebar/30 px-5 py-4 dark:border-dark-border dark:bg-dark-bg-sidebar/60">
            <button
              onClick={onClose}
              className={clsx(
                'w-full py-2.5 rounded-lg text-sm font-medium transition-colors',
                allCompleted
                  ? 'bg-green-600 hover:bg-green-700 text-white'
                  : 'bg-bg-hover hover:bg-border-light text-text-main dark:bg-dark-bg-sidebar/70 dark:hover:bg-dark-border dark:text-dark-text-main'
              )}
            >
              {allCompleted ? t('sync.viewResults', 'View Results') : t('common.close', 'Close')}
            </button>
          </div>
        )}
      </div>
    </div>,
    document.body
  );
};

export default SyncProgress;
