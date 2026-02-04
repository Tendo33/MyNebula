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
} from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

export type SyncStepStatus = 'pending' | 'running' | 'completed' | 'failed';

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
  title = 'Syncing Data',
  canClose = false,
}) => {
  const { t } = useTranslation();

  // Calculate overall progress
  const completedSteps = steps.filter(s => s.status === 'completed').length;
  const totalSteps = steps.length;
  const overallProgress = totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0;

  // Check if all done
  const allCompleted = steps.every(s => s.status === 'completed');
  const hasFailed = steps.some(s => s.status === 'failed');

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-[90] flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" />

      {/* Modal */}
      <div className="relative w-full max-w-md bg-white rounded-xl shadow-2xl border border-border-light overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border-light">
          <div className="flex items-center gap-3">
            {allCompleted ? (
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
              <h3 className="text-base font-semibold text-text-main">
                {allCompleted
                  ? t('sync.completed', 'Sync Completed')
                  : hasFailed
                  ? t('sync.failed', 'Sync Failed')
                  : title}
              </h3>
              <p className="text-sm text-text-muted">
                {allCompleted
                  ? t('sync.allDone', 'All tasks completed successfully')
                  : `${completedSteps}/${totalSteps} ${t('sync.steps', 'steps')}`}
              </p>
            </div>
          </div>

          {canClose && (
            <button
              onClick={onClose}
              className="p-2 hover:bg-bg-hover rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-text-muted" />
            </button>
          )}
        </div>

        {/* Overall Progress Bar */}
        <div className="px-5 py-3 bg-bg-sidebar/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-text-muted">
              {t('sync.overallProgress', 'Overall Progress')}
            </span>
            <span className="text-xs font-medium text-text-main">
              {Math.round(overallProgress)}%
            </span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={clsx(
                'h-full rounded-full transition-all duration-500',
                hasFailed ? 'bg-red-500' : allCompleted ? 'bg-green-500' : 'bg-action-primary'
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
                'flex items-start gap-3 p-3 rounded-lg transition-colors',
                step.status === 'running' && 'bg-action-primary/5 ring-1 ring-action-primary/20',
                step.status === 'failed' && 'bg-red-50 ring-1 ring-red-200'
              )}
            >
              {/* Step Icon */}
              <div
                className={clsx(
                  'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
                  step.status === 'completed' && 'bg-green-100 text-green-600',
                  step.status === 'running' && 'bg-action-primary/20 text-action-primary',
                  step.status === 'failed' && 'bg-red-100 text-red-600',
                  step.status === 'pending' && 'bg-gray-100 text-text-dim'
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
                      step.status === 'pending' ? 'text-text-muted' : 'text-text-main'
                    )}
                  >
                    {step.label}
                  </span>
                  {getStatusIcon(step.status)}
                </div>

                {step.description && (
                  <p className="text-xs text-text-muted mt-0.5">{step.description}</p>
                )}

                {/* Progress bar for running step */}
                {step.status === 'running' && step.progress !== undefined && (
                  <div className="mt-2">
                    <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-action-primary rounded-full transition-all duration-300"
                        style={{ width: `${step.progress}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-text-dim mt-1">
                      {step.progress}%
                    </span>
                  </div>
                )}

                {/* Error message */}
                {step.status === 'failed' && step.error && (
                  <p className="text-xs text-red-600 mt-1">{step.error}</p>
                )}
              </div>

              {/* Connector line */}
              {index < steps.length - 1 && (
                <div className="absolute left-[2.15rem] top-[3.5rem] w-0.5 h-8 bg-border-light" />
              )}
            </div>
          ))}
        </div>

        {/* Footer */}
        {(allCompleted || hasFailed) && (
          <div className="px-5 py-4 border-t border-border-light bg-bg-sidebar/30">
            <button
              onClick={onClose}
              className={clsx(
                'w-full py-2.5 rounded-lg text-sm font-medium transition-colors',
                allCompleted
                  ? 'bg-green-600 hover:bg-green-700 text-white'
                  : 'bg-gray-100 hover:bg-gray-200 text-text-main'
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
