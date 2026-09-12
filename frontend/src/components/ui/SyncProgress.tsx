import React, { useEffect, useId, useRef } from 'react';
import { createPortal } from 'react-dom';
import { clsx } from 'clsx';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
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
      return <Check className="h-4 w-4 text-success" />;
    case 'running':
      return <Loader2 className="w-4 h-4 text-action-primary animate-spin" />;
    case 'failed':
      return <AlertCircle className="h-4 w-4 text-danger" />;
    case 'warning':
      return <AlertCircle className="h-4 w-4 text-warning" />;
    default:
      return <Circle className="w-4 h-4 text-text-muted" />;
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
  const dialogRef = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);
  const canCloseRef = useRef(canClose);
  const titleId = useId();
  const displayTitle = title || t('sync.title', 'Syncing Data');

  useEffect(() => {
    onCloseRef.current = onClose;
    canCloseRef.current = canClose;
  }, [canClose, onClose]);

  useEffect(() => {
    if (!isOpen) return;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && canCloseRef.current) onCloseRef.current?.();
      if (event.key !== 'Tab' || !dialogRef.current) return;
      const focusable = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        )
      );
      if (focusable.length === 0) {
        event.preventDefault();
        dialogRef.current.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      previouslyFocused?.focus();
    };
  }, [isOpen]);

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
      {/* Backdrop overlay */}
      <div className="absolute inset-0 bg-overlay" />

      {/* Modal */}
      <Card className="relative w-full max-w-md overflow-hidden overscroll-contain py-0">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        className="focus:outline-none"
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border-light px-5 py-4 dark:border-dark-border">
          <div className="flex items-center gap-3">
            {allCompleted && hasWarnings ? (
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-warning-bg">
                <AlertCircle className="h-5 w-5 text-warning" />
              </div>
            ) : allCompleted ? (
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-success-bg">
                <Check className="h-5 w-5 text-success" />
              </div>
            ) : hasFailed ? (
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-danger-bg">
                <AlertCircle className="h-5 w-5 text-danger" />
              </div>
            ) : (
              <div className="w-10 h-10 rounded-full bg-action-primary/10 flex items-center justify-center">
                <Loader2 className="w-5 h-5 text-action-primary animate-spin" />
              </div>
            )}
            <div>
              <h3 id={titleId} className="text-base font-semibold text-text-main dark:text-dark-text-main">
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
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={onClose}
              aria-label={t('common.close', 'Close')}
            >
              <X />
            </Button>
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
          <div
            role="progressbar"
            aria-label={t('sync.overallProgress', 'Overall Progress')}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(overallProgress)}
            className="h-2 bg-border-light rounded-full overflow-hidden dark:bg-dark-border"
          >
            <div
              className={clsx(
                'h-full rounded-full transition-[width] duration-500 motion-reduce:transition-none',
                hasFailed
                  ? 'bg-danger'
                  : hasWarnings
                  ? 'bg-warning'
                  : allCompleted
                  ? 'bg-success'
                  : 'bg-action-primary'
              )}
              style={{ width: `${overallProgress}%` }}
            />
          </div>
        </div>

        {/* Steps List */}
        <div className="px-5 py-4 space-y-3" aria-live="polite">
          {steps.map((step, index) => (
            <div
              key={step.id}
              className={clsx(
                'relative flex items-start gap-3 rounded-xl p-3 transition-colors',
                step.status === 'running' && 'bg-action-primary/5 ring-1 ring-action-primary/20',
                step.status === 'warning' && 'bg-warning-bg ring-1 ring-warning/30',
                step.status === 'failed' && 'bg-danger-bg ring-1 ring-danger/30'
              )}
            >
              {/* Step Icon */}
              <div
                className={clsx(
                  'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
                  step.status === 'completed' && 'bg-success-bg text-success',
                  step.status === 'running' && 'bg-action-primary/20 text-action-primary',
                  step.status === 'warning' && 'bg-warning-bg text-warning',
                  step.status === 'failed' && 'bg-danger-bg text-danger',
                  step.status === 'pending' && 'bg-bg-hover text-text-muted dark:bg-dark-bg-sidebar/70 dark:text-dark-text-main/60'
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
                        className="h-full bg-action-primary rounded-full transition-[width] duration-300 motion-reduce:transition-none"
                        style={{ width: `${step.progress}%` }}
                      />
                    </div>
                    <span className="mt-1 text-xs text-text-muted">
                      {step.progress}%
                    </span>
                  </div>
                )}

                {/* Error message */}
                {(step.status === 'failed' || step.status === 'warning') && step.error && (
                  <p className={clsx('mt-1 text-xs', step.status === 'failed' ? 'text-danger' : 'text-warning-foreground')}>
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
            <Button
              type="button"
              variant={allCompleted ? 'default' : 'outline'}
              className="w-full"
              onClick={onClose}
            >
              {allCompleted ? t('sync.viewResults', 'View Results') : t('common.close', 'Close')}
            </Button>
          </div>
        )}
      </div>
      </Card>
    </div>,
    document.body
  );
};

export default SyncProgress;
