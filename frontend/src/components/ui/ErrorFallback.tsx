import { AlertCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { FallbackProps } from 'react-error-boundary';

export function ErrorFallback({ error, resetErrorBoundary }: FallbackProps) {
  const { t } = useTranslation();

  return (
    <div
      role="alert"
      className="flex min-h-[50vh] flex-col items-center justify-center gap-4 p-8 text-center"
    >
      <div className="rounded-full bg-danger-bg p-3">
        <AlertCircle aria-hidden="true" className="h-6 w-6 text-danger" />
      </div>
      <h2 className="text-lg font-semibold text-text-main dark:text-dark-text-main">
        {t('common.error')}
      </h2>
      <p className="max-w-md text-sm text-text-muted dark:text-dark-text-main/60">
        {(error instanceof Error ? error.message : null) || t('common.load_failed')}
      </p>
      <button
        type="button"
        onClick={resetErrorBoundary}
        className="header-action"
      >
        {t('common.retry')}
      </button>
    </div>
  );
}
