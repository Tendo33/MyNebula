import { useTranslation } from 'react-i18next';
import type { FallbackProps } from 'react-error-boundary';

export function ErrorFallback({ error, resetErrorBoundary }: FallbackProps) {
  const { t } = useTranslation();

  return (
    <div
      role="alert"
      className="flex min-h-[50vh] flex-col items-center justify-center gap-4 p-8 text-center"
    >
      <div className="rounded-full bg-red-100 p-3 dark:bg-red-900/30">
        <svg className="h-6 w-6 text-red-600 dark:text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
      <h2 className="text-lg font-semibold text-text-main dark:text-dark-text-main">
        {t('common.error', 'Something went wrong')}
      </h2>
      <p className="max-w-md text-sm text-text-muted dark:text-dark-text-main/60">
        {error?.message || t('common.load_failed', 'Failed to load data')}
      </p>
      <button
        onClick={resetErrorBoundary}
        className="rounded-md bg-action-primary px-4 py-2 text-sm font-medium text-white hover:bg-action-hover transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/50"
      >
        {t('common.retry', 'Retry')}
      </button>
    </div>
  );
}
