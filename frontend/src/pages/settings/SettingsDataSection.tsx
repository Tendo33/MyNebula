import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import { AlertTriangle, Database, Loader2, RefreshCw } from 'lucide-react';
import type { SyncInfoResponse } from '../../api/v2/settings';

interface SettingsDataSectionProps {
  syncInfo: SyncInfoResponse | null;
  refreshLoading: boolean;
  syncing: boolean;
  reclusterLoading: boolean;
  showConfirmDialog: boolean;
  onShowConfirm: () => void;
  onHideConfirm: () => void;
  onConfirmRefresh: () => void;
}

export const SettingsDataSection = ({
  syncInfo,
  refreshLoading,
  syncing,
  reclusterLoading,
  showConfirmDialog,
  onShowConfirm,
  onHideConfirm,
  onConfirmRefresh,
}: SettingsDataSectionProps) => {
  const { t } = useTranslation();
  const dialogRef = useRef<HTMLDivElement>(null);
  const cancelButtonRef = useRef<HTMLButtonElement>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!showConfirmDialog) return;

    previouslyFocusedRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const focusTimer = window.setTimeout(() => {
      cancelButtonRef.current?.focus();
    }, 0);

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Tab' || !dialogRef.current) return;
      const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      if (focusable.length === 0) return;
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

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.clearTimeout(focusTimer);
      window.removeEventListener('keydown', handleKeyDown);
      previouslyFocusedRef.current?.focus();
    };
  }, [showConfirmDialog]);

  return (
    <>
      <section>
        <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-4 px-2 select-none">
          {t('settings.data_management')}
        </h2>
        <div className="space-y-2">
          <div className="p-3">
            <div className="flex items-center gap-2 mb-3">
              <Database className="w-4 h-4 text-text-muted" />
              <label className="text-sm font-medium text-text-main">{t('settings.repo_stats')}</label>
            </div>
            <div className="grid grid-cols-2 gap-4 bg-bg-sidebar/50 rounded-md p-4">
              <div>
                <div className="text-2xl font-semibold text-text-main">{syncInfo?.total_repos ?? '-'}</div>
                <div className="text-xs text-text-muted">{t('settings.total_repos')}</div>
              </div>
              <div>
                <div className="text-2xl font-semibold text-text-main">{syncInfo?.synced_repos ?? '-'}</div>
                <div className="text-xs text-text-muted">{t('settings.synced')}</div>
              </div>
              <div>
                <div className="text-2xl font-semibold text-text-main">{syncInfo?.embedded_repos ?? '-'}</div>
                <div className="text-xs text-text-muted">{t('settings.vectorized')}</div>
              </div>
              <div>
                <div className="text-2xl font-semibold text-text-main">{syncInfo?.summarized_repos ?? '-'}</div>
                <div className="text-xs text-text-muted">{t('settings.summarized')}</div>
              </div>
            </div>
            {syncInfo?.last_sync_at && (
              <div className="mt-2 text-xs text-text-muted">
                {t('settings.last_run')}: {new Date(syncInfo.last_sync_at).toLocaleString()}
              </div>
            )}
          </div>

          <div className="p-3">
            <div className="flex items-center gap-2 mb-3">
              <RefreshCw className="w-4 h-4 text-text-muted" />
              <label className="text-sm font-medium text-text-main">{t('settings.full_refresh')}</label>
            </div>
            <p className="text-xs text-text-muted mb-3">{t('settings.full_refresh_desc')}</p>
            <button
              onClick={onShowConfirm}
              disabled={refreshLoading || syncing || reclusterLoading}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30',
                'bg-red-50 border border-red-200 text-red-700 hover:bg-red-100',
                (refreshLoading || syncing || reclusterLoading) && 'opacity-50 cursor-not-allowed'
              )}
            >
              {refreshLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  {t('settings.refreshing')}
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4" />
                  {t('settings.execute_full_refresh')}
                </>
              )}
            </button>
          </div>
        </div>
      </section>

      {showConfirmDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div
            ref={dialogRef}
            className="bg-bg-main rounded-lg shadow-xl max-w-md w-full mx-4 p-6 dark:bg-dark-bg-main"
            role="dialog"
            aria-modal="true"
            aria-labelledby="full-refresh-title"
            aria-describedby="full-refresh-desc"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-red-100 rounded-full">
                <AlertTriangle className="w-6 h-6 text-red-600" />
              </div>
              <h3 id="full-refresh-title" className="text-lg font-semibold text-text-main dark:text-dark-text-main">
                {t('settings.confirm_full_refresh_title')}
              </h3>
            </div>
            <p id="full-refresh-desc" className="text-sm text-text-muted mb-6 dark:text-dark-text-main/70">
              {t('settings.confirm_full_refresh_desc', { count: syncInfo?.total_repos ?? 0 })}
            </p>
            <ul className="text-sm text-text-muted mb-6 space-y-1 list-disc list-inside dark:text-dark-text-main/70">
              <li>{t('settings.confirm_step_fetch')}</li>
              <li>{t('settings.confirm_step_summarize')}</li>
              <li>{t('settings.confirm_step_embed')}</li>
              <li>{t('settings.confirm_step_cluster')}</li>
            </ul>
            <p className="text-xs text-amber-600 bg-amber-50 p-2 rounded mb-6">{t('settings.confirm_warning')}</p>
            <div className="flex justify-end gap-3">
              <button
                ref={cancelButtonRef}
                onClick={onHideConfirm}
                className="px-4 py-2 text-sm font-medium text-text-main bg-bg-hover hover:bg-border-light rounded-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30 dark:text-dark-text-main dark:bg-dark-bg-sidebar/70 dark:hover:bg-dark-border"
              >
                {t('common.cancel')}
              </button>
              <button
                onClick={onConfirmRefresh}
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30"
              >
                {t('settings.execute_full_refresh')}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
