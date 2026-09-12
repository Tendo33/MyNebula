import { useTranslation } from 'react-i18next';
import { AlertTriangle, Database, RefreshCw } from 'lucide-react';
import type { SyncInfoResponse } from '../../api/v2/settings';

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Spinner } from '@/components/ui/spinner';

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

  return (
    <>
      <section>
        <h2 className="section-heading mb-4 select-none">
          {t('settings.data_management')}
        </h2>
        <div className="flex flex-col gap-2">
          <Card variant="muted" className="p-4">
            <div className="mb-3 flex items-center gap-2">
              <Database className="size-4 text-text-muted" />
              <label className="text-sm font-medium text-text-main">{t('settings.repo_stats')}</label>
            </div>
            <div className="grid grid-cols-2 gap-4 rounded-xl border border-border-light bg-bg-elevated p-4">
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
          </Card>

          <Card variant="muted" className="p-4">
            <div className="mb-3 flex items-center gap-2">
              <RefreshCw className="size-4 text-text-muted" />
              <label className="text-sm font-medium text-text-main">{t('settings.full_refresh')}</label>
            </div>
            <p className="mb-3 text-xs text-text-muted">{t('settings.full_refresh_desc')}</p>
            <Button
              type="button"
              variant="destructive"
              onClick={onShowConfirm}
              disabled={refreshLoading || syncing || reclusterLoading}
            >
              {refreshLoading ? <Spinner data-icon="inline-start" /> : <RefreshCw data-icon="inline-start" />}
              {refreshLoading ? t('settings.refreshing') : t('settings.execute_full_refresh')}
            </Button>
          </Card>
        </div>
      </section>

      <AlertDialog
        open={showConfirmDialog}
        onOpenChange={(open) => {
          if (!open) onHideConfirm();
        }}
      >
        <AlertDialogContent className="sm:max-w-md">
          <AlertDialogHeader>
            <AlertDialogMedia className="bg-danger-bg text-danger">
              <AlertTriangle />
            </AlertDialogMedia>
            <AlertDialogTitle>{t('settings.confirm_full_refresh_title')}</AlertDialogTitle>
            <AlertDialogDescription>
              {t('settings.confirm_full_refresh_desc', { count: syncInfo?.total_repos ?? 0 })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <ul className="flex list-inside list-disc flex-col gap-1 text-sm text-muted-foreground">
            <li>{t('settings.confirm_step_fetch')}</li>
            <li>{t('settings.confirm_step_summarize')}</li>
            <li>{t('settings.confirm_step_embed')}</li>
            <li>{t('settings.confirm_step_cluster')}</li>
          </ul>
          <p className="rounded-md border border-border bg-muted p-3 text-xs text-muted-foreground">
            {t('settings.confirm_warning')}
          </p>
          <AlertDialogFooter>
            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={onConfirmRefresh}>
              {t('settings.execute_full_refresh')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
};
