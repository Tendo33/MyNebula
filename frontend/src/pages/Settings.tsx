import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import { AlertTriangle, LogOut, RefreshCw, Server, Shield, Sparkles } from 'lucide-react';

import { Alert, AlertDescription } from '../components/ui/alert';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Separator } from '../components/ui/separator';
import { Slider } from '../components/ui/slider';
import { Spinner } from '../components/ui/spinner';

import { Sidebar } from '../components/layout/Sidebar';
import { HeaderActions } from '../components/layout/HeaderActions';
import { SyncProgress } from '../components/ui/SyncProgress';
import { useGraph } from '../contexts/GraphContext';
import { useAdminAuth } from '../contexts/AdminAuthContext';
import {
  SettingsLoginForm,
  SettingsAppearance,
  SettingsSchedule,
  SettingsDataSection,
} from './settings/index';
import { useSettingsAuth } from './settings/hooks/useSettingsAuth';
import { useSettingsSchedule } from './settings/hooks/useSettingsSchedule';
import { useSettingsSyncControls } from './settings/hooks/useSettingsSyncControls';
import { API_BASE_URL } from '../api/client';

/**
 * Settings page.
 *
 * State and orchestration live in `./settings/hooks/*`; presentational blocks
 * live in `./settings/*`. This component wires them together and owns only the
 * shared error/warning banners.
 */
const Settings = () => {
  const { t } = useTranslation();
  const { settings, updateSettings, refreshData, syncing, setSyncing, setSyncStep } = useGraph();
  const { isChecking, isAuthenticated, login, logout } = useAdminAuth();

  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);

  const scheduleState = useSettingsSchedule({
    isAuthenticated,
    settings,
    updateSettings,
    setError,
    setWarning,
    t,
  });

  const syncControls = useSettingsSyncControls({
    maxClusters: settings.maxClusters,
    minClusters: settings.minClusters,
    setSyncing,
    setSyncStep,
    refreshData,
    reloadSettings: scheduleState.loadScheduleData,
    setError,
    setWarning,
    t,
  });

  const { resetSyncUi } = syncControls;
  const { loadScheduleData, clearScheduleState } = scheduleState;

  const handleSignedOut = useCallback(() => {
    resetSyncUi();
    clearScheduleState();
  }, [resetSyncUi, clearScheduleState]);

  const auth = useSettingsAuth({
    isAuthenticated,
    login,
    logout,
    onLoggedIn: loadScheduleData,
    onLoggedOut: handleSignedOut,
    t,
  });

  useEffect(() => {
    if (!isAuthenticated) {
      handleSignedOut();
      return;
    }

    loadScheduleData();
  }, [isAuthenticated, loadScheduleData, handleSignedOut]);

  const githubTokenStatus = useMemo(() => {
    if (scheduleState.syncInfo?.github_token_configured === true) {
      return { state: 'connected' as const, label: t('settings.connected') };
    }

    if (scheduleState.syncInfo?.github_token_configured === false) {
      return { state: 'not_configured' as const, label: t('settings.not_configured') };
    }

    if (error) {
      return { state: 'unknown' as const, label: t('settings.status_unknown') };
    }

    return { state: 'loading' as const, label: t('common.loading') };
  }, [scheduleState.syncInfo, error, t]);

  const operationsBusy = syncing || syncControls.refreshLoading || syncControls.reclusterLoading;

  return (
    <div className="page-shell">
      <Sidebar />
      <main id="main-content" className="page-main">
        <header className="page-header">
          <div className="page-header-inner">
            <h1 className="page-title select-none">{t('settings.title')}</h1>
          </div>
          <div className="flex items-center gap-3">
            <HeaderActions />
            {isAuthenticated && (
              <Button variant="outline" onClick={auth.handleAdminLogout}>
                <LogOut data-icon="inline-start" />
                {t('app.logout')}
              </Button>
            )}
          </div>
        </header>

        {isChecking ? (
          <div className="flex flex-1 items-center justify-center">
            <Spinner className="size-6 text-text-muted" />
          </div>
        ) : !isAuthenticated ? (
          <SettingsLoginForm
            loginUsername={auth.loginUsername}
            loginPassword={auth.loginPassword}
            loginLoading={auth.loginLoading}
            loginError={auth.loginError}
            adminAuthConfigured={auth.adminAuthConfigured}
            onUsernameChange={auth.setLoginUsername}
            onPasswordChange={auth.setLoginPassword}
            onSubmit={auth.handleAdminLogin}
          />
        ) : (
          <div className="page-content">
            <div className="flex max-w-4xl flex-col gap-12">
              {error && (
                <Alert variant="destructive">
                  <AlertTriangle />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
              {warning && (
                <Alert variant="warning">
                  <AlertTriangle />
                  <AlertDescription>{warning}</AlertDescription>
                </Alert>
              )}

              <SettingsAppearance settings={settings} updateSettings={updateSettings} />

              <Separator />

              <section>
                <h2 className="section-heading mb-4 select-none">{t('settings.operations')}</h2>
                <div className="flex flex-col gap-4">
                  <Card className="p-5">
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <Sparkles className="size-4 text-text-muted" />
                        <span className="text-sm font-medium text-text-main">
                          {t('dashboard.sync_button')}
                        </span>
                      </div>
                      <Button
                        onClick={syncControls.handleSyncStars}
                        disabled={operationsBusy}
                      >
                        {syncing ? <Spinner data-icon="inline-start" /> : null}
                        {syncing ? t('dashboard.syncing') : t('dashboard.sync_button')}
                      </Button>
                    </div>
                  </Card>

                  <Card className="flex flex-col gap-3 p-5">
                    <div className="flex items-center gap-2">
                      <RefreshCw className="w-4 h-4 text-text-muted" />
                      <span className="text-sm font-medium text-text-main">
                        {t('graph.recluster')}
                      </span>
                    </div>

                    <div>
                      <div className="flex items-center justify-between">
                        <label htmlFor="settings-max-clusters" className="text-xs text-text-muted">
                          {t('graph.max_clusters')}
                        </label>
                        <span className="text-xs font-mono tabular-nums text-text-muted">
                          {settings.maxClusters}
                        </span>
                      </div>
                      <Slider
                        min={2}
                        max={20}
                        step={1}
                        value={settings.maxClusters}
                        onValueChange={(value) => {
                          const nextMax = Array.isArray(value) ? value[0] : value;
                          if (typeof nextMax !== 'number') return;
                          if (nextMax < settings.minClusters) {
                            updateSettings({ maxClusters: nextMax, minClusters: nextMax });
                            return;
                          }
                          updateSettings({ maxClusters: nextMax });
                        }}
                        className="mt-2"
                      />
                    </div>

                    <div>
                      <div className="flex items-center justify-between">
                        <label htmlFor="settings-min-clusters" className="text-xs text-text-muted">
                          {t('graph.min_clusters')}
                        </label>
                        <span className="text-xs font-mono tabular-nums text-text-muted">
                          {settings.minClusters}
                        </span>
                      </div>
                      <Slider
                        min={2}
                        max={20}
                        step={1}
                        value={settings.minClusters}
                        onValueChange={(value) => {
                          const nextMin = Array.isArray(value) ? value[0] : value;
                          if (typeof nextMin !== 'number') return;
                          if (nextMin > settings.maxClusters) {
                            updateSettings({ minClusters: nextMin, maxClusters: nextMin });
                            return;
                          }
                          updateSettings({ minClusters: nextMin });
                        }}
                        className="mt-2"
                      />
                    </div>

                    <Button
                      variant="outline"
                      onClick={syncControls.handleRecluster}
                      disabled={operationsBusy}
                      title={t('graph.recluster_hint')}
                    >
                      {syncControls.reclusterLoading ? <Spinner data-icon="inline-start" /> : null}
                      {syncControls.reclusterLoading
                        ? t('graph.reclustering')
                        : t('graph.recluster')}
                    </Button>
                  </Card>
                </div>
              </section>

              <Separator />

              <section>
                <h2 className="section-heading mb-4 select-none">{t('settings.connection')}</h2>
                <div className="flex flex-col gap-2">
                  <Card variant="muted" className="p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Server className="w-4 h-4 text-text-muted" />
                      <label
                        htmlFor="settings-api-endpoint"
                        className="text-sm font-medium text-text-main"
                      >
                        {t('settings.api_endpoint')}
                      </label>
                    </div>
                    <Input
                      id="settings-api-endpoint"
                      type="text"
                      value={API_BASE_URL}
                      readOnly
                      className="font-mono text-muted-foreground"
                    />
                  </Card>

                  <Card variant="muted" className="flex flex-row items-center justify-between p-4 transition-colors">
                    <div className="flex items-center gap-2">
                      <Shield className="w-4 h-4 text-text-muted" />
                      <span className="text-sm font-medium text-text-main">
                        {t('settings.github_token_status')}
                      </span>
                    </div>
                    <Badge
                      variant={
                        githubTokenStatus.state === 'connected'
                          ? 'secondary'
                          : githubTokenStatus.state === 'not_configured'
                            ? 'outline'
                            : 'ghost'
                      }
                      className={clsx(
                        githubTokenStatus.state === 'connected' &&
                          'border-success/30 bg-success-bg text-success',
                        githubTokenStatus.state === 'not_configured' &&
                          'border-warning/40 bg-warning-bg text-warning-foreground'
                      )}
                    >
                      <span
                        className={clsx(
                          'size-2 rounded-full',
                          githubTokenStatus.state === 'connected' && 'bg-success animate-pulse',
                          githubTokenStatus.state === 'not_configured' && 'bg-warning',
                          githubTokenStatus.state === 'unknown' && 'bg-text-dim',
                          githubTokenStatus.state === 'loading' && 'bg-text-dim/70 animate-pulse'
                        )}
                      />
                      {githubTokenStatus.label}
                    </Badge>
                  </Card>
                </div>
              </section>

              <Separator />

              <SettingsSchedule
                schedule={scheduleState.schedule}
                scheduleLoading={scheduleState.scheduleLoading}
                onToggle={scheduleState.handleScheduleToggle}
                onTimeChange={scheduleState.handleTimeChange}
              />

              <Separator />

              <SettingsDataSection
                syncInfo={scheduleState.syncInfo}
                refreshLoading={syncControls.refreshLoading}
                syncing={syncing}
                reclusterLoading={syncControls.reclusterLoading}
                showConfirmDialog={syncControls.showConfirmDialog}
                onShowConfirm={() => syncControls.setShowConfirmDialog(true)}
                onHideConfirm={() => syncControls.setShowConfirmDialog(false)}
                onConfirmRefresh={syncControls.handleFullRefresh}
              />
            </div>
          </div>
        )}
      </main>

      <SyncProgress
        isOpen={syncControls.showSyncProgress}
        onClose={() => syncControls.handleSyncProgressClose(syncing)}
        steps={syncControls.translatedSteps}
        title={syncControls.progressTitle || t('sync.title', 'Syncing Data')}
        canClose={!syncing}
      />
    </div>
  );
};

export default Settings;
