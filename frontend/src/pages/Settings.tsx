import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import { AlertTriangle, Loader2, LogOut, RefreshCw, Server, Shield, Sparkles } from 'lucide-react';

import { Sidebar } from '../components/layout/Sidebar';
import { LanguageSwitch } from '../components/layout/LanguageSwitch';
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
            <div>
              <div className="section-kicker mb-1 px-0">{t('sidebar.settings')}</div>
              <h1 className="page-title select-none">{t('settings.title')}</h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <LanguageSwitch />
            {isAuthenticated && (
              <button
                onClick={auth.handleAdminLogout}
                className="header-action min-h-11 px-4 text-xs"
              >
                <LogOut className="w-3.5 h-3.5" />
                {t('app.logout')}
              </button>
            )}
          </div>
        </header>

        {isChecking ? (
          <div className="flex-1 flex items-center justify-center">
            <Loader2 className="w-6 h-6 animate-spin text-text-muted" />
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
            <div className="max-w-4xl space-y-12">
              {error && (
                <div className="status-banner" data-tone="error">
                  <AlertTriangle className="w-4 h-4" />
                  {error}
                </div>
              )}
              {warning && (
                <div className="status-banner" data-tone="warning">
                  <AlertTriangle className="w-4 h-4" />
                  {warning}
                </div>
              )}

              <SettingsAppearance settings={settings} updateSettings={updateSettings} />

              <hr className="border-t border-border-light/80" />

              <section>
                <h2 className="section-kicker mb-4 select-none">{t('settings.operations')}</h2>
                <div className="space-y-4">
                  <div className="panel-surface p-5">
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-text-muted" />
                        <span className="text-sm font-medium text-text-main">
                          {t('dashboard.sync_button')}
                        </span>
                      </div>
                      <button
                        onClick={syncControls.handleSyncStars}
                        disabled={operationsBusy}
                        className={clsx(
                          'inline-flex min-h-[2.75rem] items-center gap-2 rounded-xl px-4 text-sm font-medium transition-colors',
                          operationsBusy
                            ? 'bg-bg-hover text-text-dim cursor-not-allowed dark:bg-dark-bg-sidebar/70 dark:text-dark-text-main/60'
                            : 'bg-text-main text-bg-main hover:bg-text-main/90 shadow-sm'
                        )}
                      >
                        {syncing && <Loader2 className="w-4 h-4 animate-spin" />}
                        {syncing ? t('dashboard.syncing') : t('dashboard.sync_button')}
                      </button>
                    </div>
                  </div>

                  <div className="panel-surface space-y-3 p-5">
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
                      <input
                        id="settings-max-clusters"
                        type="range"
                        min={2}
                        max={20}
                        step={1}
                        value={settings.maxClusters}
                        onChange={(e) => {
                          const nextMax = Number(e.target.value);
                          if (nextMax < settings.minClusters) {
                            updateSettings({ maxClusters: nextMax, minClusters: nextMax });
                            return;
                          }
                          updateSettings({ maxClusters: nextMax });
                        }}
                        className="w-full mt-2"
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
                      <input
                        id="settings-min-clusters"
                        type="range"
                        min={2}
                        max={20}
                        step={1}
                        value={settings.minClusters}
                        onChange={(e) => {
                          const nextMin = Number(e.target.value);
                          if (nextMin > settings.maxClusters) {
                            updateSettings({ minClusters: nextMin, maxClusters: nextMin });
                            return;
                          }
                          updateSettings({ minClusters: nextMin });
                        }}
                        className="w-full mt-2"
                      />
                    </div>

                    <button
                      onClick={syncControls.handleRecluster}
                      disabled={operationsBusy}
                      className={clsx(
                        'inline-flex min-h-[2.75rem] items-center gap-2 rounded-xl border px-4 text-sm font-medium transition-colors',
                        operationsBusy
                          ? 'bg-bg-hover text-text-dim border-border-light cursor-not-allowed dark:bg-dark-bg-sidebar/70 dark:text-dark-text-main/60 dark:border-dark-border'
                          : 'bg-bg-main text-text-main border-border-light hover:bg-bg-hover dark:bg-dark-bg-main dark:text-dark-text-main dark:border-dark-border dark:hover:bg-dark-bg-sidebar/70'
                      )}
                      title={t('graph.recluster_hint')}
                    >
                      {syncControls.reclusterLoading && (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      )}
                      {syncControls.reclusterLoading
                        ? t('graph.reclustering')
                        : t('graph.recluster')}
                    </button>
                  </div>
                </div>
              </section>

              <hr className="border-t border-border-light/80" />

              <section>
                <h2 className="section-kicker mb-4 select-none">{t('settings.connection')}</h2>
                <div className="space-y-2">
                  <div className="panel-subtle p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Server className="w-4 h-4 text-text-muted" />
                      <label
                        htmlFor="settings-api-endpoint"
                        className="text-sm font-medium text-text-main"
                      >
                        {t('settings.api_endpoint')}
                      </label>
                    </div>
                    <input
                      id="settings-api-endpoint"
                      type="text"
                      value={API_BASE_URL}
                      readOnly
                      className="w-full bg-bg-sidebar/50 border border-border-light rounded-md px-3 py-2 text-sm text-text-muted font-mono"
                    />
                  </div>

                  <div className="panel-subtle flex items-center justify-between p-4 transition-colors">
                    <div className="flex items-center gap-2">
                      <Shield className="w-4 h-4 text-text-muted" />
                      <span className="text-sm font-medium text-text-main">
                        {t('settings.github_token_status')}
                      </span>
                    </div>
                    <div
                      className={clsx(
                        'flex items-center gap-2 text-sm font-medium px-3 py-1 rounded-full border',
                        githubTokenStatus.state === 'connected' &&
                          'text-green-700 bg-green-50 border-green-200',
                        githubTokenStatus.state === 'not_configured' &&
                          'text-amber-700 bg-amber-50 border-amber-200',
                        githubTokenStatus.state === 'unknown' &&
                          'text-text-muted bg-bg-hover border-border-light dark:text-dark-text-main/70 dark:bg-dark-bg-sidebar/70 dark:border-dark-border',
                        githubTokenStatus.state === 'loading' &&
                          'text-text-muted bg-bg-hover border-border-light dark:text-dark-text-main/60 dark:bg-dark-bg-sidebar/60 dark:border-dark-border'
                      )}
                    >
                      <div
                        className={clsx(
                          'w-2 h-2 rounded-full',
                          githubTokenStatus.state === 'connected' && 'bg-green-500 animate-pulse',
                          githubTokenStatus.state === 'not_configured' && 'bg-amber-500',
                          githubTokenStatus.state === 'unknown' && 'bg-text-dim',
                          githubTokenStatus.state === 'loading' && 'bg-text-dim/70 animate-pulse'
                        )}
                      />
                      {githubTokenStatus.label}
                    </div>
                  </div>
                </div>
              </section>

              <hr className="border-t border-border-light/80" />

              <SettingsSchedule
                schedule={scheduleState.schedule}
                scheduleLoading={scheduleState.scheduleLoading}
                onToggle={scheduleState.handleScheduleToggle}
                onTimeChange={scheduleState.handleTimeChange}
              />

              <hr className="border-t border-border-light/80" />

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
