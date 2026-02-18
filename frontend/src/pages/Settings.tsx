import { FormEvent, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import {
  AlertTriangle,
  Clock,
  Database,
  Eye,
  Loader2,
  LogOut,
  RefreshCw,
  Server,
  Shield,
  Sparkles,
  User,
  Zap,
} from 'lucide-react';

import { Sidebar } from '../components/layout/Sidebar';
import { LanguageSwitch } from '../components/layout/LanguageSwitch';
import { SyncProgress, SyncStepStatus } from '../components/ui/SyncProgress';
import { useGraph } from '../contexts/GraphContext';
import { useAdminAuth } from '../contexts/AdminAuthContext';
import { API_BASE_URL } from '../api/client';
import {
  startClustering,
  startEmbedding,
  startStarSync,
  startSummaries,
  getSyncStatus,
} from '../api/sync';
import {
  formatLastRunTime,
  formatNextRunTime,
  getSchedule,
  getStatusDisplay,
  getSyncInfo,
  triggerFullRefresh,
  updateSchedule,
  type ScheduleConfig,
  type ScheduleResponse,
  type SyncInfoResponse,
} from '../api/schedule';

interface TaskProgressUpdate {
  state: string;
  progress: number;
  error: string | null;
}

interface TaskCompletionResult {
  success: boolean;
  error: string | null;
}

const Settings = () => {
  const { t } = useTranslation();
  const {
    settings,
    updateSettings,
    refreshData,
    syncing,
    setSyncing,
    setSyncStep,
  } = useGraph();
  const { isChecking, isAuthenticated, login, logout } = useAdminAuth();

  const [schedule, setSchedule] = useState<ScheduleResponse | null>(null);
  const [syncInfo, setSyncInfo] = useState<SyncInfoResponse | null>(null);
  const [scheduleLoading, setScheduleLoading] = useState(false);
  const [refreshLoading, setRefreshLoading] = useState(false);
  const [reclusterLoading, setReclusterLoading] = useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [loginUsername, setLoginUsername] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  const [showSyncProgress, setShowSyncProgress] = useState(false);
  const [syncSteps, setSyncSteps] = useState<
    { id: string; status: SyncStepStatus; progress?: number; error?: string }[]
  >([
    { id: 'stars', status: 'pending' },
    { id: 'summaries', status: 'pending' },
    { id: 'embeddings', status: 'pending' },
    { id: 'clustering', status: 'pending' },
  ]);

  const translatedSteps = useMemo(
    () =>
      syncSteps.map((step) => ({
        ...step,
        label: t(`sync.step_${step.id}_label`),
        description: t(`sync.step_${step.id}_desc`),
      })),
    [syncSteps, t]
  );

  const githubTokenStatus = useMemo(() => {
    if (syncInfo?.github_token_configured === true) {
      return { state: 'connected' as const, label: t('settings.connected') };
    }

    if (syncInfo?.github_token_configured === false) {
      return { state: 'not_configured' as const, label: t('settings.not_configured') };
    }

    if (error) {
      return { state: 'unknown' as const, label: t('settings.status_unknown') };
    }

    return { state: 'loading' as const, label: t('common.loading') };
  }, [syncInfo, error, t]);

  const loadScheduleData = useCallback(async () => {
    try {
      setError(null);
      const [scheduleData, infoData] = await Promise.all([getSchedule(), getSyncInfo()]);
      setSchedule(scheduleData);
      setSyncInfo(infoData);
    } catch (err) {
      setError(t('settings.load_schedule_error'));
      console.error('Failed to load schedule data:', err);
    }
  }, [t]);

  useEffect(() => {
    if (!isAuthenticated) {
      setSchedule(null);
      setSyncInfo(null);
      return;
    }

    loadScheduleData();
  }, [isAuthenticated, loadScheduleData]);

  const waitForTaskComplete = async (
    taskId: number,
    onProgress?: (update: TaskProgressUpdate) => void
  ): Promise<TaskCompletionResult> => {
    while (true) {
      try {
        const status = await getSyncStatus(taskId);
        const progress = Number.isFinite(status.progress_percent)
          ? Math.max(0, Math.min(100, status.progress_percent))
          : 0;

        onProgress?.({
          state: status.status,
          progress,
          error: status.error_message,
        });

        if (status.status === 'completed') {
          return { success: true, error: null };
        }

        if (status.status === 'failed') {
          console.error('Task failed:', status.error_message);
          return { success: false, error: status.error_message };
        }
      } catch (err) {
        console.error('Poll status error:', err);
        return {
          success: false,
          error: err instanceof Error ? err.message : 'poll_status_failed',
        };
      }

      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  };

  const updateStepStatus = (
    stepId: string,
    status: SyncStepStatus,
    stepError?: string,
    progress?: number
  ) => {
    setSyncSteps((prev) =>
      prev.map((step) =>
        step.id === stepId
          ? {
              ...step,
              status,
              error: stepError,
              progress:
                progress !== undefined
                  ? Math.max(0, Math.min(100, Math.round(progress)))
                  : status === 'running'
                  ? step.progress
                  : undefined,
            }
          : step
      )
    );
  };

  const resetSteps = () => {
    setSyncSteps([
      { id: 'stars', status: 'pending', progress: 0 },
      { id: 'summaries', status: 'pending', progress: 0 },
      { id: 'embeddings', status: 'pending', progress: 0 },
      { id: 'clustering', status: 'pending', progress: 0 },
    ]);
  };

  const handleAdminLogin = async (e: FormEvent) => {
    e.preventDefault();
    setLoginLoading(true);
    setLoginError(null);

    try {
      await login(loginUsername.trim(), loginPassword);
      setLoginPassword('');
      await loadScheduleData();
    } catch (err) {
      setLoginError(t('settings.login_failed'));
      console.error('Admin login failed:', err);
    } finally {
      setLoginLoading(false);
    }
  };

  const handleAdminLogout = async () => {
    await logout();
    setSchedule(null);
    setSyncInfo(null);
  };

  const handleScheduleToggle = async () => {
    if (!schedule) return;

    setScheduleLoading(true);
    try {
      const newConfig: ScheduleConfig = {
        is_enabled: !schedule.is_enabled,
        schedule_hour: schedule.schedule_hour,
        schedule_minute: schedule.schedule_minute,
        timezone: schedule.timezone,
      };
      const updated = await updateSchedule(newConfig);
      setSchedule(updated);
    } catch (err) {
      setError(t('settings.update_schedule_error'));
      console.error('Failed to update schedule:', err);
    } finally {
      setScheduleLoading(false);
    }
  };

  const handleTimeChange = async (hour: number, minute: number) => {
    if (!schedule) return;

    setScheduleLoading(true);
    try {
      const newConfig: ScheduleConfig = {
        is_enabled: schedule.is_enabled,
        schedule_hour: hour,
        schedule_minute: minute,
        timezone: schedule.timezone,
      };
      const updated = await updateSchedule(newConfig);
      setSchedule(updated);
    } catch (err) {
      setError(t('settings.update_time_error'));
      console.error('Failed to update schedule time:', err);
    } finally {
      setScheduleLoading(false);
    }
  };

  const handleSyncStars = async () => {
    try {
      setError(null);
      setSyncing(true);
      setShowSyncProgress(true);
      resetSteps();

      updateStepStatus('stars', 'running', undefined, 0);
      setSyncStep(t('sync.step_stars_label'));
      const starsResult = await startStarSync('incremental');
      const starsResultStatus = await waitForTaskComplete(starsResult.task_id, ({ progress }) => {
        updateStepStatus('stars', 'running', undefined, progress);
      });
      if (!starsResultStatus.success) {
        const starsError = starsResultStatus.error || t('errors.sync_stars_failed');
        updateStepStatus('stars', 'failed', starsError);
        throw new Error(starsError);
      }
      updateStepStatus('stars', 'completed', undefined, 100);

      updateStepStatus('summaries', 'running', undefined, 0);
      setSyncStep(t('sync.step_summaries_label'));
      try {
        const summariesResult = await startSummaries();
        const summariesStatus = await waitForTaskComplete(
          summariesResult.task_id,
          ({ progress }) => {
            updateStepStatus('summaries', 'running', undefined, progress);
          }
        );
        if (!summariesStatus.success) {
          console.warn('Summaries generation skipped:', summariesStatus.error);
        }
      } catch (err) {
        console.warn('Summaries generation skipped:', err);
      }
      updateStepStatus('summaries', 'completed', undefined, 100);

      updateStepStatus('embeddings', 'running', undefined, 0);
      setSyncStep(t('sync.step_embeddings_label'));
      const embeddingResult = await startEmbedding();
      const embeddingStatus = await waitForTaskComplete(embeddingResult.task_id, ({ progress }) => {
        updateStepStatus('embeddings', 'running', undefined, progress);
      });
      if (!embeddingStatus.success) {
        const embeddingError = embeddingStatus.error || t('errors.embedding_failed');
        updateStepStatus('embeddings', 'failed', embeddingError);
        throw new Error(embeddingError);
      }
      updateStepStatus('embeddings', 'completed', undefined, 100);

      updateStepStatus('clustering', 'running', undefined, 0);
      setSyncStep(t('sync.step_clustering_label'));
      const clusterResult = await startClustering(
        true,
        settings.maxClusters,
        settings.minClusters
      );
      const clusterStatus = await waitForTaskComplete(clusterResult.task_id, ({ progress }) => {
        updateStepStatus('clustering', 'running', undefined, progress);
      });
      if (!clusterStatus.success) {
        const clusterError = clusterStatus.error || t('errors.clustering_failed');
        updateStepStatus('clustering', 'failed', clusterError);
        throw new Error(clusterError);
      }
      updateStepStatus('clustering', 'completed', undefined, 100);

      await refreshData();
      await loadScheduleData();
    } catch (err) {
      setError(t('errors.sync_failed'));
      console.error('Sync failed:', err);
    } finally {
      setSyncing(false);
      setSyncStep('');
    }
  };

  const handleRecluster = async () => {
    setReclusterLoading(true);
    setError(null);

    try {
      const started = await startClustering(true, settings.maxClusters, settings.minClusters);
      const clusterStatus = await waitForTaskComplete(started.task_id);
      if (!clusterStatus.success) {
        throw new Error(clusterStatus.error || t('errors.clustering_failed'));
      }

      await refreshData();
      await loadScheduleData();
    } catch (err) {
      setError(t('errors.clustering_failed'));
      console.error('Re-cluster failed:', err);
    } finally {
      setReclusterLoading(false);
    }
  };

  const handleFullRefresh = async () => {
    setShowConfirmDialog(false);
    setRefreshLoading(true);

    try {
      await triggerFullRefresh();
      await loadScheduleData();
    } catch (err) {
      setError(t('settings.full_refresh_error'));
      console.error('Failed to trigger full refresh:', err);
    } finally {
      setRefreshLoading(false);
    }
  };

  const handleSyncProgressClose = () => {
    setShowSyncProgress(false);
    if (!syncing) {
      resetSteps();
    }
  };

  return (
    <div className="flex min-h-screen bg-bg-main text-text-main">
      <Sidebar />
      <main className="flex-1 flex flex-col min-w-0" style={{ marginLeft: 'var(--sidebar-width, 240px)' }}>
        <header className="flex items-center justify-between h-14 px-8 border-b border-border-light sticky top-0 bg-bg-main/95 backdrop-blur-sm z-40">
          <h1 className="text-base font-semibold text-text-main select-none tracking-tight">
            {t('settings.title')}
          </h1>
          <div className="flex items-center gap-3">
            <LanguageSwitch />
            {isAuthenticated && (
              <button
                onClick={handleAdminLogout}
                className="h-8 px-3 rounded-md text-xs font-medium border border-border-light bg-white hover:bg-bg-hover text-text-main transition-colors inline-flex items-center gap-1.5"
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
          <section className="flex-1 flex items-center justify-center px-8">
            <div className="w-full max-w-md bg-white border border-border-light rounded-xl shadow-sm p-6">
              <div className="flex items-center gap-3 mb-5">
                <div className="w-10 h-10 rounded-full bg-bg-sidebar flex items-center justify-center">
                  <Shield className="w-5 h-5 text-text-main" />
                </div>
                <div>
                  <h2 className="text-base font-semibold text-text-main">{t('settings.admin_access')}</h2>
                  <p className="text-xs text-text-muted">{t('settings.login_required_desc')}</p>
                </div>
              </div>

              <form className="space-y-4" onSubmit={handleAdminLogin}>
                <div>
                  <label className="block text-xs text-text-muted mb-1">{t('settings.username')}</label>
                  <div className="relative">
                    <User className="w-4 h-4 text-text-dim absolute left-3 top-1/2 -translate-y-1/2" />
                    <input
                      type="text"
                      value={loginUsername}
                      onChange={(e) => setLoginUsername(e.target.value)}
                      className="w-full h-10 pl-9 pr-3 rounded-md border border-border-light text-sm bg-white focus:outline-none focus:ring-1 focus:ring-black"
                      autoComplete="username"
                      required
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs text-text-muted mb-1">{t('settings.password')}</label>
                  <input
                    type="password"
                    value={loginPassword}
                    onChange={(e) => setLoginPassword(e.target.value)}
                    className="w-full h-10 px-3 rounded-md border border-border-light text-sm bg-white focus:outline-none focus:ring-1 focus:ring-black"
                    autoComplete="current-password"
                    required
                  />
                </div>

                {loginError && (
                  <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-md px-3 py-2">
                    {loginError}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={loginLoading}
                  className={clsx(
                    'w-full h-10 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2',
                    loginLoading
                      ? 'bg-gray-100 text-gray-400 border border-border-light cursor-not-allowed'
                      : 'bg-black text-white hover:bg-gray-800'
                  )}
                >
                  {loginLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                  {t('app.login')}
                </button>
              </form>
            </div>
          </section>
        ) : (
          <div className="max-w-3xl px-8 py-10 space-y-12">
            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-md text-sm text-red-700 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" />
                {error}
              </div>
            )}

            <section>
              <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-4 px-2 select-none">
                {t('settings.appearance')}
              </h2>
              <div className="space-y-2">
                <div
                  className="flex items-center justify-between p-3 rounded-md hover:bg-bg-hover transition-all group cursor-pointer"
                  onClick={() => updateSettings({ hqRendering: !settings.hqRendering })}
                >
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-white transition-colors">
                      <Zap className="w-5 h-5 text-text-muted group-hover:text-text-main" />
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-text-main">{t('settings.hq_rendering')}</span>
                      <span className="text-xs text-text-muted">{t('settings.hq_rendering_desc')}</span>
                    </div>
                  </div>
                  <button
                    className={clsx(
                      'relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none',
                      settings.hqRendering ? 'bg-black' : 'bg-gray-300'
                    )}
                  >
                    <span
                      className={clsx(
                        'inline-block h-5 w-5 transform rounded-full bg-white transition duration-200 ease-in-out shadow-sm',
                        settings.hqRendering ? 'translate-x-5' : 'translate-x-0.5'
                      )}
                    />
                  </button>
                </div>

                <div
                  className="flex items-center justify-between p-3 rounded-md hover:bg-bg-hover transition-all group cursor-pointer"
                  onClick={() => updateSettings({ showTrajectories: !settings.showTrajectories })}
                >
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-white transition-colors">
                      <Eye className="w-5 h-5 text-text-muted group-hover:text-text-main" />
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-text-main">{t('settings.show_trajectories')}</span>
                      <span className="text-xs text-text-muted">{t('settings.show_trajectories_desc')}</span>
                    </div>
                  </div>
                  <button
                    className={clsx(
                      'relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none',
                      settings.showTrajectories ? 'bg-black' : 'bg-gray-300'
                    )}
                  >
                    <span
                      className={clsx(
                        'inline-block h-5 w-5 transform rounded-full bg-white transition duration-200 ease-in-out shadow-sm',
                        settings.showTrajectories ? 'translate-x-5' : 'translate-x-0.5'
                      )}
                    />
                  </button>
                </div>
              </div>
            </section>

            <hr className="border-t border-border-light" />

            <section>
              <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-4 px-2 select-none">
                {t('settings.operations')}
              </h2>
              <div className="space-y-4">
                <div className="p-3 border border-border-light rounded-md">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-text-muted" />
                      <span className="text-sm font-medium text-text-main">{t('dashboard.sync_button')}</span>
                    </div>
                    <button
                      onClick={handleSyncStars}
                      disabled={syncing || refreshLoading || reclusterLoading}
                      className={clsx(
                        'h-9 px-4 rounded-md text-sm font-medium transition-all flex items-center gap-2',
                        syncing || refreshLoading || reclusterLoading
                          ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                          : 'bg-black text-white hover:bg-gray-800 shadow-sm'
                      )}
                    >
                      {syncing && <Loader2 className="w-4 h-4 animate-spin" />}
                      {syncing ? t('dashboard.syncing') : t('dashboard.sync_button')}
                    </button>
                  </div>
                </div>

                <div className="p-3 border border-border-light rounded-md space-y-3">
                  <div className="flex items-center gap-2">
                    <RefreshCw className="w-4 h-4 text-text-muted" />
                    <span className="text-sm font-medium text-text-main">{t('graph.recluster')}</span>
                  </div>

                  <div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-text-muted">{t('graph.max_clusters')}</span>
                      <span className="text-xs font-mono tabular-nums text-text-dim">{settings.maxClusters}</span>
                    </div>
                    <input
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
                      <span className="text-xs text-text-muted">{t('graph.min_clusters')}</span>
                      <span className="text-xs font-mono tabular-nums text-text-dim">{settings.minClusters}</span>
                    </div>
                    <input
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
                    onClick={handleRecluster}
                    disabled={reclusterLoading || syncing || refreshLoading}
                    className={clsx(
                      'h-9 px-4 rounded-md text-sm font-medium border transition-colors inline-flex items-center gap-2',
                      reclusterLoading || syncing || refreshLoading
                        ? 'bg-gray-100 text-gray-400 border-border-light cursor-not-allowed'
                        : 'bg-white text-text-main border-border-light hover:bg-bg-hover'
                    )}
                    title={t('graph.recluster_hint')}
                  >
                    {reclusterLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                    {reclusterLoading ? t('graph.reclustering') : t('graph.recluster')}
                  </button>
                </div>
              </div>
            </section>

            <hr className="border-t border-border-light" />

            <section>
              <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-4 px-2 select-none">
                {t('settings.connection')}
              </h2>
              <div className="space-y-2">
                <div className="p-3">
                  <div className="flex items-center gap-2 mb-3">
                    <Server className="w-4 h-4 text-text-muted" />
                    <label className="text-sm font-medium text-text-main">{t('settings.api_endpoint')}</label>
                  </div>
                  <input
                    type="text"
                    value={API_BASE_URL}
                    readOnly
                    className="w-full bg-bg-sidebar/50 border border-border-light rounded-md px-3 py-2 text-sm text-text-muted font-mono"
                  />
                </div>

                <div className="flex items-center justify-between p-3 rounded-md hover:bg-bg-hover transition-colors">
                  <div className="flex items-center gap-2">
                    <Shield className="w-4 h-4 text-text-muted" />
                    <label className="text-sm font-medium text-text-main">{t('settings.github_token_status')}</label>
                  </div>
                  <div
                    className={clsx(
                      'flex items-center gap-2 text-sm font-medium px-3 py-1 rounded-full border',
                      githubTokenStatus.state === 'connected' && 'text-green-700 bg-green-50 border-green-200',
                      githubTokenStatus.state === 'not_configured' && 'text-amber-700 bg-amber-50 border-amber-200',
                      githubTokenStatus.state === 'unknown' && 'text-gray-700 bg-gray-100 border-gray-200',
                      githubTokenStatus.state === 'loading' && 'text-gray-600 bg-gray-50 border-gray-200'
                    )}
                  >
                    <div
                      className={clsx(
                        'w-2 h-2 rounded-full',
                        githubTokenStatus.state === 'connected' && 'bg-green-500 animate-pulse',
                        githubTokenStatus.state === 'not_configured' && 'bg-amber-500',
                        githubTokenStatus.state === 'unknown' && 'bg-gray-500',
                        githubTokenStatus.state === 'loading' && 'bg-gray-400 animate-pulse'
                      )}
                    />
                    {githubTokenStatus.label}
                  </div>
                </div>
              </div>
            </section>

            <hr className="border-t border-border-light" />

            <section>
              <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-4 px-2 select-none">
                {t('settings.scheduled_sync')}
              </h2>

              <div className="space-y-2">
                <div
                  className="flex items-center justify-between p-3 rounded-md hover:bg-bg-hover transition-all group cursor-pointer"
                  onClick={!scheduleLoading ? handleScheduleToggle : undefined}
                >
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-white transition-colors">
                      <Clock className="w-5 h-5 text-text-muted group-hover:text-text-main" />
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-text-main">{t('settings.enable_scheduled_sync')}</span>
                      <span className="text-xs text-text-muted">{t('settings.enable_scheduled_sync_desc')}</span>
                    </div>
                  </div>
                  <button
                    disabled={scheduleLoading}
                    className={clsx(
                      'relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none',
                      schedule?.is_enabled ? 'bg-black' : 'bg-gray-300',
                      scheduleLoading && 'opacity-50 cursor-not-allowed'
                    )}
                  >
                    {scheduleLoading ? (
                      <Loader2 className="w-4 h-4 text-white mx-auto animate-spin" />
                    ) : (
                      <span
                        className={clsx(
                          'inline-block h-5 w-5 transform rounded-full bg-white transition duration-200 ease-in-out shadow-sm',
                          schedule?.is_enabled ? 'translate-x-5' : 'translate-x-0.5'
                        )}
                      />
                    )}
                  </button>
                </div>

                {schedule?.is_enabled && (
                  <div className="p-3 pl-14">
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2">
                        <label className="text-sm text-text-muted">{t('settings.execution_time')}:</label>
                        <select
                          value={schedule.schedule_hour}
                          onChange={(e) => handleTimeChange(Number(e.target.value), schedule.schedule_minute)}
                          disabled={scheduleLoading}
                          className="bg-bg-sidebar border border-border-light rounded-md px-3 py-1.5 text-sm text-text-main focus:outline-none focus:ring-1 focus:ring-black"
                        >
                          {Array.from({ length: 24 }, (_, i) => (
                            <option key={i} value={i}>
                              {i.toString().padStart(2, '0')}
                            </option>
                          ))}
                        </select>
                        <span className="text-text-muted">:</span>
                        <select
                          value={schedule.schedule_minute}
                          onChange={(e) => handleTimeChange(schedule.schedule_hour, Number(e.target.value))}
                          disabled={scheduleLoading}
                          className="bg-bg-sidebar border border-border-light rounded-md px-3 py-1.5 text-sm text-text-main focus:outline-none focus:ring-1 focus:ring-black"
                        >
                          {[0, 15, 30, 45].map((m) => (
                            <option key={m} value={m}>
                              {m.toString().padStart(2, '0')}
                            </option>
                          ))}
                        </select>
                      </div>
                      <span className="text-xs text-text-muted">({schedule.timezone})</span>
                    </div>
                  </div>
                )}

                <div className="p-3 pl-14 space-y-1">
                  <div className="flex items-center gap-2 text-xs text-text-muted">
                    <span>{t('settings.last_run')}:</span>
                    <span className="text-text-main">
                      {schedule ? formatLastRunTime(schedule.last_run_at, t) : t('common.loading')}
                    </span>
                    {schedule?.last_run_status && (
                      <span className={clsx('font-medium', getStatusDisplay(schedule.last_run_status, t).color)}>
                        ({getStatusDisplay(schedule.last_run_status, t).text})
                      </span>
                    )}
                  </div>
                  {schedule?.is_enabled && schedule.next_run_at && (
                    <div className="flex items-center gap-2 text-xs text-text-muted">
                      <span>{t('settings.next_run')}:</span>
                      <span className="text-text-main">
                        {formatNextRunTime(schedule.next_run_at, schedule.timezone, t)}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </section>

            <hr className="border-t border-border-light" />

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
                    onClick={() => setShowConfirmDialog(true)}
                    disabled={refreshLoading || syncing || reclusterLoading}
                    className={clsx(
                      'flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors',
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
          </div>
        )}

        {showConfirmDialog && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-red-100 rounded-full">
                  <AlertTriangle className="w-6 h-6 text-red-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">{t('settings.confirm_full_refresh_title')}</h3>
              </div>
              <p className="text-sm text-gray-600 mb-6">
                {t('settings.confirm_full_refresh_desc', { count: syncInfo?.total_repos ?? 0 })}
              </p>
              <ul className="text-sm text-gray-600 mb-6 space-y-1 list-disc list-inside">
                <li>{t('settings.confirm_step_fetch')}</li>
                <li>{t('settings.confirm_step_summarize')}</li>
                <li>{t('settings.confirm_step_embed')}</li>
                <li>{t('settings.confirm_step_cluster')}</li>
              </ul>
              <p className="text-xs text-amber-600 bg-amber-50 p-2 rounded mb-6">{t('settings.confirm_warning')}</p>
              <div className="flex justify-end gap-3">
                <button
                  onClick={() => setShowConfirmDialog(false)}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
                >
                  {t('common.cancel')}
                </button>
                <button
                  onClick={handleFullRefresh}
                  className="px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-md transition-colors"
                >
                  {t('settings.execute_full_refresh')}
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      <SyncProgress
        isOpen={showSyncProgress}
        onClose={handleSyncProgressClose}
        steps={translatedSteps}
        title={t('sync.title', 'Syncing Data')}
        canClose={!syncing}
      />
    </div>
  );
};

export default Settings;
