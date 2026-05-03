import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { isAxiosError } from 'axios';
import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import {
  AlertTriangle,
  Loader2,
  LogOut,
  RefreshCw,
  Server,
  Shield,
  Sparkles,
} from 'lucide-react';

import { Sidebar } from '../components/layout/Sidebar';
import { LanguageSwitch } from '../components/layout/LanguageSwitch';
import { SyncProgress, SyncStepStatus } from '../components/ui/SyncProgress';
import { useGraph } from '../contexts/GraphContext';
import { useAdminAuth } from '../contexts/AdminAuthContext';
import { SettingsLoginForm, SettingsAppearance, SettingsSchedule, SettingsDataSection } from './settings/index';
import {
  buildTranslatedSteps,
  createFullRefreshSteps,
  createPipelineSyncSteps,
  finalizeCompletedSteps,
  mapFullRefreshProgressSteps,
  mapPipelineProgressSteps,
  normalizeFullRefreshPhase,
  normalizePipelinePhase,
} from './settings/progress';
import { API_BASE_URL } from '../api/client';
import {
  getPipelineStatusV2,
  startReclusterV2,
  startSyncPipelineV2,
  type PipelineStatusResponse,
} from '../api/v2/sync';
import {
  getFullRefreshJobStatusV2,
  getSettingsV2,
  triggerFullRefreshV2,
  updateGraphDefaultsV2,
  updateScheduleV2,
  type FullRefreshJobStatus,
  type ScheduleConfig,
  type ScheduleResponse,
  type SyncInfoResponse,
} from '../api/v2/settings';
import { getAdminAuthConfig } from '../api/auth';
import { pollUntilComplete, type TaskCompletionResult } from './settings/polling';
import { logClientError } from '../utils/debug';

const isPartialFailureResult = (result: TaskCompletionResult): boolean =>
  !result.success &&
  !result.cancelled &&
  typeof result.error === 'string' &&
  result.error.includes('partial failure');

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
  const [warning, setWarning] = useState<string | null>(null);

  const [loginUsername, setLoginUsername] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);
  const [adminAuthConfigured, setAdminAuthConfigured] = useState<boolean | null>(null);
  const [progressTitle, setProgressTitle] = useState('');

  const [showSyncProgress, setShowSyncProgress] = useState(false);
  const [syncSteps, setSyncSteps] = useState<
    { id: string; status: SyncStepStatus; progress?: number; error?: string }[]
  >(createPipelineSyncSteps);
  const graphDefaultsRef = useRef<{ max: number; min: number } | null>(null);
  const activePollControllerRef = useRef<AbortController | null>(null);
  const latestFullRefreshJobRef = useRef<FullRefreshJobStatus | null>(null);

  const translatedSteps = useMemo(
    () => buildTranslatedSteps(syncSteps, t),
    [syncSteps, t]
  );

  useEffect(() => {
    setProgressTitle(t('sync.title', 'Syncing Data'));
  }, [t]);

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
      setWarning(null);
      const settingsPayload = await getSettingsV2();
      setSchedule(settingsPayload.schedule);
      setSyncInfo(settingsPayload.sync_info);

      updateSettings({
        maxClusters: settingsPayload.graph_defaults.max_clusters,
        minClusters: settingsPayload.graph_defaults.min_clusters,
      });
      graphDefaultsRef.current = {
        max: settingsPayload.graph_defaults.max_clusters,
        min: settingsPayload.graph_defaults.min_clusters,
      };
    } catch (err) {
      setError(t('settings.load_schedule_error'));
      logClientError('Failed to load schedule data:', err);
    }
  }, [t, updateSettings]);

  useEffect(() => {
    if (!isAuthenticated) {
      activePollControllerRef.current?.abort();
      activePollControllerRef.current = null;
      setSchedule(null);
      setSyncInfo(null);
      return;
    }

    loadScheduleData();
  }, [isAuthenticated, loadScheduleData]);

  useEffect(() => {
    if (!isAuthenticated) return;
    const current = { max: settings.maxClusters, min: settings.minClusters };
    const previous = graphDefaultsRef.current;
    if (previous && previous.max === current.max && previous.min === current.min) {
      return;
    }

    const timer = window.setTimeout(async () => {
      try {
        await updateGraphDefaultsV2({
          max_clusters: current.max,
          min_clusters: current.min,
        });
        graphDefaultsRef.current = current;
      } catch (err) {
        setError(t('settings.update_graph_defaults_error', 'Failed to save graph defaults'));
        logClientError('Failed to update graph defaults:', err);
      }
    }, 500);

    return () => window.clearTimeout(timer);
  }, [isAuthenticated, settings.maxClusters, settings.minClusters, t]);

  useEffect(() => {
    if (isAuthenticated) {
      setAdminAuthConfigured(true);
      return;
    }

    getAdminAuthConfig()
      .then((config) => setAdminAuthConfigured(config.enabled))
      .catch(() => setAdminAuthConfigured(null));
  }, [isAuthenticated]);

  useEffect(() => {
    if (!showConfirmDialog) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setShowConfirmDialog(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showConfirmDialog]);

  const waitForPipelineComplete = async (
    runId: number,
    signal: AbortSignal,
    onProgress?: (pipeline: PipelineStatusResponse) => void
  ): Promise<TaskCompletionResult> => {
    return pollUntilComplete({
      signal,
      poll: () => getPipelineStatusV2(runId),
      onProgress,
      isSuccess: (pipeline) => pipeline.status === 'completed',
      isFailure: (pipeline) =>
        pipeline.status === 'failed' || pipeline.status === 'partial_failed',
      getFailureError: (pipeline) => pipeline.last_error || null,
      getPollError: (err) =>
        err instanceof Error ? err.message : 'poll_pipeline_status_failed',
    });
  };

  const waitForFullRefreshJobComplete = async (
    taskId: number,
    signal: AbortSignal,
    onProgress?: (job: FullRefreshJobStatus) => void
  ): Promise<TaskCompletionResult> => {
    return pollUntilComplete({
      signal,
      poll: async () => (await getFullRefreshJobStatusV2(taskId)).job,
      onProgress,
      isSuccess: (job) => job.status === 'completed' || job.status === 'partial_failed',
      isFailure: (job) => job.status === 'failed',
      getFailureError: (job) => job.last_error,
      getPollError: (err) => (err instanceof Error ? err.message : 'poll_job_status_failed'),
    });
  };

  const beginPollingOperation = useCallback(() => {
    activePollControllerRef.current?.abort();
    const controller = new AbortController();
    activePollControllerRef.current = controller;
    return controller;
  }, []);

  const finishPollingOperation = useCallback((controller: AbortController) => {
    if (activePollControllerRef.current === controller) {
      activePollControllerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      activePollControllerRef.current?.abort();
      activePollControllerRef.current = null;
    };
  }, []);

  const resetSteps = () => {
    setSyncSteps(createPipelineSyncSteps());
  };

  const resetFullRefreshStepState = () => {
    setSyncSteps(createFullRefreshSteps());
  };

  const updatePipelineProgress = (pipeline: PipelineStatusResponse) => {
    const phaseOrder = ['stars', 'embeddings', 'clustering', 'snapshot'] as const;
    const normalizedPhase = normalizePipelinePhase(pipeline.phase);
    const phaseIndex =
      normalizedPhase !== null ? phaseOrder.findIndex((phase) => phase === normalizedPhase) : -1;

    if (phaseIndex >= 0) {
      setSyncStep(t(`sync.step_${phaseOrder[phaseIndex]}_label`, phaseOrder[phaseIndex]));
    }

    setSyncSteps((previous) => mapPipelineProgressSteps(previous, pipeline));
  };

  const updateFullRefreshProgress = (job: FullRefreshJobStatus) => {
    latestFullRefreshJobRef.current = job;
    const phaseOrder = ['reset', 'stars', 'embeddings', 'clustering', 'snapshot'] as const;
    const normalizedPhase = normalizeFullRefreshPhase(job.phase);
    const stepProgressSpan = 100 / phaseOrder.length;
    const phaseIndex =
      normalizedPhase !== null ? phaseOrder.findIndex((phase) => phase === normalizedPhase) : -1;
    const inferredIndex = Math.min(
      phaseOrder.length - 1,
      Math.max(
        0,
        Math.floor(Math.max(0, Math.min(99.999, job.progress_percent)) / stepProgressSpan)
      )
    );
    const currentIndex =
      job.status === 'running' || job.status === 'pending'
        ? Math.max(phaseIndex, inferredIndex)
        : phaseIndex;

    if (currentIndex >= 0) {
      setSyncStep(t(`sync.step_${phaseOrder[currentIndex]}_label`, phaseOrder[currentIndex]));
    }

    setSyncSteps((previous) => mapFullRefreshProgressSteps(previous, job, t));
  };

  const handleAdminLogin = async (e: FormEvent) => {
    e.preventDefault();
    if (adminAuthConfigured === false) {
      setLoginError(t('settings.admin_not_configured'));
      return;
    }
    setLoginLoading(true);
    setLoginError(null);

    try {
      await login(loginUsername.trim(), loginPassword);
      setLoginPassword('');
      await loadScheduleData();
    } catch (err) {
      if (isAxiosError(err) && err.response?.status === 503) {
        setLoginError(t('settings.admin_not_configured'));
      } else {
        setLoginError(t('settings.login_failed'));
      }
      logClientError('Admin login failed:', err);
    } finally {
      setLoginLoading(false);
    }
  };

  const handleAdminLogout = async () => {
    activePollControllerRef.current?.abort();
    activePollControllerRef.current = null;
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
      const updated = await updateScheduleV2(newConfig);
      setSchedule(updated.schedule);
    } catch (err) {
      setError(t('settings.update_schedule_error'));
      logClientError('Failed to update schedule:', err);
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
      const updated = await updateScheduleV2(newConfig);
      setSchedule(updated.schedule);
    } catch (err) {
      setError(t('settings.update_time_error'));
      logClientError('Failed to update schedule time:', err);
    } finally {
      setScheduleLoading(false);
    }
  };

  const handleSyncStars = async () => {
    try {
      setError(null);
      setSyncing(true);
      setShowSyncProgress(true);
      setProgressTitle(t('sync.title', 'Syncing Data'));
      resetSteps();

      const started = await startSyncPipelineV2({
        mode: 'incremental',
        use_llm: true,
        max_clusters: settings.maxClusters,
        min_clusters: settings.minClusters,
      });
      const controller = beginPollingOperation();
      const pipelineResult = await waitForPipelineComplete(
        started.pipeline_run_id,
        controller.signal,
        updatePipelineProgress
      );
      finishPollingOperation(controller);
      if (pipelineResult.cancelled) {
        return;
      }
      if (isPartialFailureResult(pipelineResult)) {
        await refreshData();
        await loadScheduleData();
        setWarning(
          t(
            'sync.partial_failed_warning',
            'Sync completed with warnings. Check the latest run details before retrying.'
          )
        );
        return;
      }
      if (!pipelineResult.success) {
        throw new Error(pipelineResult.error || t('errors.sync_failed'));
      }

      setSyncSteps((prev) =>
        prev.map((step) =>
          step.status === 'warning'
            ? { ...step, progress: 100 }
            : { ...step, status: 'completed', progress: 100, error: undefined }
        )
      );

      await refreshData();
      await loadScheduleData();
    } catch (err) {
      setError(t('errors.sync_failed'));
      logClientError('Sync failed:', err);
    } finally {
      activePollControllerRef.current?.abort();
      setSyncing(false);
      setSyncStep('');
    }
  };

  const handleRecluster = async () => {
    setWarning(null);
    setReclusterLoading(true);
    setError(null);
    setSyncing(true);
    setShowSyncProgress(true);
    setProgressTitle(t('graph.recluster'));
    resetSteps();

    try {
      const started = await startReclusterV2({
        max_clusters: settings.maxClusters,
        min_clusters: settings.minClusters,
      });
      const controller = beginPollingOperation();
      const pipelineResult = await waitForPipelineComplete(
        started.pipeline_run_id,
        controller.signal,
        updatePipelineProgress
      );
      finishPollingOperation(controller);
      if (pipelineResult.cancelled) {
        return;
      }
      if (isPartialFailureResult(pipelineResult)) {
        await refreshData();
        await loadScheduleData();
        setWarning(
          t(
            'graph.recluster_partial_failed',
            'Re-cluster completed with warnings. Review the latest pipeline details.'
          )
        );
        return;
      }
      if (!pipelineResult.success) {
        throw new Error(pipelineResult.error || t('errors.clustering_failed'));
      }

      await refreshData();
      await loadScheduleData();
    } catch (err) {
      setError(t('errors.clustering_failed'));
      logClientError('Re-cluster failed:', err);
    } finally {
      activePollControllerRef.current?.abort();
      setSyncing(false);
      setSyncStep('');
      setReclusterLoading(false);
    }
  };

  const handleFullRefresh = async () => {
    setShowConfirmDialog(false);
    activePollControllerRef.current?.abort();
    latestFullRefreshJobRef.current = null;
    setRefreshLoading(true);
    setError(null);
    setWarning(null);
    setSyncing(true);
    setShowSyncProgress(true);
    setProgressTitle(t('settings.full_refresh'));
    resetFullRefreshStepState();

    try {
      const started = await triggerFullRefreshV2();
      const controller = beginPollingOperation();
      const fullRefreshStatus = await waitForFullRefreshJobComplete(
        started.task.task_id,
        controller.signal,
        updateFullRefreshProgress
      );
      finishPollingOperation(controller);
      if (fullRefreshStatus.cancelled) {
        return;
      }
      if (!fullRefreshStatus.success) {
        throw new Error(fullRefreshStatus.error || t('settings.full_refresh_error'));
      }

      const latestJob = latestFullRefreshJobRef.current as FullRefreshJobStatus | null;
      const partialFailures =
        (latestJob?.error_details?.partial_failures ?? []) as Array<{
          phase: string;
          task_id: number;
          failed_items: number;
        }>;
      if (latestJob?.status === 'partial_failed') {
        const failedItems = partialFailures.reduce(
          (sum: number, entry) => sum + (entry.failed_items ?? 0),
          0,
        );
        const warningMessage = t(
          'settings.full_refresh_partial_failed',
          `Full refresh completed with warnings (${partialFailures.length} stages, ${failedItems} failed items).`
        );

        setSyncSteps((prev) => finalizeCompletedSteps(prev, partialFailures, t));

        await refreshData();
        await loadScheduleData();
        setWarning(warningMessage);
        return;
      }

      setSyncSteps((prev) => finalizeCompletedSteps(prev, partialFailures, t));

      await refreshData();
      await loadScheduleData();
    } catch (err) {
      setError(t('settings.full_refresh_error'));
      logClientError('Failed to trigger full refresh:', err);
    } finally {
      setRefreshLoading(false);
      setSyncing(false);
      setSyncStep('');
    }
  };

  const handleSyncProgressClose = () => {
    activePollControllerRef.current?.abort();
    activePollControllerRef.current = null;
    setShowSyncProgress(false);
    if (!syncing) {
      resetSteps();
    }
  };

  return (
    <div className="page-shell">
      <Sidebar />
      <main className="page-main">
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
                onClick={handleAdminLogout}
                className="header-action min-h-0 px-3 text-xs"
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
            loginUsername={loginUsername}
            loginPassword={loginPassword}
            loginLoading={loginLoading}
            loginError={loginError}
            adminAuthConfigured={adminAuthConfigured}
            onUsernameChange={setLoginUsername}
            onPasswordChange={setLoginPassword}
            onSubmit={handleAdminLogin}
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
              <h2 className="section-kicker mb-4 select-none">
                {t('settings.operations')}
              </h2>
              <div className="space-y-4">
                <div className="panel-surface p-5">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-text-muted" />
                      <span className="text-sm font-medium text-text-main">{t('dashboard.sync_button')}</span>
                    </div>
                  <button
                    onClick={handleSyncStars}
                    disabled={syncing || refreshLoading || reclusterLoading}
                    className={clsx(
                      'inline-flex min-h-[2.75rem] items-center gap-2 rounded-xl px-4 text-sm font-medium transition-all',
                      syncing || refreshLoading || reclusterLoading
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
                      'inline-flex min-h-[2.75rem] items-center gap-2 rounded-xl border px-4 text-sm font-medium transition-colors',
                      reclusterLoading || syncing || refreshLoading
                        ? 'bg-bg-hover text-text-dim border-border-light cursor-not-allowed dark:bg-dark-bg-sidebar/70 dark:text-dark-text-main/60 dark:border-dark-border'
                        : 'bg-bg-main text-text-main border-border-light hover:bg-bg-hover dark:bg-dark-bg-main dark:text-dark-text-main dark:border-dark-border dark:hover:bg-dark-bg-sidebar/70'
                    )}
                    title={t('graph.recluster_hint')}
                  >
                    {reclusterLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                    {reclusterLoading ? t('graph.reclustering') : t('graph.recluster')}
                  </button>
                </div>
              </div>
            </section>

            <hr className="border-t border-border-light/80" />

            <section>
              <h2 className="section-kicker mb-4 select-none">
                {t('settings.connection')}
              </h2>
              <div className="space-y-2">
                <div className="panel-subtle p-4">
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

                <div className="panel-subtle flex items-center justify-between p-4 transition-colors">
                  <div className="flex items-center gap-2">
                    <Shield className="w-4 h-4 text-text-muted" />
                    <label className="text-sm font-medium text-text-main">{t('settings.github_token_status')}</label>
                  </div>
                  <div
                    className={clsx(
                      'flex items-center gap-2 text-sm font-medium px-3 py-1 rounded-full border',
                      githubTokenStatus.state === 'connected' && 'text-green-700 bg-green-50 border-green-200',
                      githubTokenStatus.state === 'not_configured' && 'text-amber-700 bg-amber-50 border-amber-200',
                      githubTokenStatus.state === 'unknown' && 'text-text-muted bg-bg-hover border-border-light dark:text-dark-text-main/70 dark:bg-dark-bg-sidebar/70 dark:border-dark-border',
                      githubTokenStatus.state === 'loading' && 'text-text-dim bg-bg-hover border-border-light dark:text-dark-text-main/60 dark:bg-dark-bg-sidebar/60 dark:border-dark-border'
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
              schedule={schedule}
              scheduleLoading={scheduleLoading}
              onToggle={handleScheduleToggle}
              onTimeChange={handleTimeChange}
            />

            <hr className="border-t border-border-light/80" />

            <SettingsDataSection
              syncInfo={syncInfo}
              refreshLoading={refreshLoading}
              syncing={syncing}
              reclusterLoading={reclusterLoading}
              showConfirmDialog={showConfirmDialog}
              onShowConfirm={() => setShowConfirmDialog(true)}
              onHideConfirm={() => setShowConfirmDialog(false)}
              onConfirmRefresh={handleFullRefresh}
            />
            </div>
          </div>
        )}
      </main>

      <SyncProgress
        isOpen={showSyncProgress}
        onClose={handleSyncProgressClose}
        steps={translatedSteps}
        title={progressTitle || t('sync.title', 'Syncing Data')}
        canClose={!syncing}
      />
    </div>
  );
};

export default Settings;
