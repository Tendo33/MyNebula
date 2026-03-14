import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { isAxiosError } from 'axios';
import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import {
  AlertTriangle,
  Clock,
  Database,
  Eye,
  Link2,
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
import { formatLastRunTime, formatNextRunTime, getStatusDisplay } from '../utils/scheduleFormat';
import { getAdminAuthConfig } from '../api/auth';

interface TaskCompletionResult {
  success: boolean;
  error: string | null;
}

const createPipelineSyncSteps = (): { id: string; status: SyncStepStatus; progress?: number; error?: string }[] => ([
  { id: 'stars', status: 'pending', progress: 0 },
  { id: 'embeddings', status: 'pending', progress: 0 },
  { id: 'clustering', status: 'pending', progress: 0 },
  { id: 'snapshot', status: 'pending', progress: 0 },
]);

const createFullRefreshSteps = (): { id: string; status: SyncStepStatus; progress?: number; error?: string }[] => ([
  { id: 'reset', status: 'pending', progress: 0 },
  { id: 'stars', status: 'pending', progress: 0 },
  { id: 'embeddings', status: 'pending', progress: 0 },
  { id: 'clustering', status: 'pending', progress: 0 },
  { id: 'snapshot', status: 'pending', progress: 0 },
]);

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
  const [adminAuthConfigured, setAdminAuthConfigured] = useState<boolean | null>(null);
  const [progressTitle, setProgressTitle] = useState('');

  const [showSyncProgress, setShowSyncProgress] = useState(false);
  const [syncSteps, setSyncSteps] = useState<
    { id: string; status: SyncStepStatus; progress?: number; error?: string }[]
  >(createPipelineSyncSteps);
  const graphDefaultsRef = useRef<{ max: number; min: number } | null>(null);

  const translatedSteps = useMemo(
    () =>
      syncSteps.map((step) => ({
        ...step,
        label: t(
          `sync.step_${step.id}_label`,
          step.id === 'reset' ? t('settings.confirm_step_fetch') : step.id
        ),
        description: t(
          `sync.step_${step.id}_desc`,
          step.id === 'reset' ? t('settings.full_refresh_desc') : ''
        ),
      })),
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
      const settingsPayload = await getSettingsV2();
      setSchedule(settingsPayload.schedule);
      setSyncInfo(settingsPayload.sync_info);

      updateSettings({
        maxClusters: settingsPayload.graph_defaults.max_clusters,
        minClusters: settingsPayload.graph_defaults.min_clusters,
        relatedMinSemantic: settingsPayload.graph_defaults.related_min_semantic,
        hqRendering: settingsPayload.graph_defaults.hq_rendering,
        showTrajectories: settingsPayload.graph_defaults.show_trajectories,
      });
      graphDefaultsRef.current = {
        max: settingsPayload.graph_defaults.max_clusters,
        min: settingsPayload.graph_defaults.min_clusters,
      };
    } catch (err) {
      setError(t('settings.load_schedule_error'));
      console.error('Failed to load schedule data:', err);
    }
  }, [t, updateSettings]);

  useEffect(() => {
    if (!isAuthenticated) {
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
        console.error('Failed to update graph defaults:', err);
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
    onProgress?: (pipeline: PipelineStatusResponse) => void
  ): Promise<TaskCompletionResult> => {
    while (true) {
      try {
        const pipeline = await getPipelineStatusV2(runId);
        onProgress?.(pipeline);

        if (pipeline.status === 'completed') {
          return { success: true, error: null };
        }

        if (pipeline.status === 'failed' || pipeline.status === 'partial_failed') {
          return { success: false, error: pipeline.last_error || null };
        }
      } catch (err) {
        return {
          success: false,
          error: err instanceof Error ? err.message : 'poll_pipeline_status_failed',
        };
      }

      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  };

  const waitForFullRefreshJobComplete = async (
    taskId: number,
    onProgress?: (job: FullRefreshJobStatus) => void
  ): Promise<TaskCompletionResult> => {
    while (true) {
      try {
        const jobResponse = await getFullRefreshJobStatusV2(taskId);
        onProgress?.(jobResponse.job);

        if (jobResponse.job.status === 'completed') {
          return { success: true, error: null };
        }

        if (jobResponse.job.status === 'failed') {
          return { success: false, error: jobResponse.job.last_error };
        }
      } catch (err) {
        return {
          success: false,
          error: err instanceof Error ? err.message : 'poll_job_status_failed',
        };
      }

      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  };

  const resetSteps = () => {
    setSyncSteps(createPipelineSyncSteps());
  };

  const resetFullRefreshStepState = () => {
    setSyncSteps(createFullRefreshSteps());
  };

  const normalizePipelinePhase = (
    phase: string
  ): 'stars' | 'embeddings' | 'clustering' | 'snapshot' | null => {
    switch (phase) {
      case 'stars':
      case 'star':
        return 'stars';
      case 'embedding':
      case 'embeddings':
        return 'embeddings';
      case 'cluster':
      case 'clustering':
        return 'clustering';
      case 'snapshot':
        return 'snapshot';
      default:
        return null;
    }
  };

  const updatePipelineProgress = (pipeline: PipelineStatusResponse) => {
    const phaseOrder = ['stars', 'embeddings', 'clustering', 'snapshot'] as const;
    const normalizedPhase = normalizePipelinePhase(pipeline.phase);
    const phaseIndex =
      normalizedPhase !== null ? phaseOrder.findIndex((phase) => phase === normalizedPhase) : -1;

    if (phaseIndex >= 0) {
      setSyncStep(t(`sync.step_${phaseOrder[phaseIndex]}_label`, phaseOrder[phaseIndex]));
    }

    setSyncSteps((previous) =>
      previous.map((step, index) => {
        if (pipeline.status === 'completed') {
          return { ...step, status: 'completed', progress: 100, error: undefined };
        }

        if (pipeline.status === 'failed' || pipeline.status === 'partial_failed') {
          if (phaseIndex >= 0 && index < phaseIndex) {
            return { ...step, status: 'completed', progress: 100, error: undefined };
          }
          if (
            step.id === normalizedPhase ||
            (normalizedPhase === null && index === previous.length - 1)
          ) {
            return {
              ...step,
              status: 'failed',
              error: pipeline.last_error || undefined,
            };
          }
          return { ...step, status: 'pending', progress: 0, error: undefined };
        }

        if (phaseIndex >= 0 && index < phaseIndex) {
          return { ...step, status: 'completed', progress: 100, error: undefined };
        }
        if (phaseIndex >= 0 && index === phaseIndex) {
          return {
            ...step,
            status: 'running',
            progress: 40,
            error: undefined,
          };
        }
        return { ...step, status: 'pending', progress: 0, error: undefined };
      })
    );
  };

  const normalizeFullRefreshPhase = (
    phase: string
  ): 'reset' | 'stars' | 'embeddings' | 'clustering' | 'snapshot' | null => {
    switch (phase) {
      case 'reset':
        return 'reset';
      case 'stars':
      case 'star':
        return 'stars';
      case 'embedding':
      case 'embeddings':
      case 'summary':
      case 'summaries':
        return 'embeddings';
      case 'cluster':
      case 'clustering':
        return 'clustering';
      case 'snapshot':
      case 'complete':
      case 'completed':
        return 'snapshot';
      case 'full_refresh':
      default:
        return null;
    }
  };

  const updateFullRefreshProgress = (job: FullRefreshJobStatus) => {
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

    setSyncSteps((previous) =>
      previous.map((step, index) => {
        if (job.status === 'completed') {
          return { ...step, status: 'completed', progress: 100, error: undefined };
        }

        if (job.status === 'failed') {
          if (currentIndex >= 0 && index < currentIndex) {
            return { ...step, status: 'completed', progress: 100, error: undefined };
          }
          if (
            step.id === normalizedPhase ||
            (normalizedPhase === null && currentIndex >= 0 && step.id === phaseOrder[currentIndex]) ||
            (currentIndex < 0 && index === previous.length - 1)
          ) {
            return {
              ...step,
              status: 'failed',
              error: job.last_error || undefined,
            };
          }
          return { ...step, status: 'pending', progress: 0, error: undefined };
        }

        if (currentIndex >= 0 && index < currentIndex) {
          return { ...step, status: 'completed', progress: 100, error: undefined };
        }
        if (currentIndex >= 0 && index === currentIndex) {
          const perStepProgress = Math.max(
            0,
            Math.min(
              100,
              ((job.progress_percent - currentIndex * stepProgressSpan) / stepProgressSpan) * 100
            )
          );
          return {
            ...step,
            status: 'running',
            progress: Math.round(perStepProgress),
            error: undefined,
          };
        }
        return { ...step, status: 'pending', progress: 0, error: undefined };
      })
    );
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
      const updated = await updateScheduleV2(newConfig);
      setSchedule(updated.schedule);
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
      const updated = await updateScheduleV2(newConfig);
      setSchedule(updated.schedule);
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
      setProgressTitle(t('sync.title', 'Syncing Data'));
      resetSteps();

      const started = await startSyncPipelineV2({
        mode: 'incremental',
        use_llm: true,
        max_clusters: settings.maxClusters,
        min_clusters: settings.minClusters,
      });
      const pipelineResult = await waitForPipelineComplete(
        started.pipeline_run_id,
        updatePipelineProgress
      );
      if (!pipelineResult.success) {
        throw new Error(pipelineResult.error || t('errors.sync_failed'));
      }

      setSyncSteps((prev) =>
        prev.map((step) => ({ ...step, status: 'completed', progress: 100, error: undefined }))
      );

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
      const started = await startReclusterV2({
        max_clusters: settings.maxClusters,
        min_clusters: settings.minClusters,
      });
      const pipelineResult = await waitForPipelineComplete(started.pipeline_run_id);
      if (!pipelineResult.success) {
        throw new Error(pipelineResult.error || t('errors.clustering_failed'));
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
    setError(null);
    setSyncing(true);
    setShowSyncProgress(true);
    setProgressTitle(t('settings.full_refresh'));
    resetFullRefreshStepState();

    try {
      const started = await triggerFullRefreshV2();
      const fullRefreshStatus = await waitForFullRefreshJobComplete(
        started.task.task_id,
        updateFullRefreshProgress
      );
      if (!fullRefreshStatus.success) {
        throw new Error(fullRefreshStatus.error || t('settings.full_refresh_error'));
      }

      setSyncSteps((prev) =>
        prev.map((step) => ({ ...step, status: 'completed', progress: 100, error: undefined }))
      );

      await refreshData();
      await loadScheduleData();
    } catch (err) {
      setError(t('settings.full_refresh_error'));
      console.error('Failed to trigger full refresh:', err);
    } finally {
      setRefreshLoading(false);
      setSyncing(false);
      setSyncStep('');
    }
  };

  const handleSyncProgressClose = () => {
    setShowSyncProgress(false);
    if (!syncing) {
      resetSteps();
    }
  };

  return (
    <div className="flex min-h-screen bg-bg-main text-text-main dark:bg-dark-bg-main dark:text-dark-text-main">
      <Sidebar />
      <main className="flex-1 flex flex-col min-w-0" style={{ marginLeft: 'var(--sidebar-width, 240px)' }}>
        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between min-h-[3.5rem] px-4 sm:px-8 py-3 sm:py-0 border-b border-border-light sticky top-0 bg-bg-main/95 backdrop-blur-sm z-40 dark:bg-dark-bg-main/95 dark:border-dark-border">
          <h1 className="text-base font-semibold text-text-main select-none tracking-tight">
            {t('settings.title')}
          </h1>
          <div className="flex items-center gap-3">
            <LanguageSwitch />
            {isAuthenticated && (
              <button
                onClick={handleAdminLogout}
                className="h-8 px-3 rounded-md text-xs font-medium border border-border-light bg-bg-main hover:bg-bg-hover text-text-main transition-colors inline-flex items-center gap-1.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30 dark:bg-dark-bg-main dark:border-dark-border dark:text-dark-text-main dark:hover:bg-dark-bg-sidebar/70"
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
            <div className="w-full max-w-md bg-bg-main border border-border-light rounded-xl shadow-sm p-6 dark:bg-dark-bg-main dark:border-dark-border">
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
                {adminAuthConfigured === false && (
                  <div className="text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded-md px-3 py-2">
                    {t('settings.admin_not_configured')}
                  </div>
                )}

                <div>
                  <label htmlFor="admin-username" className="block text-xs text-text-muted mb-1">
                    {t('settings.username')}
                  </label>
                  <div className="relative">
                    <User className="w-4 h-4 text-text-dim absolute left-3 top-1/2 -translate-y-1/2" />
                    <input
                      id="admin-username"
                      type="text"
                      value={loginUsername}
                      onChange={(e) => setLoginUsername(e.target.value)}
                      className="w-full h-10 pl-9 pr-3 rounded-md border border-border-light text-sm bg-bg-main focus:outline-none focus:ring-1 focus:ring-text-main/30 dark:bg-dark-bg-main dark:border-dark-border dark:text-dark-text-main"
                      autoComplete="username"
                      required
                    />
                  </div>
                </div>

                <div>
                  <label htmlFor="admin-password" className="block text-xs text-text-muted mb-1">
                    {t('settings.password')}
                  </label>
                  <input
                    id="admin-password"
                    type="password"
                    value={loginPassword}
                    onChange={(e) => setLoginPassword(e.target.value)}
                    className="w-full h-10 px-3 rounded-md border border-border-light text-sm bg-bg-main focus:outline-none focus:ring-1 focus:ring-text-main/30 dark:bg-dark-bg-main dark:border-dark-border dark:text-dark-text-main"
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
                  disabled={loginLoading || adminAuthConfigured === false}
                  className={clsx(
                    'w-full h-10 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2',
                    loginLoading || adminAuthConfigured === false
                      ? 'bg-bg-hover text-text-dim border border-border-light cursor-not-allowed dark:bg-dark-bg-sidebar/70 dark:text-dark-text-main/60 dark:border-dark-border'
                      : 'bg-text-main text-bg-main hover:bg-text-main/90'
                  )}
                >
                  {loginLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                  {t('app.login')}
                </button>
              </form>
            </div>
          </section>
        ) : (
          <div className="max-w-3xl px-4 sm:px-8 py-6 sm:py-10 space-y-12">
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
                <div className="flex items-center justify-between p-3 rounded-md hover:bg-bg-hover transition-all group">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-bg-main transition-colors dark:group-hover:bg-dark-bg-main">
                      <Zap className="w-5 h-5 text-text-muted group-hover:text-text-main" />
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-text-main">{t('settings.hq_rendering')}</span>
                      <span className="text-xs text-text-muted">{t('settings.hq_rendering_desc')}</span>
                    </div>
                  </div>
                  <button
                    className={clsx(
                      'relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30',
                      settings.hqRendering ? 'bg-text-main' : 'bg-border-light dark:bg-dark-border'
                    )}
                    type="button"
                    role="switch"
                    aria-checked={settings.hqRendering}
                    aria-label={t('settings.hq_rendering')}
                    onClick={() => updateSettings({ hqRendering: !settings.hqRendering })}
                  >
                    <span
                      className={clsx(
                        'inline-block h-5 w-5 transform rounded-full bg-bg-main transition duration-200 ease-in-out shadow-sm dark:bg-dark-bg-main',
                        settings.hqRendering ? 'translate-x-5' : 'translate-x-0.5'
                      )}
                    />
                  </button>
                </div>

                <div className="flex items-center justify-between p-3 rounded-md hover:bg-bg-hover transition-all group">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-bg-main transition-colors dark:group-hover:bg-dark-bg-main">
                      <Eye className="w-5 h-5 text-text-muted group-hover:text-text-main" />
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-text-main">{t('settings.show_trajectories')}</span>
                      <span className="text-xs text-text-muted">{t('settings.show_trajectories_desc')}</span>
                    </div>
                  </div>
                  <button
                    className={clsx(
                      'relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30',
                      settings.showTrajectories ? 'bg-text-main' : 'bg-border-light dark:bg-dark-border'
                    )}
                    type="button"
                    role="switch"
                    aria-checked={settings.showTrajectories}
                    aria-label={t('settings.show_trajectories')}
                    onClick={() => updateSettings({ showTrajectories: !settings.showTrajectories })}
                  >
                    <span
                      className={clsx(
                        'inline-block h-5 w-5 transform rounded-full bg-bg-main transition duration-200 ease-in-out shadow-sm dark:bg-dark-bg-main',
                        settings.showTrajectories ? 'translate-x-5' : 'translate-x-0.5'
                      )}
                    />
                  </button>
                </div>

                <div className="p-3 border border-border-light rounded-md space-y-2">
                  <div className="flex items-center gap-2">
                    <Link2 className="w-4 h-4 text-text-muted" />
                    <span className="text-sm font-medium text-text-main">
                      {t('settings.related_min_semantic')}
                    </span>
                  </div>
                  <p className="text-xs text-text-muted">{t('settings.related_min_semantic_desc')}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-muted">{t('repoDetails.similar', 'Similar')}</span>
                    <span className="text-xs font-mono tabular-nums text-text-dim">
                      {settings.relatedMinSemantic.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.5}
                    max={0.9}
                    step={0.01}
                    value={settings.relatedMinSemantic}
                    onChange={(e) => {
                      const nextMinSemantic = Number(e.target.value);
                      updateSettings({ relatedMinSemantic: nextMinSemantic });
                    }}
                    className="w-full"
                  />
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
                      'h-9 px-4 rounded-md text-sm font-medium transition-all flex items-center gap-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30',
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
                      'h-9 px-4 rounded-md text-sm font-medium border transition-colors inline-flex items-center gap-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30',
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

            <hr className="border-t border-border-light" />

            <section>
              <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-4 px-2 select-none">
                {t('settings.scheduled_sync')}
              </h2>

              <div className="space-y-2">
                <div className="flex items-center justify-between p-3 rounded-md hover:bg-bg-hover transition-all group">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-bg-main transition-colors dark:group-hover:bg-dark-bg-main">
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
                      'relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30',
                      schedule?.is_enabled ? 'bg-text-main' : 'bg-border-light dark:bg-dark-border',
                      scheduleLoading && 'opacity-50 cursor-not-allowed'
                    )}
                    type="button"
                    role="switch"
                    aria-checked={Boolean(schedule?.is_enabled)}
                    aria-label={t('settings.enable_scheduled_sync')}
                    onClick={!scheduleLoading ? handleScheduleToggle : undefined}
                  >
                    {scheduleLoading ? (
                      <Loader2 className="w-4 h-4 text-white mx-auto animate-spin" />
                    ) : (
                      <span
                        className={clsx(
                          'inline-block h-5 w-5 transform rounded-full bg-bg-main transition duration-200 ease-in-out shadow-sm dark:bg-dark-bg-main',
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
                  {syncInfo?.single_user_mode && (
                    <div className="mt-2 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-md px-2 py-1">
                      {t('settings.single_user_mode_notice')}
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
          </div>
        )}

        {showConfirmDialog && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div
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
                  onClick={() => setShowConfirmDialog(false)}
                  className="px-4 py-2 text-sm font-medium text-text-main bg-bg-hover hover:bg-border-light rounded-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30 dark:text-dark-text-main dark:bg-dark-bg-sidebar/70 dark:hover:bg-dark-border"
                >
                  {t('common.cancel')}
                </button>
                <button
                  onClick={handleFullRefresh}
                  className="px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30"
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
        title={progressTitle || t('sync.title', 'Syncing Data')}
        canClose={!syncing}
      />
    </div>
  );
};

export default Settings;
