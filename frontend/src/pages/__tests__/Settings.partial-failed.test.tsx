import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';

const refreshData = vi.fn();
const updateSettings = vi.fn();
const setSyncing = vi.fn();
const setSyncStep = vi.fn();
const loadSettings = vi.fn();
const triggerFullRefreshV2 = vi.fn();
const startSyncPipelineV2 = vi.fn();
const startReclusterV2 = vi.fn();
const pollUntilComplete = vi.fn();

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key,
  }),
}));

vi.mock('../../contexts/GraphContext', () => ({
  useGraph: () => ({
    settings: {
      maxClusters: 8,
      minClusters: 3,
      relatedMinSemantic: 0.65,
      hqRendering: true,
      showTrajectories: true,
    },
    updateSettings,
    refreshData,
    syncing: false,
    setSyncing,
    setSyncStep,
  }),
}));

vi.mock('../../contexts/AdminAuthContext', () => ({
  useAdminAuth: () => ({
    isChecking: false,
    isAuthenticated: true,
    login: vi.fn(),
    logout: vi.fn(),
  }),
}));

vi.mock('../../components/layout/Sidebar', () => ({
  Sidebar: () => <div data-testid="sidebar" />,
}));

vi.mock('../../components/layout/LanguageSwitch', () => ({
  LanguageSwitch: () => <div data-testid="lang-switch" />,
}));

vi.mock('../../components/ui/SyncProgress', () => ({
  SyncProgress: ({ steps }: { steps: Array<{ id: string; status: string }> }) => (
    <div data-testid="sync-progress">
      {steps.map((step) => (
        <div key={step.id} data-testid={`step-${step.id}`}>
          {step.status}
        </div>
      ))}
    </div>
  ),
}));

vi.mock('../settings/index', () => ({
  SettingsLoginForm: () => null,
  SettingsAppearance: () => <div data-testid="appearance" />,
  SettingsSchedule: () => <div data-testid="schedule" />,
  SettingsDataSection: ({ onConfirmRefresh }: { onConfirmRefresh: () => void }) => (
    <button onClick={onConfirmRefresh}>trigger refresh</button>
  ),
}));

vi.mock('../../api/v2/settings', () => ({
  getSettingsV2: (...args: unknown[]) => loadSettings(...args),
  triggerFullRefreshV2: (...args: unknown[]) => triggerFullRefreshV2(...args),
  getFullRefreshJobStatusV2: vi.fn(),
  updateGraphDefaultsV2: vi.fn(),
  updateScheduleV2: vi.fn(),
}));

vi.mock('../../api/v2/sync', () => ({
  getPipelineStatusV2: vi.fn(),
  startReclusterV2: (...args: unknown[]) => startReclusterV2(...args),
  startSyncPipelineV2: (...args: unknown[]) => startSyncPipelineV2(...args),
}));

vi.mock('../../api/auth', () => ({
  getAdminAuthConfig: vi.fn(),
}));

vi.mock('../settings/polling', () => ({
  pollUntilComplete: (...args: unknown[]) => pollUntilComplete(...args),
}));

import Settings from '../Settings';

describe('Settings partial failed warning', () => {
  beforeEach(() => {
    refreshData.mockReset();
    loadSettings.mockReset();
    triggerFullRefreshV2.mockReset();
    startSyncPipelineV2.mockReset();
    startReclusterV2.mockReset();
    pollUntilComplete.mockReset();
    loadSettings.mockResolvedValue({
      schedule: {
        is_enabled: false,
        schedule_hour: 9,
        schedule_minute: 0,
        timezone: 'Asia/Shanghai',
        last_run_at: null,
        last_run_status: null,
        last_run_error: null,
        next_run_at: null,
      },
      sync_info: {
        last_sync_at: null,
        github_token_configured: true,
        single_user_mode: true,
        total_repos: 10,
        synced_repos: 10,
        embedded_repos: 9,
        summarized_repos: 8,
        schedule: null,
      },
      graph_defaults: {
        max_clusters: 8,
        min_clusters: 3,
        related_min_semantic: 0.65,
        hq_rendering: true,
        show_trajectories: true,
      },
    });
    triggerFullRefreshV2.mockResolvedValue({
      task: { task_id: 123, message: 'started', reset_count: 10 },
    });
    pollUntilComplete.mockImplementation(async ({ onProgress }) => {
      onProgress?.({
        task_id: 123,
        task_type: 'full_refresh',
        status: 'partial_failed',
        phase: 'complete',
        progress_percent: 100,
        eta_seconds: null,
        last_error: null,
        retryable: false,
        started_at: null,
        completed_at: null,
        error_details: {
          partial_failures: [{ phase: 'stars', task_id: 9, failed_items: 2 }],
        },
      });
      return { success: true, error: null, cancelled: false };
    });
  });

  it('shows a warning banner when full refresh completes with partial failures', async () => {
    render(<Settings />);

    await waitFor(() => expect(loadSettings).toHaveBeenCalled());

    fireEvent.click(screen.getByText('trigger refresh'));

    await waitFor(() => {
      expect(
        screen.getByText(/Full refresh completed with warnings/i)
      ).toBeInTheDocument();
    });
    expect(screen.getByTestId('step-stars')).toHaveTextContent('warning');
    expect(screen.getByTestId('step-embeddings')).toHaveTextContent('completed');
  });

  it('preserves local appearance settings when syncing backend graph defaults', async () => {
    render(<Settings />);

    await waitFor(() => expect(loadSettings).toHaveBeenCalled());

    expect(updateSettings).toHaveBeenCalledWith({
      maxClusters: 8,
      minClusters: 3,
    });
    expect(updateSettings).not.toHaveBeenCalledWith(
      expect.objectContaining({
        relatedMinSemantic: expect.any(Number),
      })
    );
    expect(updateSettings).not.toHaveBeenCalledWith(
      expect.objectContaining({
        hqRendering: expect.any(Boolean),
      })
    );
    expect(updateSettings).not.toHaveBeenCalledWith(
      expect.objectContaining({
        showTrajectories: expect.any(Boolean),
      })
    );
  });

  it('shows a warning banner when incremental sync completes with partial failures', async () => {
    startSyncPipelineV2.mockResolvedValue({ pipeline_run_id: 456 });
    pollUntilComplete.mockResolvedValueOnce({
      success: false,
      error: 'Pipeline phase partial failure phase=stars task_id=22 failed_items=2',
      cancelled: false,
    });

    render(<Settings />);

    await waitFor(() => expect(loadSettings).toHaveBeenCalled());

    fireEvent.click(screen.getByRole('button', { name: 'dashboard.sync_button' }));

    await waitFor(() => {
      expect(
        screen.getByText(/completed with warnings/i)
      ).toBeInTheDocument();
    });
  });

  it('shows a warning banner when recluster completes with partial failures', async () => {
    startReclusterV2.mockResolvedValue({ pipeline_run_id: 789 });
    pollUntilComplete.mockResolvedValueOnce({
      success: false,
      error: 'Pipeline phase partial failure phase=clustering task_id=33 failed_items=1',
      cancelled: false,
    });

    render(<Settings />);

    await waitFor(() => expect(loadSettings).toHaveBeenCalled());

    fireEvent.click(screen.getByRole('button', { name: 'graph.recluster' }));

    await waitFor(() => {
      expect(
        screen.getByText(/completed with warnings/i)
      ).toBeInTheDocument();
    });
  });
});
