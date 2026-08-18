import { useCallback, useEffect, useRef, useState } from 'react';
import type { TFunction } from 'i18next';

import {
  getSettingsV2,
  updateGraphDefaultsV2,
  updateScheduleV2,
  type ScheduleConfig,
  type ScheduleResponse,
  type SyncInfoResponse,
} from '../../../api/v2/settings';
import { logClientError } from '../../../utils/debug';
import type { GraphSettings } from '../../../contexts/GraphContext';

/**
 * Settings payload loading, schedule mutation, and debounced graph-defaults
 * persistence.
 *
 * `graphDefaultsRef` holds the last values known to the server so a re-render
 * that does not change them never issues a write.
 */
export const useSettingsSchedule = ({
  isAuthenticated,
  settings,
  updateSettings,
  setError,
  setWarning,
  t,
}: {
  isAuthenticated: boolean;
  settings: GraphSettings;
  updateSettings: (next: Partial<GraphSettings>) => void;
  setError: (message: string | null) => void;
  setWarning: (message: string | null) => void;
  t: TFunction;
}) => {
  const [schedule, setSchedule] = useState<ScheduleResponse | null>(null);
  const [syncInfo, setSyncInfo] = useState<SyncInfoResponse | null>(null);
  const [scheduleLoading, setScheduleLoading] = useState(false);
  const graphDefaultsRef = useRef<{ max: number; min: number } | null>(null);

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
  }, [setError, setWarning, t, updateSettings]);

  const clearScheduleState = useCallback(() => {
    setSchedule(null);
    setSyncInfo(null);
  }, []);

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
  }, [isAuthenticated, settings.maxClusters, settings.minClusters, setError, t]);

  const handleScheduleToggle = useCallback(async () => {
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
  }, [schedule, setError, t]);

  const handleTimeChange = useCallback(
    async (hour: number, minute: number) => {
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
    },
    [schedule, setError, t]
  );

  return {
    schedule,
    syncInfo,
    scheduleLoading,
    loadScheduleData,
    clearScheduleState,
    handleScheduleToggle,
    handleTimeChange,
  };
};
