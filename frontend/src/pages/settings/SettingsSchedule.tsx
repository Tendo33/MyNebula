import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import { Clock } from 'lucide-react';
import type { ScheduleResponse } from '../../api/v2/settings';
import { formatLastRunTime, formatNextRunTime, getStatusDisplay } from '../../utils/scheduleFormat';
import { Card } from '@/components/ui/card';
import { NativeSelect } from '@/components/ui/native-select';
import { Spinner } from '@/components/ui/spinner';
import { Switch } from '@/components/ui/switch';

interface SettingsScheduleProps {
  schedule: ScheduleResponse | null;
  scheduleLoading: boolean;
  onToggle: () => void;
  onTimeChange: (hour: number, minute: number) => void;
}

export const SettingsSchedule = ({
  schedule,
  scheduleLoading,
  onToggle,
  onTimeChange,
}: SettingsScheduleProps) => {
  const { t } = useTranslation();

  return (
    <section>
      <h2 className="section-heading mb-4 select-none">
        {t('settings.scheduled_sync')}
      </h2>

      <div className="flex flex-col gap-2">
        <Card variant="muted" className="flex flex-row items-center justify-between p-4 transition-all group">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-bg-main transition-colors dark:group-hover:bg-dark-bg-main">
              <Clock className="w-5 h-5 text-text-muted group-hover:text-text-main" />
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-medium text-text-main">{t('settings.enable_scheduled_sync')}</span>
              <span className="text-xs text-text-muted">{t('settings.enable_scheduled_sync_desc')}</span>
            </div>
          </div>
          {scheduleLoading ? (
            <Spinner className="text-muted-foreground" />
          ) : (
            <Switch
              checked={Boolean(schedule?.is_enabled)}
              onCheckedChange={() => onToggle()}
              aria-label={t('settings.enable_scheduled_sync')}
            />
          )}
        </Card>

        {schedule?.is_enabled && (
          <Card variant="muted" className="ml-0 p-4 sm:ml-14">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2">
                <label className="text-sm text-text-muted">{t('settings.execution_time')}:</label>
                <NativeSelect
                  value={schedule.schedule_hour}
                  onChange={(e) => onTimeChange(Number(e.target.value), schedule.schedule_minute)}
                  disabled={scheduleLoading}
                  aria-label={t('settings.execution_time')}
                >
                  {Array.from({ length: 24 }, (_, i) => (
                    <option key={i} value={i}>
                      {i.toString().padStart(2, '0')}
                    </option>
                  ))}
                </NativeSelect>
                <span className="text-text-muted">:</span>
                <NativeSelect
                  value={schedule.schedule_minute}
                  onChange={(e) => onTimeChange(schedule.schedule_hour, Number(e.target.value))}
                  disabled={scheduleLoading}
                  aria-label={t('settings.execution_time')}
                >
                  {[0, 15, 30, 45].map((m) => (
                    <option key={m} value={m}>
                      {m.toString().padStart(2, '0')}
                    </option>
                  ))}
                </NativeSelect>
              </div>
              <span className="text-xs text-text-muted">({schedule.timezone})</span>
            </div>
          </Card>
        )}

        <Card variant="muted" className="ml-0 flex flex-col gap-1 p-4 sm:ml-14">
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
        </Card>
      </div>
    </section>
  );
};
