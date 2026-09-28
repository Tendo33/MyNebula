import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import { Clock } from 'lucide-react';
import type { ScheduleResponse } from '../../api/v2/settings';
import { formatLastRunTime, formatNextRunTime, getStatusDisplay } from '../../utils/scheduleFormat';
import { Card } from '@/components/ui/card';
import { SelectField } from '@/components/ui/select';
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
        <Card variant="muted" className="group flex flex-row items-center justify-between gap-3 p-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="shrink-0 p-2 rounded-md bg-bg-sidebar group-hover:bg-bg-main transition-colors dark:group-hover:bg-dark-bg-main">
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
          <Card variant="muted" className="p-4">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex flex-wrap items-center gap-2">
                <label className="text-sm text-text-muted">{t('settings.execution_time')}:</label>
                <div className="flex items-center gap-2">
                  <SelectField
                    value={String(schedule.schedule_hour)}
                    onValueChange={(value) => onTimeChange(Number(value), schedule.schedule_minute)}
                    disabled={scheduleLoading}
                    aria-label={t('settings.execution_hour')}
                    options={Array.from({ length: 24 }, (_, hour) => ({
                      value: String(hour),
                      label: String(hour).padStart(2, '0'),
                    }))}
                  />
                  <span className="text-text-muted">:</span>
                  <SelectField
                    value={String(schedule.schedule_minute)}
                    onValueChange={(value) => onTimeChange(schedule.schedule_hour, Number(value))}
                    disabled={scheduleLoading}
                    aria-label={t('settings.execution_minute')}
                    options={[0, 15, 30, 45].map((minute) => ({
                      value: String(minute),
                      label: String(minute).padStart(2, '0'),
                    }))}
                  />
                </div>
              </div>
              <span className="text-xs text-text-muted">({schedule.timezone})</span>
            </div>
          </Card>
        )}

        <Card variant="muted" className="flex flex-col gap-1 p-4">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-text-muted">
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
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-text-muted">
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
