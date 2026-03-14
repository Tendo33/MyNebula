type TranslationFn = (key: string, options?: Record<string, unknown>) => string;

export const formatNextRunTime = (
  nextRunAt: string | null,
  timezone: string,
  t: TranslationFn
): string => {
  if (!nextRunAt) return t('time.not_set');

  try {
    const date = new Date(nextRunAt);
    return date.toLocaleString(undefined, {
      timeZone: timezone,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return nextRunAt;
  }
};

export const formatLastRunTime = (lastRunAt: string | null, t: TranslationFn): string => {
  if (!lastRunAt) return t('time.never_run');

  try {
    const date = new Date(lastRunAt);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) {
      return t('time.days_ago', { count: diffDays });
    }
    if (diffHours > 0) {
      return t('time.hours_ago', { count: diffHours });
    }
    const diffMinutes = Math.floor(diffMs / (1000 * 60));
    return diffMinutes > 0 ? t('time.minutes_ago', { count: diffMinutes }) : t('time.just_now');
  } catch {
    return lastRunAt;
  }
};

export const getStatusDisplay = (
  status: string | null,
  t: TranslationFn
): { text: string; color: string } => {
  switch (status) {
    case 'success':
      return { text: t('time.status.success'), color: 'text-green-600' };
    case 'failed':
      return { text: t('time.status.failed'), color: 'text-red-600' };
    case 'running':
      return { text: t('time.status.running'), color: 'text-action-primary' };
    default:
      return { text: t('time.status.unknown'), color: 'text-text-dim' };
  }
};
