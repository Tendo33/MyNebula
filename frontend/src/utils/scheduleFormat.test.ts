import { describe, expect, it } from 'vitest';

import { formatNextRunTime } from './scheduleFormat';

const t = (key: string) => key;

describe('scheduleFormat', () => {
  it('formats timezone-aware UTC timestamps in the requested timezone', () => {
    const timestamp = '2026-04-08T01:00:00Z';
    const expected = new Date(timestamp).toLocaleString(undefined, {
      timeZone: 'Asia/Shanghai',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });

    expect(formatNextRunTime(timestamp, 'Asia/Shanghai', t)).toBe(expected);
  });

  it('treats legacy naive timestamps as UTC instead of local browser time', () => {
    const timestamp = '2026-04-08T01:00:00';
    const expected = new Date('2026-04-08T01:00:00Z').toLocaleString(undefined, {
      timeZone: 'Asia/Shanghai',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });

    expect(formatNextRunTime(timestamp, 'Asia/Shanghai', t)).toBe(expected);
  });
});
