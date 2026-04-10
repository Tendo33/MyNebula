import { describe, expect, it } from 'vitest';

import zh from './translation.json';

describe('zh translation bundle', () => {
  it('keeps key interface labels readable', () => {
    expect(zh.sidebar.dashboard).toBe('仪表盘');
    expect(zh.settings.title).toBe('设置');
    expect(zh.app.login).toBe('登录');
  });
});
