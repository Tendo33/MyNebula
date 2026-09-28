import { describe, expect, it } from 'vitest';

import { getGraphMotionProfile } from '../graph2dMotion';

describe('getGraphMotionProfile', () => {
  it('keeps snapshot-projected coordinates static', () => {
    const profile = getGraphMotionProfile(true, false);

    expect(profile.cooldownTicks).toBe(0);
    expect(profile.cooldownTime).toBe(0);
    expect(profile.forceScale).toBe(0);
  });

  it('keeps a longer layout pass when coordinates are missing', () => {
    const projected = getGraphMotionProfile(true, false);
    const unpositioned = getGraphMotionProfile(false, false);

    expect(unpositioned.cooldownTicks).toBeGreaterThan(projected.cooldownTicks);
    expect(unpositioned.forceScale).toBe(1);
  });

  it('disables spatial motion for reduced-motion users', () => {
    const profile = getGraphMotionProfile(true, true);

    expect(profile.cooldownTicks).toBe(0);
    expect(profile.cooldownTime).toBe(0);
    expect(profile.forceScale).toBe(0);
  });
});
