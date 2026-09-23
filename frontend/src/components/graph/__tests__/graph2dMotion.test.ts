import { describe, expect, it } from 'vitest';

import { getGraphMotionProfile } from '../graph2dMotion';

describe('getGraphMotionProfile', () => {
  it('gently settles projected coordinates instead of freezing them', () => {
    const profile = getGraphMotionProfile(true, false);

    expect(profile.cooldownTicks).toBeGreaterThan(0);
    expect(profile.cooldownTicks).toBeLessThanOrEqual(80);
    expect(profile.cooldownTime).toBeLessThanOrEqual(1500);
    expect(profile.forceScale).toBeGreaterThan(0);
    expect(profile.forceScale).toBeLessThan(0.25);
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
