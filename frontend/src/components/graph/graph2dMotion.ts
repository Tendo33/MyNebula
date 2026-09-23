export interface GraphMotionProfile {
  alphaDecay: number;
  velocityDecay: number;
  cooldownTicks: number;
  cooldownTime: number;
  warmupTicks: number;
  forceScale: number;
}

export const getGraphMotionProfile = (
  hasProjectedPositions: boolean,
  reduceMotion: boolean
): GraphMotionProfile => {
  if (reduceMotion) {
    return {
      alphaDecay: 1,
      velocityDecay: 0.9,
      cooldownTicks: 0,
      cooldownTime: 0,
      warmupTicks: 0,
      forceScale: 0,
    };
  }

  if (hasProjectedPositions) {
    return {
      alphaDecay: 0.09,
      velocityDecay: 0.55,
      cooldownTicks: 70,
      cooldownTime: 1400,
      warmupTicks: 0,
      forceScale: 0.15,
    };
  }

  return {
    alphaDecay: 0.05,
    velocityDecay: 0.28,
    cooldownTicks: 160,
    cooldownTime: 8000,
    warmupTicks: 0,
    forceScale: 1,
  };
};
