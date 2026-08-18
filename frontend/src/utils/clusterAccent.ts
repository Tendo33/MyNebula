const FALLBACK_CLUSTER_COLORS = [
  '#4F46E5',
  '#0F766E',
  '#B45309',
  '#BE185D',
  '#2563EB',
  '#7C3AED',
  '#047857',
  '#C2410C',
];

const addAlpha = (hex: string, alpha: string) => {
  if (!hex.startsWith('#')) return hex;
  const normalized = hex.length === 4
    ? `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}`
    : hex;
  return `${normalized}${alpha}`;
};

const hexToRgb = (hex: string): { r: number; g: number; b: number } | null => {
  if (!hex.startsWith('#')) return null;
  const normalized =
    hex.length === 4 ? `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}` : hex;
  if (normalized.length < 7) return null;
  return {
    r: Number.parseInt(normalized.slice(1, 3), 16),
    g: Number.parseInt(normalized.slice(3, 5), 16),
    b: Number.parseInt(normalized.slice(5, 7), 16),
  };
};

const relativeLuminance = ({ r, g, b }: { r: number; g: number; b: number }) => {
  const channel = (value: number) => {
    const v = value / 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
};

const LIGHT_SURFACE = { r: 251, g: 250, b: 246 }; // --color-bg-main, light
const DARK_SURFACE = { r: 29, g: 33, b: 44 }; // --color-bg-sidebar, dark

const contrastAgainst = (hex: string, surface: { r: number; g: number; b: number }) => {
  const rgb = hexToRgb(hex);
  if (!rgb) return 21;
  const l1 = relativeLuminance(rgb);
  const l2 = relativeLuminance(surface);
  return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
};

/** Contrast ratio of a colour against the light surface chips sit on. */
export const contrastOnLightSurface = (hex: string): number =>
  contrastAgainst(hex, LIGHT_SURFACE);

/** Contrast ratio of a colour against the dark surface chips sit on. */
export const contrastOnDarkSurface = (hex: string): number =>
  contrastAgainst(hex, DARK_SURFACE);

/**
 * Nudge a cluster hue until its text meets the WCAG AA 4.5:1 floor.
 *
 * Direction depends on the surface: darken against the cream light surface,
 * lighten against the near-black dark one. A single darkening pass — the first
 * version of this — made chips *worse* in dark mode, dropping #ef4444 to
 * 3.23:1. Hue identity is preserved either way; `base` and `dot` keep the raw
 * colour.
 */
const readableTextColor = (hex: string, mode: 'light' | 'dark'): string => {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;
  const measure = mode === 'light' ? contrastOnLightSurface : contrastOnDarkSurface;
  const factor = mode === 'light' ? 0.9 : 1.12;
  let { r, g, b } = rgb;
  for (let step = 0; step < 24 && measure(toHex({ r, g, b })) < 4.5; step += 1) {
    r = Math.min(255, Math.round(r * factor));
    g = Math.min(255, Math.round(g * factor));
    b = Math.min(255, Math.round(b * factor));
  }
  return toHex({ r, g, b });
};

const toHex = ({ r, g, b }: { r: number; g: number; b: number }) =>
  `#${[r, g, b].map((v) => Math.max(0, Math.min(255, v)).toString(16).padStart(2, '0')).join('')}`;

export const getClusterAccent = ({
  id,
  color,
}: {
  id: number;
  color?: string | null;
}) => {
  const base =
    color && color.trim().length > 0
      ? color
      : FALLBACK_CLUSTER_COLORS[Math.abs(id) % FALLBACK_CLUSTER_COLORS.length];

  return {
    base,
    dot: base,
    softBackground: addAlpha(base, '20'),
    strongBackground: addAlpha(base, '2E'),
    softBorder: addAlpha(base, '44'),
    strongBorder: addAlpha(base, '6E'),
    text: readableTextColor(base, 'light'),
    textOnDark: readableTextColor(base, 'dark'),
  };
};
