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

/** Contrast ratio of a colour against the light surface chips sit on. */
export const contrastOnLightSurface = (hex: string): number => {
  const rgb = hexToRgb(hex);
  if (!rgb) return 21;
  const l1 = relativeLuminance(rgb);
  const l2 = relativeLuminance({ r: 251, g: 250, b: 246 }); // --bg-main
  return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
};

/**
 * Darken a cluster hue until its text meets the WCAG AA 4.5:1 floor.
 *
 * The previous implementation used the raw hue at 88% alpha, which put bright
 * accents such as `#10b981` at 2.43:1 on the cream surface — every chip on the
 * Data page failed AA. Darkening preserves the hue identity while making the
 * label legible.
 */
const readableTextColor = (hex: string): string => {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;
  let { r, g, b } = rgb;
  for (let step = 0; step < 24 && contrastOnLightSurface(toHex({ r, g, b })) < 4.5; step += 1) {
    r = Math.round(r * 0.9);
    g = Math.round(g * 0.9);
    b = Math.round(b * 0.9);
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
    text: readableTextColor(base),
  };
};
