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
    text: addAlpha(base, 'E0'),
  };
};
