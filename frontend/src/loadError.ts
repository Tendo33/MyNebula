/** Collapse a transport error into one line that fits the existing center column. */
export const loadErrorDetail = (message: string | null | undefined): string => {
  const text = (message ?? '').replace(/\s+/g, ' ').trim();
  if (!text) return '';
  return text.length > 180 ? `${text.slice(0, 179)}…` : text;
};
