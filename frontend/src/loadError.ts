/**
 * Keep only the part of a transport error a visitor can act on.
 * Parse and schema failures are developer problems; they stay in the console.
 */
export const loadErrorDetail = (message: string | null | undefined): string => {
  const text = (message ?? '').replace(/\s+/g, ' ').trim();
  const status = text.match(/^HTTP\s+(\d{3})$/i);
  return status ? `HTTP ${status[1]}` : '';
};
