export type ThemePreference = 'light' | 'dark' | 'system';

const STORAGE_KEY = 'nebula_theme';
const DURATION_MS = 220;
const media = window.matchMedia('(prefers-color-scheme: dark)');

const read = (): ThemePreference => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark' || stored === 'system') return stored;
  } catch {
    // Private mode: follow the system.
  }
  return 'system';
};

let preference = read();

const resolve = (pref: ThemePreference): 'light' | 'dark' =>
  pref === 'system' ? (media.matches ? 'dark' : 'light') : pref;

const apply = () => {
  const resolved = resolve(preference);
  document.documentElement.classList.toggle('dark', resolved === 'dark');
  document.documentElement.style.colorScheme = resolved;
};

export const initTheme = (): void => {
  apply();
  media.addEventListener('change', () => preference === 'system' && apply());
};

export const getTheme = (): ThemePreference => preference;

/** Switch theme with a circular wipe from the click point when supported. */
export const setTheme = (next: ThemePreference, origin?: { x: number; y: number }): void => {
  const changesLook = resolve(next) !== resolve(preference);
  const commit = () => {
    preference = next;
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Preference stays for this session.
    }
    apply();
  };
  const doc = document as Document & { startViewTransition?: (cb: () => void) => { finished: Promise<void> } };
  if (!changesLook || !origin || !doc.startViewTransition || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    commit();
    return;
  }
  const radius = Math.hypot(Math.max(origin.x, innerWidth - origin.x), Math.max(origin.y, innerHeight - origin.y));
  const style = document.createElement('style');
  style.textContent = `
    ::view-transition-old(root), ::view-transition-new(root) { animation: none; mix-blend-mode: normal; }
    ::view-transition-new(root) { animation: theme-wipe ${DURATION_MS}ms cubic-bezier(0.22, 1, 0.36, 1) both; }
    @keyframes theme-wipe {
      from { clip-path: circle(0px at ${origin.x}px ${origin.y}px); }
      to { clip-path: circle(${radius}px at ${origin.x}px ${origin.y}px); }
    }`;
  document.head.appendChild(style);
  doc.startViewTransition(commit).finished.finally(() => style.remove());
};
