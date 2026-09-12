import { useCallback, useEffect, useState } from 'react';

import { runThemeViewTransition } from '../lib/themeTransition';

export type ThemePreference = 'light' | 'dark' | 'system';

const STORAGE_KEY = 'nebula_theme';
const PREFERENCES: ThemePreference[] = ['light', 'dark', 'system'];

const readStoredPreference = (): ThemePreference => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && (PREFERENCES as string[]).includes(stored)) {
      return stored as ThemePreference;
    }
  } catch {
    // Private mode or a blocked store: fall through to the system default.
  }
  return 'system';
};

const prefersDark = () =>
  typeof window !== 'undefined' &&
  typeof window.matchMedia === 'function' &&
  window.matchMedia('(prefers-color-scheme: dark)').matches;

export const resolveTheme = (preference: ThemePreference): 'light' | 'dark' =>
  preference === 'system' ? (prefersDark() ? 'dark' : 'light') : preference;

/**
 * Theme preference, applied by toggling `html.dark`.
 *
 * Tailwind is configured with `darkMode: 'class'`, so the ~275 `dark:` variants
 * across the codebase only take effect once that class exists. Nothing set it
 * before this hook, which left the entire dark palette unreachable and
 * unverified.
 */
export const useTheme = () => {
  const [preference, setPreference] = useState<ThemePreference>(readStoredPreference);

  useEffect(() => {
    const apply = () => {
      const resolved = resolveTheme(preference);
      document.documentElement.classList.toggle('dark', resolved === 'dark');
      document.documentElement.style.colorScheme = resolved;
    };
    apply();

    if (preference !== 'system' || typeof window.matchMedia !== 'function') {
      return;
    }
    // Follow the OS while the preference is "system".
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    media.addEventListener('change', apply);
    return () => media.removeEventListener('change', apply);
  }, [preference]);

  const updatePreference = useCallback((
    next: ThemePreference,
    origin?: { clientX: number; clientY: number }
  ) => {
    const apply = () => {
      setPreference(next);
      try {
        localStorage.setItem(STORAGE_KEY, next);
      } catch {
        // Preference stays in memory for this session.
      }
    };

    const currentResolved = resolveTheme(preference);
    const nextResolved = resolveTheme(next);
    if (currentResolved === nextResolved) {
      apply();
      return;
    }

    runThemeViewTransition(origin, apply);
  }, [preference]);

  return { preference, resolved: resolveTheme(preference), setPreference: updatePreference };
};
