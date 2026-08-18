import { useCallback, useEffect, useState } from 'react';

import { logClientWarn } from '../../../utils/debug';
import { MAX_RECENT_SEARCHES, RECENT_SEARCHES_KEY } from '../commandPaletteTypes';

/**
 * Recent-search history persisted to localStorage.
 *
 * Every storage access is guarded: a private-mode or quota failure must degrade
 * to an in-memory list rather than break the palette.
 */
export const useRecentSearches = () => {
  const [recentSearches, setRecentSearches] = useState<string[]>([]);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(RECENT_SEARCHES_KEY);
      if (stored) {
        setRecentSearches(JSON.parse(stored));
      }
    } catch {
      logClientWarn('Failed to load recent searches');
    }
  }, []);

  const addRecentSearch = useCallback((query: string) => {
    if (!query.trim()) return;

    setRecentSearches((prev) => {
      const filtered = prev.filter((s) => s !== query);
      const updated = [query, ...filtered].slice(0, MAX_RECENT_SEARCHES);
      try {
        localStorage.setItem(RECENT_SEARCHES_KEY, JSON.stringify(updated));
      } catch {
        logClientWarn('Failed to save recent searches');
      }
      return updated;
    });
  }, []);

  const clearRecentSearches = useCallback(() => {
    setRecentSearches([]);
    try {
      localStorage.removeItem(RECENT_SEARCHES_KEY);
    } catch {
      logClientWarn('Failed to clear recent searches');
    }
  }, []);

  return { recentSearches, addRecentSearch, clearRecentSearches };
};
