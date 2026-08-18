import { useCallback, useEffect, useRef, useState } from 'react';

import type { ImageCache } from '../graph2dTypes';

/**
 * Owner-avatar image cache with a debounced redraw signal.
 *
 * Avatars load asynchronously while the canvas paints. Re-rendering per loaded
 * image causes a render storm on a large graph, so loads are batched and
 * flushed once after a short idle period. The flush timer is cleared on unmount
 * — a fix from the 2026-05-09 health check that must survive any refactor.
 */
export const AVATAR_REDRAW_DEBOUNCE_MS = 200;

export const useAvatarImageCache = () => {
  const imageCacheRef = useRef<ImageCache>(new Map());
  const [, forceUpdate] = useState(0);
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const triggerAvatarRedraw = useCallback(() => {
    if (flushTimerRef.current) clearTimeout(flushTimerRef.current);
    flushTimerRef.current = setTimeout(() => {
      forceUpdate((n) => n + 1);
      flushTimerRef.current = null;
    }, AVATAR_REDRAW_DEBOUNCE_MS);
  }, []);

  useEffect(
    () => () => {
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current);
      }
    },
    []
  );

  return { imageCacheRef, triggerAvatarRedraw };
};
