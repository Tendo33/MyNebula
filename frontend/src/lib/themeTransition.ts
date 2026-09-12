import { flushSync } from 'react-dom';

type Point = { clientX: number; clientY: number };

const DURATION_MS = 220;
const EASING = 'cubic-bezier(0.22, 1, 0.36, 1)';
const STYLE_ID = 'mynebula-theme-view-transition';

const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  typeof window.matchMedia === 'function' &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * Circular clip-path theme wipe, adapted from Great UI's circular-theme-provider.
 * Duration stays in the product 150–220ms window; reduced motion skips the wipe.
 */
export const runThemeViewTransition = (origin: Point | undefined, update: () => void) => {
  const doc = document as Document & {
    startViewTransition?: (callback: () => void) => { finished?: Promise<void> };
  };

  if (!origin || !doc.startViewTransition || prefersReducedMotion()) {
    update();
    return;
  }

  const x = origin.clientX;
  const y = origin.clientY;
  const endRadius = Math.hypot(
    Math.max(x, window.innerWidth - x),
    Math.max(y, window.innerHeight - y)
  );

  let styleEl = document.getElementById(STYLE_ID) as HTMLStyleElement | null;
  if (!styleEl) {
    styleEl = document.createElement('style');
    styleEl.id = STYLE_ID;
    document.head.appendChild(styleEl);
  }
  styleEl.textContent = `
    ::view-transition-old(root),
    ::view-transition-new(root) {
      animation: none !important;
      mix-blend-mode: normal;
    }
    ::view-transition-new(root) {
      animation: mynebula-theme-wipe ${DURATION_MS}ms ${EASING} both !important;
    }
    @keyframes mynebula-theme-wipe {
      from { clip-path: circle(0px at ${x}px ${y}px); }
      to { clip-path: circle(${endRadius}px at ${x}px ${y}px); }
    }
  `;

  const cleanup = () => {
    document.getElementById(STYLE_ID)?.remove();
  };

  try {
    const transition = doc.startViewTransition(() => {
      flushSync(update);
    });
    if (transition.finished) {
      void transition.finished.then(cleanup).catch(cleanup);
    } else {
      window.setTimeout(cleanup, DURATION_MS);
    }
  } catch {
    cleanup();
    update();
  }
};
