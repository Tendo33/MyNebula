import { useEffect, useRef } from 'react';

/**
 * Trap Tab inside an open dialog and restore focus to the previously focused
 * element on close.
 *
 * Without the trap, Tab walks into the inert page behind the modal. Shared
 * between the command palette and any future dialog, so the behaviour does not
 * get reimplemented per surface.
 */
export const useDialogFocusTrap = (isOpen: boolean) => {
  const dialogRef = useRef<HTMLDivElement>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!isOpen) return;

    previouslyFocusedRef.current = document.activeElement as HTMLElement | null;

    const handleTab = (event: KeyboardEvent) => {
      if (event.key !== 'Tab' || !dialogRef.current) return;
      const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    window.addEventListener('keydown', handleTab);
    return () => {
      window.removeEventListener('keydown', handleTab);
      previouslyFocusedRef.current?.focus();
    };
  }, [isOpen]);

  return dialogRef;
};
