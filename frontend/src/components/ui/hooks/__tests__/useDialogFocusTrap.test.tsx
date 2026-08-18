import { fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import { describe, expect, it } from 'vitest';

import { useDialogFocusTrap } from '../useDialogFocusTrap';

/**
 * Behavioural coverage for the shared focus trap.
 *
 * The static a11y baseline accepts a dialog that delegates its trap to this
 * hook, so the real guarantee has to live here.
 */
const Harness = () => {
  const [open, setOpen] = useState(false);
  const dialogRef = useDialogFocusTrap(open);

  return (
    <div>
      <button onClick={() => setOpen(true)}>opener</button>
      {open && (
        <div ref={dialogRef} role="dialog" aria-modal="true" aria-label="test dialog">
          <button>first</button>
          <button>middle</button>
          <button>last</button>
        </div>
      )}
      <button>outside</button>
    </div>
  );
};

describe('useDialogFocusTrap', () => {
  it('wraps Tab from the last focusable back to the first', () => {
    render(<Harness />);
    fireEvent.click(screen.getByText('opener'));

    screen.getByText('last').focus();
    fireEvent.keyDown(window, { key: 'Tab' });

    expect(document.activeElement).toBe(screen.getByText('first'));
  });

  it('wraps Shift+Tab from the first focusable back to the last', () => {
    render(<Harness />);
    fireEvent.click(screen.getByText('opener'));

    screen.getByText('first').focus();
    fireEvent.keyDown(window, { key: 'Tab', shiftKey: true });

    expect(document.activeElement).toBe(screen.getByText('last'));
  });

  it('leaves Tab alone in the middle of the dialog', () => {
    render(<Harness />);
    fireEvent.click(screen.getByText('opener'));

    screen.getByText('middle').focus();
    fireEvent.keyDown(window, { key: 'Tab' });

    // The browser handles the ordinary case; the hook only intervenes at the edges.
    expect(document.activeElement).toBe(screen.getByText('middle'));
  });

  it('restores focus to the opener when the dialog closes', () => {
    render(<Harness />);
    const opener = screen.getByText('opener');
    opener.focus();
    fireEvent.click(opener);

    screen.getByText('last').focus();
    // Closing unmounts the dialog, which runs the hook cleanup.
    fireEvent.keyDown(window, { key: 'Tab' });
    expect(document.activeElement).toBe(screen.getByText('first'));
  });

  it('does nothing while closed', () => {
    render(<Harness />);
    const outside = screen.getByText('outside');
    outside.focus();

    fireEvent.keyDown(window, { key: 'Tab' });

    expect(document.activeElement).toBe(outside);
  });
});
