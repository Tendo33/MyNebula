import { fireEvent, render, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { SyncProgress } from '../SyncProgress';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

describe('SyncProgress accessibility', () => {
  it('moves focus into the modal, supports Escape, and restores focus', async () => {
    const trigger = document.createElement('button');
    document.body.appendChild(trigger);
    trigger.focus();
    const onClose = vi.fn();

    const view = render(
      <SyncProgress
        isOpen
        canClose
        onClose={onClose}
        steps={[{ id: 'stars', label: 'Stars', status: 'running', progress: 40 }]}
      />
    );

    const dialog = await waitFor(() => view.getByRole('dialog'));
    expect(dialog).toHaveFocus();

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledOnce();

    view.unmount();
    expect(trigger).toHaveFocus();
    trigger.remove();
  });

  it('keeps keyboard focus inside a non-closable modal', async () => {
    const view = render(
      <SyncProgress
        isOpen
        steps={[{ id: 'stars', label: 'Stars', status: 'running', progress: 40 }]}
      />
    );

    const dialog = await waitFor(() => view.getByRole('dialog'));
    fireEvent.keyDown(document, { key: 'Tab' });

    expect(dialog).toHaveFocus();
  });

  it('does not reset focus when callback identity or closeability changes', async () => {
    const firstClose = vi.fn();
    const nextClose = vi.fn();
    const view = render(
      <SyncProgress
        isOpen
        canClose
        onClose={firstClose}
        steps={[{ id: 'stars', label: 'Stars', status: 'running' }]}
      />
    );

    const closeButton = await waitFor(() => view.getByRole('button', { name: 'Close' }));
    closeButton.focus();
    view.rerender(
      <SyncProgress
        isOpen
        canClose
        onClose={nextClose}
        steps={[{ id: 'stars', label: 'Stars', status: 'running' }]}
      />
    );

    expect(closeButton).toHaveFocus();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(firstClose).not.toHaveBeenCalled();
    expect(nextClose).toHaveBeenCalledOnce();
  });
});
