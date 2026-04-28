import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const {
  getAdminSession,
  loginAdmin,
  logoutAdmin,
  setUnauthorizedHandler,
} = vi.hoisted(() => ({
  getAdminSession: vi.fn(),
  loginAdmin: vi.fn(),
  logoutAdmin: vi.fn(),
  setUnauthorizedHandler: vi.fn(),
}));

vi.mock('../api/auth', () => ({
  getAdminSession,
  loginAdmin,
  logoutAdmin,
}));

vi.mock('../api/client', () => ({
  setUnauthorizedHandler,
}));

import { AdminAuthProvider, useAdminAuth } from './AdminAuthContext';

const Probe = () => {
  const { isAuthenticated, username, isChecking, refreshSession } = useAdminAuth();

  return (
    <div>
      <span>{isChecking ? 'checking' : isAuthenticated ? 'auth' : 'guest'}</span>
      <span>{username ?? 'none'}</span>
      <button type="button" onClick={() => void refreshSession()}>
        refresh
      </button>
    </div>
  );
};

describe('AdminAuthContext', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('preserves authenticated session when refresh fails transiently', async () => {
    getAdminSession
      .mockResolvedValueOnce({ authenticated: true, username: 'owner' })
      .mockRejectedValueOnce(new Error('network down'));

    render(
      <AdminAuthProvider>
        <Probe />
      </AdminAuthProvider>
    );

    await screen.findByText('auth');
    expect(screen.getByText('owner')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'refresh' }));

    await waitFor(() => {
      expect(getAdminSession).toHaveBeenCalledTimes(2);
    });

    await waitFor(() => {
      expect(screen.getByText('auth')).toBeInTheDocument();
      expect(screen.getByText('owner')).toBeInTheDocument();
    });
  });
});
