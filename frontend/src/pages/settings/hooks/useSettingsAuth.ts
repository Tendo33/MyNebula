import { useEffect, useState, type FormEvent } from 'react';
import { isAxiosError } from 'axios';
import type { TFunction } from 'i18next';

import { getAdminAuthConfig } from '../../../api/auth';
import { logClientError } from '../../../utils/debug';

/**
 * Admin login form state and the admin-auth availability probe.
 *
 * `adminAuthConfigured` is tri-state: `true` when the server has admin auth
 * configured, `false` when it explicitly does not, and `null` when the probe
 * itself failed — the login form renders differently for each.
 */
export const useSettingsAuth = ({
  isAuthenticated,
  login,
  logout,
  onLoggedIn,
  onLoggedOut,
  t,
}: {
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  onLoggedIn: () => Promise<void>;
  onLoggedOut: () => void;
  t: TFunction;
}) => {
  const [loginUsername, setLoginUsername] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);
  const [adminAuthConfigured, setAdminAuthConfigured] = useState<boolean | null>(null);

  useEffect(() => {
    if (isAuthenticated) {
      setAdminAuthConfigured(true);
      return;
    }

    getAdminAuthConfig()
      .then((config) => setAdminAuthConfigured(config.enabled))
      .catch(() => setAdminAuthConfigured(null));
  }, [isAuthenticated]);

  const handleAdminLogin = async (e: FormEvent) => {
    e.preventDefault();
    if (adminAuthConfigured === false) {
      setLoginError(t('settings.admin_not_configured'));
      return;
    }
    setLoginLoading(true);
    setLoginError(null);

    try {
      await login(loginUsername.trim(), loginPassword);
      setLoginPassword('');
      await onLoggedIn();
    } catch (err) {
      if (isAxiosError(err) && err.response?.status === 503) {
        setLoginError(t('settings.admin_not_configured'));
      } else {
        setLoginError(t('settings.login_failed'));
      }
      logClientError('Admin login failed:', err);
    } finally {
      setLoginLoading(false);
    }
  };

  const handleAdminLogout = async () => {
    onLoggedOut();
    await logout();
  };

  return {
    loginUsername,
    loginPassword,
    loginLoading,
    loginError,
    adminAuthConfigured,
    setLoginUsername,
    setLoginPassword,
    handleAdminLogin,
    handleAdminLogout,
  };
};
