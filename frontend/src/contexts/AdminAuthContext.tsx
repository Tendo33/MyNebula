import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { getAdminSession, loginAdmin, logoutAdmin } from '../api/auth';
import { setUnauthorizedHandler } from '../api/client';

interface AdminAuthContextValue {
  isChecking: boolean;
  isAuthenticated: boolean;
  username: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshSession: () => Promise<void>;
}

const AdminAuthContext = createContext<AdminAuthContextValue | null>(null);

export const AdminAuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isChecking, setIsChecking] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [username, setUsername] = useState<string | null>(null);

  const clearSession = useCallback(() => {
    setIsAuthenticated(false);
    setUsername(null);
    setIsChecking(false);
  }, []);

  const refreshSession = useCallback(async () => {
    try {
      const session = await getAdminSession();
      setIsAuthenticated(session.authenticated);
      setUsername(session.authenticated ? session.username : null);
    } catch {
      clearSession();
    } finally {
      setIsChecking(false);
    }
  }, [clearSession]);

  useEffect(() => {
    refreshSession();
  }, [refreshSession]);

  useEffect(() => {
    setUnauthorizedHandler((error) => {
      const requestUrl = error.config?.url ?? '';
      if (requestUrl.includes('/auth/login')) {
        return;
      }
      clearSession();
    });

    return () => {
      setUnauthorizedHandler(null);
    };
  }, [clearSession]);

  useEffect(() => {
    if (!isAuthenticated || typeof window === 'undefined') {
      return;
    }

    const revalidateSession = () => {
      void refreshSession();
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        revalidateSession();
      }
    };

    window.addEventListener('focus', revalidateSession);
    document.addEventListener('visibilitychange', handleVisibilityChange);
    const timer = window.setInterval(revalidateSession, 5 * 60 * 1000);

    return () => {
      window.removeEventListener('focus', revalidateSession);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.clearInterval(timer);
    };
  }, [isAuthenticated, refreshSession]);

  const login = useCallback(async (inputUsername: string, password: string) => {
    const session = await loginAdmin({ username: inputUsername, password });
    setIsAuthenticated(session.authenticated);
    setUsername(session.authenticated ? session.username : null);
  }, []);

  const logout = useCallback(async () => {
    try {
      await logoutAdmin();
    } finally {
      clearSession();
    }
  }, [clearSession]);

  const value = useMemo(
    () => ({
      isChecking,
      isAuthenticated,
      username,
      login,
      logout,
      refreshSession,
    }),
    [isAuthenticated, isChecking, login, logout, refreshSession, username]
  );

  return <AdminAuthContext.Provider value={value}>{children}</AdminAuthContext.Provider>;
};

export const useAdminAuth = (): AdminAuthContextValue => {
  const context = useContext(AdminAuthContext);
  if (!context) {
    throw new Error('useAdminAuth must be used within an AdminAuthProvider');
  }
  return context;
};
