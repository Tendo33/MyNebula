import { FormEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import { Loader2, Shield, User } from 'lucide-react';

interface SettingsLoginFormProps {
  loginUsername: string;
  loginPassword: string;
  loginLoading: boolean;
  loginError: string | null;
  adminAuthConfigured: boolean | null;
  onUsernameChange: (v: string) => void;
  onPasswordChange: (v: string) => void;
  onSubmit: (e: FormEvent) => void;
}

export const SettingsLoginForm = ({
  loginUsername,
  loginPassword,
  loginLoading,
  loginError,
  adminAuthConfigured,
  onUsernameChange,
  onPasswordChange,
  onSubmit,
}: SettingsLoginFormProps) => {
  const { t } = useTranslation();

  return (
    <section className="flex-1 flex items-center justify-center px-8">
      <div className="w-full max-w-md bg-bg-main border border-border-light rounded-xl shadow-sm p-6 dark:bg-dark-bg-main dark:border-dark-border">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-10 h-10 rounded-full bg-bg-sidebar flex items-center justify-center">
            <Shield className="w-5 h-5 text-text-main" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-text-main">{t('settings.admin_access')}</h2>
            <p className="text-xs text-text-muted">{t('settings.login_required_desc')}</p>
          </div>
        </div>

        <form className="space-y-4" onSubmit={onSubmit}>
          {adminAuthConfigured === false && (
            <div className="text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded-md px-3 py-2">
              {t('settings.admin_not_configured')}
            </div>
          )}

          <div>
            <label htmlFor="admin-username" className="block text-xs text-text-muted mb-1">
              {t('settings.username')}
            </label>
            <div className="relative">
              <User className="w-4 h-4 text-text-dim absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                id="admin-username"
                type="text"
                value={loginUsername}
                onChange={(e) => onUsernameChange(e.target.value)}
                className="w-full h-10 pl-9 pr-3 rounded-md border border-border-light text-sm bg-bg-main focus:outline-none focus:ring-1 focus:ring-text-main/30 dark:bg-dark-bg-main dark:border-dark-border dark:text-dark-text-main"
                autoComplete="username"
                required
              />
            </div>
          </div>

          <div>
            <label htmlFor="admin-password" className="block text-xs text-text-muted mb-1">
              {t('settings.password')}
            </label>
            <input
              id="admin-password"
              type="password"
              value={loginPassword}
              onChange={(e) => onPasswordChange(e.target.value)}
              className="w-full h-10 px-3 rounded-md border border-border-light text-sm bg-bg-main focus:outline-none focus:ring-1 focus:ring-text-main/30 dark:bg-dark-bg-main dark:border-dark-border dark:text-dark-text-main"
              autoComplete="current-password"
              required
            />
          </div>

          {loginError && (
            <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-md px-3 py-2">
              {loginError}
            </div>
          )}

          <button
            type="submit"
            disabled={loginLoading || adminAuthConfigured === false}
            className={clsx(
              'w-full h-10 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2',
              loginLoading || adminAuthConfigured === false
                ? 'bg-bg-hover text-text-dim border border-border-light cursor-not-allowed dark:bg-dark-bg-sidebar/70 dark:text-dark-text-main/60 dark:border-dark-border'
                : 'bg-text-main text-bg-main hover:bg-text-main/90'
            )}
          >
            {loginLoading && <Loader2 className="w-4 h-4 animate-spin" />}
            {t('app.login')}
          </button>
        </form>
      </div>
    </section>
  );
};
