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
    <section className="flex flex-1 items-center justify-center px-6 py-10">
      <div className="panel-surface-strong w-full max-w-lg p-7 sm:p-8">
        <div className="mb-6 flex items-start gap-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-md border border-border-light bg-bg-elevated text-text-main">
            <Shield className="w-5 h-5 text-text-main dark:text-dark-text-main" />
          </div>
          <div>
            <h2 className="font-heading text-xl font-semibold text-text-main dark:text-dark-text-main">{t('settings.admin_access')}</h2>
            <p className="mt-1 text-sm text-text-muted dark:text-dark-text-main/70">{t('settings.login_required_desc')}</p>
          </div>
        </div>

        <form className="space-y-4" onSubmit={onSubmit}>
          {adminAuthConfigured === false && (
            <div className="status-banner" data-tone="warning">
              {t('settings.admin_not_configured')}
            </div>
          )}

          <div>
            <label htmlFor="admin-username" className="block text-xs text-text-muted mb-1 dark:text-dark-text-main/70">
              {t('settings.username')}
            </label>
            <div className="relative">
              <User className="w-4 h-4 text-text-muted absolute left-3 top-1/2 -translate-y-1/2 dark:text-dark-text-main/60" />
              <input
                id="admin-username"
                type="text"
                value={loginUsername}
                onChange={(e) => onUsernameChange(e.target.value)}
                className="field-surface w-full pl-9"
                autoComplete="username"
                required
              />
            </div>
          </div>

          <div>
            <label htmlFor="admin-password" className="block text-xs text-text-muted mb-1 dark:text-dark-text-main/70">
              {t('settings.password')}
            </label>
            <input
              id="admin-password"
              type="password"
              value={loginPassword}
              onChange={(e) => onPasswordChange(e.target.value)}
              className="field-surface w-full"
              autoComplete="current-password"
              required
            />
          </div>

          {loginError && (
            <div className="status-banner" data-tone="error">
              {loginError}
            </div>
          )}

          <button
            type="submit"
            disabled={loginLoading || adminAuthConfigured === false}
            className={clsx(
              'header-action w-full',
              (loginLoading || adminAuthConfigured === false) &&
                'cursor-not-allowed border border-border-light bg-bg-hover text-text-dim hover:bg-bg-hover'
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
