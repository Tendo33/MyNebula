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
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-bg-sidebar text-text-main shadow-sm">
            <Shield className="w-5 h-5 text-text-main" />
          </div>
          <div>
            <div className="section-kicker mb-2 px-0">{t('settings.title')}</div>
            <h2 className="font-heading text-xl font-semibold text-text-main">{t('settings.admin_access')}</h2>
            <p className="mt-1 text-sm text-text-muted">{t('settings.login_required_desc')}</p>
          </div>
        </div>

        <form className="space-y-4" onSubmit={onSubmit}>
          {adminAuthConfigured === false && (
            <div className="status-banner" data-tone="warning">
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
                className="field-surface h-11 w-full pl-9 pr-3 text-sm"
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
              className="field-surface h-11 w-full px-3 text-sm"
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
              'flex h-12 w-full items-center justify-center gap-2 rounded-2xl text-sm font-semibold transition-colors',
              loginLoading || adminAuthConfigured === false
                ? 'bg-bg-hover text-text-dim border border-border-light cursor-not-allowed dark:bg-dark-bg-sidebar/70 dark:text-dark-text-main/60 dark:border-dark-border'
                : 'bg-text-main text-bg-main shadow-sm hover:-translate-y-px hover:bg-text-main/92'
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
