import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import { Eye, Link2, Monitor, Moon, Sun, Zap } from 'lucide-react';

import { useTheme, type ThemePreference } from '../../hooks/useTheme';
import type { GraphSettings } from '../../contexts/GraphContext';

interface SettingsAppearanceProps {
  settings: GraphSettings;
  updateSettings: (s: Partial<GraphSettings>) => void;
}

export const SettingsAppearance = ({ settings, updateSettings }: SettingsAppearanceProps) => {
  const { t } = useTranslation();
  const { preference, setPreference } = useTheme();

  const THEME_OPTIONS: { value: ThemePreference; label: string; Icon: typeof Sun }[] = [
    { value: 'light', label: t('settings.theme_light', 'Light'), Icon: Sun },
    { value: 'dark', label: t('settings.theme_dark', 'Dark'), Icon: Moon },
    { value: 'system', label: t('settings.theme_system', 'System'), Icon: Monitor },
  ];

  return (
    <section>
      <h2 className="section-heading mb-4 select-none">
        {t('settings.appearance')}
      </h2>
      <div className="space-y-2">
        {/* Theme */}
        <div className="panel-subtle p-4 transition-all group">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-bg-main transition-colors dark:group-hover:bg-dark-bg-main">
                <Moon className="w-5 h-5 text-text-muted group-hover:text-text-main" />
              </div>
              <div className="flex flex-col">
                <span className="text-sm font-medium text-text-main">
                  {t('settings.theme', 'Theme')}
                </span>
                <span className="text-xs text-text-muted">
                  {t('settings.theme_desc', 'Follow the system, or pick one.')}
                </span>
              </div>
            </div>
            <div
              role="radiogroup"
              aria-label={t('settings.theme', 'Theme')}
              className="flex items-center gap-1 rounded-md border border-border-light bg-bg-elevated p-0.5 dark:border-dark-border"
            >
              {THEME_OPTIONS.map(({ value, label, Icon }) => (
                <button
                  key={value}
                  type="button"
                  role="radio"
                  aria-checked={preference === value}
                  onClick={() => setPreference(value)}
                  className={clsx(
                    'inline-flex h-10 items-center gap-1.5 rounded-md px-3 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-link',
                    preference === value
                      ? 'bg-bg-hover text-text-main dark:bg-dark-bg-sidebar dark:text-dark-text-main'
                      : 'text-text-muted hover:bg-bg-hover hover:text-text-main dark:text-dark-text-main/70 dark:hover:bg-dark-bg-sidebar'
                  )}
                >
                  <Icon aria-hidden="true" className="h-3.5 w-3.5" />
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* HQ Rendering toggle */}
        <div className="panel-subtle flex items-center justify-between p-4 transition-all group">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-bg-main transition-colors dark:group-hover:bg-dark-bg-main">
              <Zap className="w-5 h-5 text-text-muted group-hover:text-text-main" />
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-medium text-text-main">{t('settings.hq_rendering')}</span>
              <span className="text-xs text-text-muted">{t('settings.hq_rendering_desc')}</span>
            </div>
          </div>
          <button
            className={clsx('toggle-control')}
            data-state={settings.hqRendering ? 'on' : 'off'}
            type="button"
            role="switch"
            aria-checked={settings.hqRendering}
            aria-label={t('settings.hq_rendering')}
            onClick={() => updateSettings({ hqRendering: !settings.hqRendering })}
          >
            <span
              className={clsx(
                'toggle-handle',
                settings.hqRendering ? 'translate-x-6' : 'translate-x-0.5'
              )}
            />
          </button>
        </div>

        {/* Show Trajectories toggle */}
        <div className="panel-subtle flex items-center justify-between p-4 transition-all group">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-bg-main transition-colors dark:group-hover:bg-dark-bg-main">
              <Eye className="w-5 h-5 text-text-muted group-hover:text-text-main" />
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-medium text-text-main">{t('settings.show_trajectories')}</span>
              <span className="text-xs text-text-muted">{t('settings.show_trajectories_desc')}</span>
            </div>
          </div>
          <button
            className={clsx('toggle-control')}
            data-state={settings.showTrajectories ? 'on' : 'off'}
            type="button"
            role="switch"
            aria-checked={settings.showTrajectories}
            aria-label={t('settings.show_trajectories')}
            onClick={() => updateSettings({ showTrajectories: !settings.showTrajectories })}
          >
            <span
              className={clsx(
                'toggle-handle',
                settings.showTrajectories ? 'translate-x-6' : 'translate-x-0.5'
              )}
            />
          </button>
        </div>

        {/* Related min semantic slider */}
        <div className="panel-subtle space-y-3 p-4">
          <div className="flex items-center gap-2">
            <Link2 className="w-4 h-4 text-text-muted" />
            <span className="text-sm font-medium text-text-main">
              {t('settings.related_min_semantic')}
            </span>
          </div>
          <p className="text-xs text-text-muted">{t('settings.related_min_semantic_desc')}</p>
          <div className="flex items-center justify-between">
            <span className="text-xs text-text-muted">{t('repoDetails.similar', 'Similar')}</span>
            <span className="text-xs font-mono tabular-nums text-text-muted">
              {settings.relatedMinSemantic.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={0.5}
            max={0.9}
            step={0.01}
            value={settings.relatedMinSemantic}
            onChange={(e) => updateSettings({ relatedMinSemantic: Number(e.target.value) })}
            className="w-full"
          />
        </div>
      </div>
    </section>
  );
};
