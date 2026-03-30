import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import { Eye, Link2, Zap } from 'lucide-react';
import type { GraphSettings } from '../../contexts/GraphContext';

interface SettingsAppearanceProps {
  settings: GraphSettings;
  updateSettings: (s: Partial<GraphSettings>) => void;
}

export const SettingsAppearance = ({ settings, updateSettings }: SettingsAppearanceProps) => {
  const { t } = useTranslation();

  return (
    <section>
      <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-4 px-2 select-none">
        {t('settings.appearance')}
      </h2>
      <div className="space-y-2">
        {/* HQ Rendering toggle */}
        <div className="flex items-center justify-between p-3 rounded-md hover:bg-bg-hover transition-all group">
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
            className={clsx(
              'relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30',
              settings.hqRendering ? 'bg-text-main' : 'bg-border-light dark:bg-dark-border'
            )}
            type="button"
            role="switch"
            aria-checked={settings.hqRendering}
            aria-label={t('settings.hq_rendering')}
            onClick={() => updateSettings({ hqRendering: !settings.hqRendering })}
          >
            <span
              className={clsx(
                'inline-block h-5 w-5 transform rounded-full bg-bg-main transition duration-200 ease-in-out shadow-sm dark:bg-dark-bg-main',
                settings.hqRendering ? 'translate-x-5' : 'translate-x-0.5'
              )}
            />
          </button>
        </div>

        {/* Show Trajectories toggle */}
        <div className="flex items-center justify-between p-3 rounded-md hover:bg-bg-hover transition-all group">
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
            className={clsx(
              'relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30',
              settings.showTrajectories ? 'bg-text-main' : 'bg-border-light dark:bg-dark-border'
            )}
            type="button"
            role="switch"
            aria-checked={settings.showTrajectories}
            aria-label={t('settings.show_trajectories')}
            onClick={() => updateSettings({ showTrajectories: !settings.showTrajectories })}
          >
            <span
              className={clsx(
                'inline-block h-5 w-5 transform rounded-full bg-bg-main transition duration-200 ease-in-out shadow-sm dark:bg-dark-bg-main',
                settings.showTrajectories ? 'translate-x-5' : 'translate-x-0.5'
              )}
            />
          </button>
        </div>

        {/* Related min semantic slider */}
        <div className="p-3 border border-border-light rounded-md space-y-2">
          <div className="flex items-center gap-2">
            <Link2 className="w-4 h-4 text-text-muted" />
            <span className="text-sm font-medium text-text-main">
              {t('settings.related_min_semantic')}
            </span>
          </div>
          <p className="text-xs text-text-muted">{t('settings.related_min_semantic_desc')}</p>
          <div className="flex items-center justify-between">
            <span className="text-xs text-text-muted">{t('repoDetails.similar', 'Similar')}</span>
            <span className="text-xs font-mono tabular-nums text-text-dim">
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
