import { useTranslation } from 'react-i18next';
import { Eye, Link2, Monitor, Moon, Sun, Zap } from 'lucide-react';

import { useTheme, type ThemePreference } from '../../hooks/useTheme';
import type { GraphSettings } from '../../contexts/GraphContext';
import { Card } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';

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
      <div className="flex flex-col gap-2">
        {/* Theme */}
        <Card variant="muted" className="p-4 transition-all group">
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
            <ToggleGroup
              value={[preference]}
              onValueChange={(next, eventDetails) => {
                const value = next[0];
                if (!value) return;
                const nativeEvent = eventDetails?.event as MouseEvent | PointerEvent | undefined;
                const origin =
                  nativeEvent && 'clientX' in nativeEvent
                    ? { clientX: nativeEvent.clientX, clientY: nativeEvent.clientY }
                    : undefined;
                setPreference(value as ThemePreference, origin);
              }}
              aria-label={t('settings.theme', 'Theme')}
              className="rounded-md border border-border bg-card p-0.5"
            >
              {THEME_OPTIONS.map(({ value, label, Icon }) => (
                <ToggleGroupItem key={value} value={value} aria-label={label}>
                  <Icon aria-hidden="true" data-icon="inline-start" />
                  {label}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          </div>
        </Card>

        {/* HQ Rendering toggle */}
        <Card variant="muted" className="flex flex-row items-center justify-between p-4 transition-all group">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-bg-main transition-colors dark:group-hover:bg-dark-bg-main">
              <Zap className="w-5 h-5 text-text-muted group-hover:text-text-main" />
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-medium text-text-main">{t('settings.hq_rendering')}</span>
              <span className="text-xs text-text-muted">{t('settings.hq_rendering_desc')}</span>
            </div>
          </div>
          <Switch
            checked={settings.hqRendering}
            onCheckedChange={(checked) => updateSettings({ hqRendering: checked })}
            aria-label={t('settings.hq_rendering')}
          />
        </Card>

        {/* Show Trajectories toggle */}
        <Card variant="muted" className="flex flex-row items-center justify-between p-4 transition-all group">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-bg-main transition-colors dark:group-hover:bg-dark-bg-main">
              <Eye className="w-5 h-5 text-text-muted group-hover:text-text-main" />
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-medium text-text-main">{t('settings.show_trajectories')}</span>
              <span className="text-xs text-text-muted">{t('settings.show_trajectories_desc')}</span>
            </div>
          </div>
          <Switch
            checked={settings.showTrajectories}
            onCheckedChange={(checked) => updateSettings({ showTrajectories: checked })}
            aria-label={t('settings.show_trajectories')}
          />
        </Card>

        {/* Related min semantic slider */}
        <Card variant="muted" className="flex flex-col gap-3 p-4">
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
          <Slider
            min={0.5}
            max={0.9}
            step={0.01}
            value={settings.relatedMinSemantic}
            onValueChange={(value) => {
              const next = Array.isArray(value) ? value[0] : value;
              if (typeof next === 'number') {
                updateSettings({ relatedMinSemantic: next });
              }
            }}
          />
        </Card>
      </div>
    </section>
  );
};
