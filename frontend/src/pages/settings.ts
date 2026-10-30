import { html, onAction } from '../html';
import { getLang, setLang, t } from '../i18n';
import { store } from '../store';
import { getTheme, setTheme, type ThemePreference } from '../theme';

const row = (title: string, desc: string, control: unknown) =>
  html`<div class="setting">
    <div class="setting-text"><span class="setting-title">${title}</span><span class="muted small">${desc}</span></div>
    <div class="setting-control">${control}</div>
  </div>`;

const toggleSwitch = (key: string, checked: boolean, label: string) =>
  html`<button role="switch" class="switch" data-act="switch" data-key="${key}" aria-checked="${checked}" aria-label="${label}"><span></span></button>`;

const segmented = (act: string, label: string, current: string, options: [string, string][]) =>
  html`<div class="toggle-group" role="group" aria-label="${label}">${options.map(
    ([value, text]) => html`<button class="toggle-item" data-act="${act}" data-value="${value}" aria-pressed="${current === value}">${text}</button>`,
  )}</div>`;

export const mountSettings = (root: HTMLElement): (() => void) => {
  const render = () => {
    const { settings, data } = store;
    root.innerHTML = html`<section class="settings">
      <h2>${t('settings.appearance')}</h2>
      <div class="setting-list">
        ${row(t('settings.theme'), t('settings.theme_desc'), segmented('theme', t('settings.theme'), getTheme(), [
          ['light', t('settings.theme_light')],
          ['dark', t('settings.theme_dark')],
          ['system', t('settings.theme_system')],
        ]))}
        ${row(t('settings.language'), t('settings.language_desc'), segmented('lang', t('settings.language'), getLang(), [
          ['en', 'English'],
          ['zh', '中文'],
        ]))}
        ${row(t('settings.hq_rendering'), t('settings.hq_rendering_desc'), toggleSwitch('hqRendering', settings.hqRendering, t('settings.hq_rendering')))}
        ${row(t('settings.show_trajectories'), t('settings.show_trajectories_desc'), toggleSwitch('showTrajectories', settings.showTrajectories, t('settings.show_trajectories')))}
        ${row(
          t('settings.related_min_semantic'),
          t('settings.related_min_semantic_desc'),
          html`<span class="range-field"><input type="range" class="range" min="0.5" max="0.9" step="0.01" value="${settings.relatedMinSemantic}"
            data-semantic aria-label="${t('settings.related_min_semantic')}" /><output class="tabular" data-region="semantic">${settings.relatedMinSemantic.toFixed(2)}</output></span>`,
        )}
      </div>

      <h2>${t('settings.data')}</h2>
      <dl class="setting-list facts-list">
        <div><dt>${t('settings.generated_at')}</dt><dd>${data ? new Date(data.generated_at).toLocaleString(getLang(), { dateStyle: 'medium', timeStyle: 'short' }) : '—'}</dd></div>
        <div><dt>${t('settings.repos')}</dt><dd class="tabular">${data ? `${data.nodes.length.toLocaleString()} / ${data.total_repos.toLocaleString()}` : '—'}</dd></div>
        <div><dt>${t('settings.groups_links')}</dt><dd class="tabular">${data ? `${data.clusters.length} · ${data.edges.length.toLocaleString()}` : '—'}</dd></div>
      </dl>

      <h2>${t('settings.sync')}</h2>
      <div class="sync-help">
        <p class="muted">${t('settings.sync_desc')}</p>
        <pre class="code">pnpm sync</pre>
        <p class="muted small">${t('settings.sync_flags')}</p>
      </div>
    </section>`.value;
  };

  render();
  const unsubscribe = store.subscribe(['data'], render);
  const offClick = onAction(root, 'click', {
    theme: (button, event) => {
      const { clientX: x, clientY: y } = event as MouseEvent;
      setTheme(button.dataset.value as ThemePreference, x || y ? { x, y } : undefined);
      render();
    },
    lang: (button) => setLang(button.dataset.value as 'en' | 'zh'),
    switch: (button) => {
      const key = button.dataset.key as 'hqRendering' | 'showTrajectories';
      store.updateSettings({ [key]: !store.settings[key] });
      button.setAttribute('aria-checked', String(store.settings[key]));
    },
  });
  const onInput = (event: Event) => {
    const target = event.target as HTMLInputElement;
    if (!target.matches('[data-semantic]')) return;
    store.updateSettings({ relatedMinSemantic: Number(target.value) });
    root.querySelector('[data-region="semantic"]')!.textContent = Number(target.value).toFixed(2);
  };
  root.addEventListener('input', onInput);
  return () => {
    unsubscribe();
    offClick();
    root.removeEventListener('input', onInput);
  };
};
