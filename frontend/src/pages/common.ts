import { html, type Raw } from '../html';
import { getLang, t } from '../i18n';
import { icon } from '../icons';

export const loadingState = (): Raw =>
  html`<div class="center-state" role="status"><span class="spinner"></span><span class="muted">${t('common.loading')}</span></div>`;

export const errorState = (): Raw =>
  html`<div class="center-state">
    <p>${t('common.load_failed')}</p>
    <button class="btn btn-primary" data-act="retry">${t('common.retry')}</button>
  </div>`;

/** Empty collection: point at the sync instructions on the Settings page. */
export const emptyState = (iconName: string, title: string, hint: string): Raw =>
  html`<div class="center-state">
    <span class="empty-icon">${icon(iconName)}</span>
    <h2 class="empty-title">${title}</h2>
    <p class="muted">${hint}</p>
    <a class="btn btn-primary" href="#/settings">${t('common.sync_now')}</a>
  </div>`;

export const formatDate = (value: string | null | undefined): string =>
  value ? new Date(value).toLocaleDateString(getLang(), { year: 'numeric', month: 'short', day: 'numeric' }) : '—';
