import { avatarUrl, html, type Raw } from '../html';
import { t } from '../i18n';
import { icon } from '../icons';
import { href } from '../router';
import { detailTopicLabels, normalizeTag } from '../search';
import { store } from '../store';
import { UNCATEGORIZED_LIST_ID, type GraphNode } from '../types';

export type RelatedTab = 'similar' | 'sameTags' | 'sameLang';
const RELATED_TABS: RelatedTab[] = ['similar', 'sameTags', 'sameLang'];

/** Keep the keyboard on the related-tab control after the details panel is rebuilt. */
export const focusRelatedTab = (root: ParentNode, tab: RelatedTab): void => {
  const match = [...root.querySelectorAll<HTMLElement>('[data-region="details"] [data-act="related-tab"]')].find(
    (button) => button.dataset.tab === tab,
  );
  match?.focus({ preventScroll: true });
};

/** Left and right move between the related-repo tabs. Returns the tab to select. */
export const moveRelatedTab = (event: KeyboardEvent): RelatedTab | null => {
  if ((event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') || event.altKey || event.metaKey || event.ctrlKey) return null;
  const active = document.activeElement;
  if (!(active instanceof HTMLElement) || active.dataset.act !== 'related-tab') return null;
  const index = RELATED_TABS.indexOf(active.dataset.tab as RelatedTab);
  if (index < 0) return null;
  event.preventDefault();
  const step = event.key === 'ArrowRight' ? 1 : RELATED_TABS.length - 1;
  return RELATED_TABS[(index + step) % RELATED_TABS.length];
};

const focusItem = (item: HTMLElement) => {
  item.focus({ preventScroll: true });
  item.scrollIntoView({ block: 'nearest' });
};

/** After opening a related repository, keep the keyboard on the new list. */
export const focusRelatedItem = (root: ParentNode): void => {
  const region = root.querySelector('[data-region="details"]');
  if (!(region instanceof HTMLElement)) return;
  const item = region.querySelector<HTMLElement>('.related-item');
  if (item) {
    focusItem(item);
    return;
  }
  region.querySelector<HTMLElement>('[data-act="related-tab"][aria-pressed="true"]')?.focus({ preventScroll: true });
};

/** Up and down move through the related repositories under the focused one. */
export const moveRelatedItem = (event: KeyboardEvent, root: ParentNode): boolean => {
  const key = event.key;
  if (key !== 'ArrowDown' && key !== 'ArrowUp' && key !== 'Home' && key !== 'End') return false;
  if (event.altKey || event.metaKey || event.ctrlKey) return false;
  const active = document.activeElement;
  if (!(active instanceof HTMLElement) || !root.contains(active)) return false;
  if (active.dataset.act === 'related-tab' && key === 'ArrowDown') {
    const item = active.closest('[data-region="details"]')?.querySelector<HTMLElement>('.related-item');
    if (!item) return false;
    event.preventDefault();
    focusItem(item);
    return true;
  }
  if (!active.classList.contains('related-item')) return false;
  const list = active.closest('.related-list');
  if (!(list instanceof HTMLElement)) return false;
  const items = [...list.querySelectorAll<HTMLButtonElement>('.related-item')];
  if (!items.length) return false;
  const index = items.indexOf(active as HTMLButtonElement);
  if (key === 'ArrowUp' && index <= 0) {
    const tab = list.closest('.details-related')?.querySelector<HTMLElement>('[data-act="related-tab"][aria-pressed="true"]');
    if (!tab) return false;
    event.preventDefault();
    tab.focus({ preventScroll: true });
    return true;
  }
  const next =
    key === 'Home' ? 0
    : key === 'End' ? items.length - 1
    : key === 'ArrowDown' ? (index < 0 ? 0 : Math.min(items.length - 1, index + 1))
    : Math.max(0, index - 1);
  event.preventDefault();
  if (next !== index) focusItem(items[next]);
  return true;
};

type RelatedEntry = { repo: GraphNode; reason?: string };

export const relatedFor = (node: GraphNode): Record<RelatedTab, RelatedEntry[]> => {
  const tagsOf = (n: GraphNode) => new Set([...(n.ai_tags ?? []), ...(n.topics ?? [])].map(normalizeTag).filter(Boolean));
  const nodeTags = tagsOf(node);

  // Exactly the repos this node is linked to on the canvas, strongest link first.
  const similar: RelatedEntry[] = (store.data?.edges ?? [])
    .flatMap(([s, t, score]) => (s === node.id ? [{ id: t, score }] : t === node.id ? [{ id: s, score }] : []))
    .sort((a, b) => b.score - a.score)
    .flatMap(({ id }) => {
      const repo = store.nodeById.get(id);
      if (!repo) return [];
      const shared = [...tagsOf(repo)].filter((tag) => nodeTags.has(tag)).slice(0, 2).join(', ');
      const lists = new Set(node.star_list_ids.filter((id) => id !== UNCATEGORIZED_LIST_ID));
      const sameList = repo.star_list_ids.some((id) => lists.has(id));
      return [{ repo, reason: [shared, sameList ? t('repoDetails.reason_list') : ''].filter(Boolean).join(' · ') }];
    });
  const sameTags: RelatedEntry[] = nodeTags.size
    ? store.nodes
        .filter((n) => n.id !== node.id)
        .map((n) => {
          const other = tagsOf(n);
          const overlap = [...nodeTags].filter((tag) => other.has(tag));
          const ratio = overlap.length / Math.max(Math.min(nodeTags.size, other.size), 1);
          return { repo: n, overlap, ratio };
        })
        .filter(({ overlap, ratio }) => overlap.length >= 1 && ratio >= 0.25)
        .sort((a, b) => b.repo.stargazers_count - a.repo.stargazers_count)
        .slice(0, 10)
        .map(({ repo, overlap }) => ({ repo, reason: overlap.slice(0, 2).join(', ') }))
    : [];

  const sameLang: RelatedEntry[] = node.language
    ? store.nodes
        .filter((n) => n.id !== node.id && n.language === node.language)
        .sort((a, b) => b.stargazers_count - a.stargazers_count)
        .slice(0, 10)
        .map((repo) => ({ repo }))
    : [];

  return { similar, sameTags, sameLang };
};

const avatar = (node: GraphNode, size: 'sm' | 'md') => {
  const url = avatarUrl(node.owner_avatar_url, size === 'sm' ? 48 : 80);
  return url
    ? html`<img class="avatar avatar-${size}" src="${url}" alt="" loading="lazy" decoding="async" />`
    : html`<span class="avatar avatar-${size} avatar-fallback">${node.owner.charAt(0).toUpperCase()}</span>`;
};

const relatedItem = ({ repo, reason }: RelatedEntry): Raw => {
  const stars = repo.stargazers_count >= 1000 ? `${(repo.stargazers_count / 1000).toFixed(1)}k` : String(repo.stargazers_count);
  return html`<button class="related-item" data-act="select" data-id="${repo.id}">
    ${avatar(repo, 'sm')}
    <span class="related-body">
      <span class="related-title"><span class="truncate strong">${repo.name}</span>${repo.language ? html`<span class="dim small">${repo.language}</span>` : ''}</span>
      <span class="muted small clamp-1">${repo.description || repo.ai_summary || t('common.no_description')}</span>
      ${reason ? html`<span class="dim small truncate">${reason}</span>` : ''}
    </span>
    <span class="related-stars dim small tabular">${stars}</span>
  </button>`;
};

export interface DetailsState {
  relatedTab: RelatedTab;
  /** Outside the graph (the Data page), lead the links with a way back to the node on the graph. */
  graphLink?: boolean;
  /** Clusters currently filtered, so the fact button can show pressed. */
  clusters?: ReadonlySet<number>;
  /** Languages currently filtered, so the fact button can show pressed. */
  languages?: ReadonlySet<string>;
  /** Lists currently filtered. Omit it to keep list names as text. */
  lists?: ReadonlySet<string>;
  /**
   * When set, GitHub topic badges toggle this exact filter. Null means the
   * control is available and nothing is selected. Omit it to keep badges as labels.
   */
  topic?: string | null;
}

/** Summary, facts, links, then related repos — one scroll, no tabs. */
export const renderDetails = (node: GraphNode, state: DetailsState): Raw => {
  const lists = (store.data?.star_lists ?? []).filter(
    (list) => list.id !== UNCATEGORIZED_LIST_ID && node.star_list_ids.includes(list.id),
  );
  const cluster = store.data?.clusters.find((c) => c.id === node.cluster_id);
  const topics = detailTopicLabels(node.topics ?? [], node.ai_tags ?? []);
  const selectedTopic = state.topic?.trim().toLowerCase() ?? '';
  const summary = (node.ai_summary || node.description || '').trim();
  const related = relatedFor(node);
  const current = related[state.relatedTab];
  const emptyMessage =
    state.relatedTab === 'similar'
      ? t('repoDetails.noSimilar')
      : state.relatedTab === 'sameTags'
        ? t('repoDetails.noSameTags')
        : t('repoDetails.noSameLang');
  const toggle = (tab: RelatedTab, label: string) =>
    html`<button class="toggle-item" data-act="related-tab" data-tab="${tab}" aria-pressed="${state.relatedTab === tab}">
      ${label}${related[tab].length ? html`<span class="toggle-count">${related[tab].length}</span>` : ''}
    </button>`;

  return html`<div class="details">
    <div class="details-header">
      ${avatar(node, 'md')}
      <div class="details-heading">
        <a href="${node.html_url}" target="_blank" rel="noopener noreferrer" class="details-name" title="${node.full_name}">${node.name}</a>
        <span class="muted small truncate">${node.owner}</span>
      </div>
      <button class="btn btn-ghost btn-icon btn-sm" data-act="close-details" aria-label="${t('repoDetails.close')}">${icon('x')}</button>
    </div>
    <div class="details-body">
      <dl class="facts">
        <div><dt>${t('repoDetails.stars')}</dt><dd class="tabular">${node.stargazers_count.toLocaleString()}</dd></div>
        ${node.language
          ? html`<div><dt>${t('data.language')}</dt><dd><button class="cluster-tag" data-act="language" data-id="${node.language}" aria-pressed="${state.languages?.has(node.language) ?? false}" title="${t('search.filter_by_language', { language: node.language })}">${node.language}</button></dd></div>`
          : ''}
        ${cluster
          ? html`<div><dt>${t('data.cluster')}</dt><dd><button class="cluster-tag" data-act="cluster" data-id="${cluster.id}" aria-pressed="${state.clusters?.has(cluster.id) ?? false}" title="${t('search.filter_by_cluster', { cluster: cluster.name })}"><span class="dot" style="--dot:${cluster.color}"></span><span class="truncate">${cluster.name}</span></button></dd></div>`
          : ''}
        ${lists.length
          ? html`<div><dt>${t('graph.my_lists')}</dt><dd>${
              state.lists
                ? lists.map(
                    (list) => html`<button class="cluster-tag" data-act="list" data-id="${list.id}" aria-pressed="${state.lists?.has(list.id) ?? false}" title="${list.name}">${list.name}</button>`,
                  )
                : lists.map((list) => list.name).join(' · ')
            }</dd></div>`
          : ''}
      </dl>
      <p class="details-summary">${summary || t('repoDetails.no_ai_summary')}</p>
      ${topics.length
        ? html`<div class="badges">${topics.map(({ label, filterable }) =>
            state.topic !== undefined && filterable
              ? html`<button class="badge" data-act="topic" data-id="${label}" aria-pressed="${selectedTopic === label.toLowerCase()}" title="${t('search.filter_by_topic', { topic: label })}">${label}</button>`
              : html`<span class="badge">${label}</span>`,
          )}</div>`
        : ''}
      ${state.graphLink ? html`<a class="btn btn-outline btn-sm details-graph-link" href="${href('/', { node: String(node.id) })}">${t('repoDetails.view_in_graph')}</a>` : ''}
      <div class="details-links">
        <a href="${node.html_url}" target="_blank" rel="noopener noreferrer">GitHub ${icon('external')}</a>
        <a href="https://deepwiki.com/${node.full_name}" target="_blank" rel="noopener noreferrer">DeepWiki ${icon('external')}</a>
        <a href="https://zread.ai/${node.full_name}" target="_blank" rel="noopener noreferrer">zRead ${icon('external')}</a>
      </div>
      <section class="details-related">
        <h3>${t('repoDetails.relatedRepos')}</h3>
        <div class="toggle-group">
          ${toggle('similar', t('repoDetails.similar'))}
          ${toggle('sameTags', t('repoDetails.sameTags'))}
          ${toggle('sameLang', t('repoDetails.sameLang'))}
        </div>
        ${current.length
          ? html`<div class="related-list">${current.map(relatedItem)}</div>`
          : html`<p class="muted small related-empty">${emptyMessage}</p>`}
      </section>
    </div>
  </div>`;
};

export const renderHoverCard = (node: GraphNode): Raw => html`<div class="hover-card">
  <div class="hover-card-head">
    ${avatar(node, 'sm')}
    <strong class="truncate">${node.full_name}</strong>
    <span class="stars muted small tabular nowrap">${icon('star')}${node.stargazers_count.toLocaleString()}</span>
  </div>
  ${node.description || node.ai_summary ? html`<p class="muted small clamp-3">${node.description || node.ai_summary}</p>` : ''}
  ${node.language ? html`<span class="dim xsmall">${node.language}</span>` : ''}
</div>`;
