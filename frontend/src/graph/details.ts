import { avatarUrl, html, type Raw } from '../html';
import { t } from '../i18n';
import { icon } from '../icons';
import { normalizeTag } from '../search';
import { store } from '../store';
import type { GraphNode } from '../types';

export type RelatedTab = 'similar' | 'sameTags' | 'sameLang';
type RelatedEntry = { repo: GraphNode; reason?: string; score?: number };

const factTopics = (topics: string[] = [], tags: string[] = []): string[] => {
  const seen = new Set<string>();
  const labels: string[] = [];
  for (const raw of [...topics, ...tags]) {
    const label = raw.trim();
    if (!label || seen.has(label.toLowerCase())) continue;
    seen.add(label.toLowerCase());
    labels.push(label);
    if (labels.length === 4) break;
  }
  return labels;
};

const RELATED_LIMIT = 20;

/** Readable reasons for a precomputed related item, same thresholds the API used. */
const reasonsFor = (semantic: number, overlap: number, flags: number): string[] => {
  const reasons: string[] = [];
  if (semantic >= 0.78) reasons.push(t('repoDetails.reason_very_similar'));
  else if (semantic >= 0.65) reasons.push(t('repoDetails.reason_similar'));
  if (overlap > 0) reasons.push(t('repoDetails.reason_tags'));
  if (flags & 1) reasons.push(t('repoDetails.reason_list'));
  if (flags & 2) reasons.push(t('repoDetails.reason_language'));
  return reasons;
};

export const relatedFor = (node: GraphNode): Record<RelatedTab, RelatedEntry[]> => {
  const similar: RelatedEntry[] = (store.related?.[String(node.id)] ?? [])
    .filter(([, , semantic]) => semantic >= store.settings.relatedMinSemantic)
    .slice(0, RELATED_LIMIT)
    .flatMap(([id, score, semantic, overlap, flags]) => {
      const repo = store.nodeById.get(id);
      return repo ? [{ repo, score, reason: reasonsFor(semantic, overlap, flags).join(' · ') }] : [];
    });

  const tagsOf = (n: GraphNode) => new Set([...(n.ai_tags ?? []), ...(n.topics ?? [])].map(normalizeTag).filter(Boolean));
  const nodeTags = tagsOf(node);
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

const relatedItem = ({ repo, reason, score }: RelatedEntry): Raw => {
  const stars = repo.stargazers_count >= 1000 ? `${(repo.stargazers_count / 1000).toFixed(1)}k` : String(repo.stargazers_count);
  return html`<button class="related-item" data-act="select" data-id="${repo.id}">
    ${avatar(repo, 'sm')}
    <span class="related-body">
      <span class="related-title"><span class="truncate strong">${repo.name}</span>${repo.language ? html`<span class="dim small">${repo.language}</span>` : ''}</span>
      <span class="muted small clamp-1">${repo.description || repo.ai_summary || t('common.no_description')}</span>
      ${reason || score != null
        ? html`<span class="dim small truncate">${reason ?? ''}${reason && score != null ? ' · ' : ''}${score != null ? `${(score * 100).toFixed(1)}%` : ''}</span>`
        : ''}
    </span>
    <span class="mono dim small">${stars}</span>
  </button>`;
};

export interface DetailsState {
  relatedTab: RelatedTab;
}

/** Summary, facts, links, then related repos — one scroll, no tabs. */
export const renderDetails = (node: GraphNode, state: DetailsState): Raw => {
  const listName = store.data?.star_lists.find((l) => l.id === node.star_list_id && l.id !== -1)?.name;
  const cluster = store.data?.clusters.find((c) => c.id === node.cluster_id);
  const topics = factTopics(node.topics ?? [], node.ai_tags ?? []);
  const summary = (node.ai_summary || node.description || '').trim();
  const related = relatedFor(node);
  const current = related[state.relatedTab];
  const loading = state.relatedTab === 'similar' && !store.related && !store.relatedError;
  const emptyMessage =
    state.relatedTab === 'similar'
      ? store.relatedError
        ? t('repoDetails.similarLoadFailed')
        : t('repoDetails.noSimilar')
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
        ${node.language ? html`<div><dt>${t('data.language')}</dt><dd>${node.language}</dd></div>` : ''}
        ${cluster ? html`<div><dt>${t('data.cluster')}</dt><dd><span class="dot" style="--dot:${cluster.color}"></span>${cluster.name}</dd></div>` : ''}
        ${listName ? html`<div><dt>${t('graph.my_lists')}</dt><dd>${listName}</dd></div>` : ''}
      </dl>
      <p class="details-summary">${summary || t('repoDetails.no_ai_summary')}</p>
      ${topics.length ? html`<div class="badges">${topics.map((topic) => html`<span class="badge">${topic}</span>`)}</div>` : ''}
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
          : html`<p class="muted small related-empty">${loading ? t('common.loading') : emptyMessage}</p>`}
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
