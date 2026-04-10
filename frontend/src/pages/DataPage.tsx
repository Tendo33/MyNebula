import { useMemo, useState, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useSearchParams, Link } from 'react-router-dom';
import {
  Calendar,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Layers,
  Loader2,
  Star,
  Tag,
  X,
} from 'lucide-react';

import { Sidebar } from '../components/layout/Sidebar';
import { LanguageSwitch } from '../components/layout/LanguageSwitch';
import { SearchInput } from '../components/ui/SearchInput';
import type { DataClusterInfo } from '../api/v2/data';
import { useDataReposQuery } from '../features/data/hooks/useDataReposQuery';
import { getClusterAccent } from '../utils/clusterAccent';

type SortField =
  | 'name'
  | 'language'
  | 'stargazers_count'
  | 'starred_at'
  | 'cluster'
  | 'summary'
  | 'last_commit_time';
type SortDirection = 'asc' | 'desc';

interface SortConfig {
  field: SortField;
  direction: SortDirection;
}

const PAGE_SIZES = [25, 50, 100];
const DEFAULT_PAGE_SIZE = 25;

interface SortableHeaderProps {
  label: string;
  field: SortField;
  currentSort: SortConfig;
  onSort: (field: SortField) => void;
  className?: string;
  align?: 'left' | 'center' | 'right';
}

const SortableHeader: React.FC<SortableHeaderProps> = ({
  label,
  field,
  currentSort,
  onSort,
  className = '',
  align = 'left',
}) => {
  const isActive = currentSort.field === field;
  const justifyClass =
    align === 'center' ? 'justify-center' : align === 'right' ? 'justify-end' : 'justify-start';
  const ariaSort: React.AriaAttributes['aria-sort'] = isActive
    ? currentSort.direction === 'asc'
      ? 'ascending'
      : 'descending'
    : 'none';

  return (
    <th className={`px-4 py-3 whitespace-nowrap ${className}`} scope="col" aria-sort={ariaSort}>
      <button
        type="button"
        onClick={() => onSort(field)}
        className={`flex w-full items-center gap-1 rounded-lg px-1 py-0.5 transition-colors hover:bg-bg-hover ${justifyClass} dark:hover:bg-dark-bg-sidebar/70`}
      >
        <span>{label}</span>
        <span className="flex flex-col">
          <ChevronUp
            className={`-mb-1 h-3 w-3 ${
              isActive && currentSort.direction === 'asc' ? 'text-action-primary' : 'text-text-dim'
            }`}
          />
          <ChevronDown
            className={`h-3 w-3 ${
              isActive && currentSort.direction === 'desc'
                ? 'text-action-primary'
                : 'text-text-dim'
            }`}
          />
        </span>
      </button>
    </th>
  );
};

interface ClusterBadgeProps {
  cluster: DataClusterInfo | undefined;
  onClick?: () => void;
}

const ClusterBadge: React.FC<ClusterBadgeProps> = ({ cluster, onClick }) => {
  const { t } = useTranslation();

  if (!cluster) {
    return <span className="text-xs italic text-text-dim">{t('data.unclustered')}</span>;
  }

  const accent = getClusterAccent({ id: cluster.id, color: cluster.color });

  return (
    <button
      type="button"
      onClick={(event) => {
        event.stopPropagation();
        onClick?.();
      }}
      className="chip-button"
      style={{
        backgroundColor: accent.softBackground,
        borderColor: accent.softBorder,
        color: accent.text,
      }}
    >
      <div className="h-2 w-2 rounded-full" style={{ backgroundColor: accent.dot }} />
      {cluster.name || `Cluster ${cluster.id}`}
    </button>
  );
};

const DataPage = () => {
  const { t } = useTranslation();
  const [searchParams, setSearchParams] = useSearchParams();
  const monthFilter = searchParams.get('month');
  const topicFilter = searchParams.get('topic');

  const [sortConfig, setSortConfig] = useState<SortConfig>({
    field: 'starred_at',
    direction: 'desc',
  });
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [localSearch, setLocalSearch] = useState('');
  const [selectedClusters, setSelectedClusters] = useState<Set<number>>(new Set());
  const offset = (currentPage - 1) * pageSize;

  const { repos, clusters, totalNodes, count, loading, error, retry } = useDataReposQuery({
    searchQuery: localSearch,
    clusterIds: Array.from(selectedClusters),
    month: monthFilter,
    topic: topicFilter,
    sortField: sortConfig.field,
    sortDirection: sortConfig.direction,
    limit: pageSize,
    offset,
  });

  const clusterMap = useMemo(() => {
    const map = new Map<number, DataClusterInfo>();
    clusters.forEach((cluster) => map.set(cluster.id, cluster));
    return map;
  }, [clusters]);

  const totalPages = Math.max(1, Math.ceil(count / pageSize));
  const hasActiveFilters = Boolean(
    selectedClusters.size > 0 || localSearch.trim() || monthFilter || topicFilter
  );

  useEffect(() => {
    if (currentPage > totalPages) {
      setCurrentPage(totalPages);
    }
  }, [currentPage, totalPages]);

  const handleSearch = useCallback((query: string) => {
    setLocalSearch(query);
    setCurrentPage(1);
  }, []);

  const handleSort = useCallback((field: SortField) => {
    setSortConfig((prev) => ({
      field,
      direction: prev.field === field && prev.direction === 'asc' ? 'desc' : 'asc',
    }));
    setCurrentPage(1);
  }, []);

  const handleClusterFilter = useCallback((clusterId: number) => {
    setSelectedClusters((current) => {
      const next = new Set(current);
      if (next.has(clusterId)) {
        next.delete(clusterId);
      } else {
        next.add(clusterId);
      }
      return next;
    });
    setCurrentPage(1);
  }, []);

  const formatDate = (dateStr: string | null | undefined): string => {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <div className="flex min-h-screen bg-bg-main text-text-main dark:bg-dark-bg-main dark:text-dark-text-main">
      <Sidebar />

      <main className="flex min-w-0 flex-1 flex-col" style={{ marginLeft: 'var(--sidebar-width, 240px)' }}>
        <header className="sticky top-0 z-40 flex flex-col gap-3 border-b border-border-light bg-bg-main/92 px-4 py-3 backdrop-blur-md sm:min-h-[4.5rem] sm:flex-row sm:items-center sm:justify-between sm:px-8 dark:border-dark-border dark:bg-dark-bg-main/92">
          <div className="flex items-center gap-3 select-none">
            <h2 className="text-base font-semibold tracking-tight text-text-main">{t('sidebar.data')}</h2>
            <div className="mx-1 h-4 w-px bg-border-light" />
            <span className="rounded-full bg-bg-sidebar/75 px-2.5 py-1 text-xs font-medium text-text-muted dark:bg-dark-bg-sidebar/75 dark:text-dark-text-main/70">
              {count} / {totalNodes} {t('common.repositories')}
            </span>
          </div>

          <div className="flex w-full flex-col gap-3 sm:w-auto sm:flex-row sm:items-center">
            <LanguageSwitch />

            <div className="w-full sm:w-72">
              <SearchInput
                onSearch={handleSearch}
                value={localSearch}
                placeholder={t('data.search_placeholder')}
              />
            </div>

            <div className="flex items-center gap-2 sm:hidden">
              <label className="text-xs text-text-muted">{t('data.sort', 'Sort')}:</label>
              <select
                value={sortConfig.field}
                onChange={(event) => {
                  setSortConfig((prev) => ({
                    field: event.target.value as SortField,
                    direction: prev.direction,
                  }));
                  setCurrentPage(1);
                }}
                className="field-surface h-11 flex-1 px-3 text-sm"
              >
                <option value="starred_at">{t('data.starred_date')}</option>
                <option value="name">{t('data.repository')}</option>
                <option value="stargazers_count">{t('data.stars')}</option>
                <option value="language">{t('data.language')}</option>
                <option value="cluster">{t('data.cluster')}</option>
                <option value="summary">{t('data.summary')}</option>
                <option value="last_commit_time">{t('data.last_commit')}</option>
              </select>
              <button
                type="button"
                onClick={() => {
                  setSortConfig((prev) => ({
                    field: prev.field,
                    direction: prev.direction === 'asc' ? 'desc' : 'asc',
                  }));
                  setCurrentPage(1);
                }}
                className="header-action w-11 px-0"
                aria-label={t('data.sort_direction', 'Toggle sort direction')}
              >
                {sortConfig.direction === 'asc' ? (
                  <ChevronUp className="h-4 w-4" />
                ) : (
                  <ChevronDown className="h-4 w-4" />
                )}
              </button>
            </div>

            {hasActiveFilters && (
              <button
                type="button"
                onClick={() => {
                  setSelectedClusters(new Set());
                  setLocalSearch('');
                  setSearchParams({});
                  setCurrentPage(1);
                }}
                className="header-action-ghost self-start sm:self-auto"
              >
                <X className="h-4 w-4" />
                {t('common.clear_filters')}
              </button>
            )}
          </div>
        </header>

        <div className="flex-1 overflow-auto p-4 sm:p-6">
          {loading ? (
            <div className="flex h-64 items-center justify-center">
              <Loader2 className="h-8 w-8 animate-spin text-text-muted" />
            </div>
          ) : error ? (
            <div className="flex h-64 flex-col items-center justify-center gap-3">
              <p className="text-sm text-red-600">{t('common.load_failed', 'Failed to load data')}</p>
              <button
                type="button"
                onClick={() => {
                  void retry();
                }}
                className="header-action"
              >
                {t('common.retry')}
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              {(clusters.length > 0 || monthFilter || topicFilter) && (
                <div className="panel-subtle flex flex-wrap items-center gap-3 px-4 py-3">
                  {monthFilter && (
                    <div className="flex items-center gap-2">
                      <div className="flex items-center gap-1 text-xs text-text-muted">
                        <Calendar className="h-4 w-4" />
                        <span>{t('data.filter_label')}</span>
                      </div>
                      <span className="chip-button border-action-primary/10 bg-action-primary/10 text-action-primary ring-1 ring-inset ring-action-primary/20">
                        {monthFilter}
                        <button
                          type="button"
                          onClick={() => {
                            const nextParams = new URLSearchParams(searchParams);
                            nextParams.delete('month');
                            setSearchParams(nextParams);
                          }}
                          className="rounded-full p-0.5 hover:bg-action-primary/20"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </span>
                    </div>
                  )}

                  {topicFilter && (
                    <div className="flex items-center gap-2">
                      <div className="flex items-center gap-1 text-xs text-text-muted">
                        <Tag className="h-4 w-4" />
                        <span>{t('data.filter_label')}</span>
                      </div>
                      <span className="chip-button border-action-primary/10 bg-action-primary/10 text-action-primary ring-1 ring-inset ring-action-primary/20">
                        {t('data.topic', 'Topic')}: {topicFilter}
                        <button
                          type="button"
                          onClick={() => {
                            const nextParams = new URLSearchParams(searchParams);
                            nextParams.delete('topic');
                            setSearchParams(nextParams);
                          }}
                          className="rounded-full p-0.5 hover:bg-action-primary/20"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </span>
                    </div>
                  )}

                  {clusters.length > 0 && (
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="flex items-center gap-1 text-xs text-text-muted">
                        <Layers className="h-4 w-4" />
                        <span>{t('data.filter_by_cluster')}:</span>
                      </div>
                      {clusters.map((cluster) => (
                        (() => {
                          const accent = getClusterAccent({ id: cluster.id, color: cluster.color });
                          const selected = selectedClusters.has(cluster.id);

                          return (
                            <button
                              key={cluster.id}
                              type="button"
                              onClick={() => handleClusterFilter(cluster.id)}
                              className={`chip-button ${selected ? 'ring-2 ring-offset-1 shadow-sm' : 'hover:opacity-90'}`}
                              style={
                                {
                                  backgroundColor: selected ? accent.strongBackground : accent.softBackground,
                                  borderColor: selected ? accent.strongBorder : accent.softBorder,
                                  color: accent.text,
                                  '--tw-ring-color': accent.base,
                                } as React.CSSProperties
                              }
                            >
                              <div
                                className="h-2 w-2 rounded-full"
                                style={{ backgroundColor: accent.dot }}
                              />
                              {cluster.name || `Cluster ${cluster.id}`}
                              <span className="opacity-75">({cluster.repo_count})</span>
                            </button>
                          );
                        })()
                      ))}
                    </div>
                  )}
                </div>
              )}

              <div className="panel-surface hidden w-full overflow-hidden sm:block">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="border-b border-border-light bg-bg-hover font-medium text-text-muted dark:border-dark-border dark:bg-dark-bg-sidebar/60 dark:text-dark-text-main/70">
                      <tr>
                        <th className="w-14 px-2 py-3 text-center text-xs text-text-muted/50">#</th>
                        <SortableHeader
                          label={t('data.repository')}
                          field="name"
                          currentSort={sortConfig}
                          onSort={handleSort}
                        />
                        <SortableHeader
                          label={t('data.summary')}
                          field="summary"
                          currentSort={sortConfig}
                          onSort={handleSort}
                        />
                        <SortableHeader
                          label={t('data.language')}
                          field="language"
                          currentSort={sortConfig}
                          onSort={handleSort}
                          align="center"
                        />
                        <SortableHeader
                          label={t('data.stars')}
                          field="stargazers_count"
                          currentSort={sortConfig}
                          onSort={handleSort}
                          align="center"
                        />
                        <SortableHeader
                          label={t('data.cluster')}
                          field="cluster"
                          currentSort={sortConfig}
                          onSort={handleSort}
                          align="center"
                        />
                        <SortableHeader
                          label={t('data.starred_date')}
                          field="starred_at"
                          currentSort={sortConfig}
                          onSort={handleSort}
                          align="center"
                        />
                        <SortableHeader
                          label={t('data.last_commit')}
                          field="last_commit_time"
                          currentSort={sortConfig}
                          onSort={handleSort}
                          align="center"
                        />
                        <th className="hidden w-20 px-4 py-3">{t('data.description')}</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border-light dark:divide-dark-border">
                      {repos.map((repo) => (
                        <tr
                          key={repo.id}
                          className="transition-colors hover:bg-bg-hover/50 dark:hover:bg-dark-bg-sidebar/60"
                        >
                          <td className="px-2 py-3 text-center">
                            {repo.owner_avatar_url ? (
                              <img
                                src={repo.owner_avatar_url}
                                alt={repo.owner}
                                className="mx-auto h-6 min-h-6 w-6 min-w-6 object-cover"
                                loading="lazy"
                                decoding="async"
                                width={24}
                                height={24}
                              />
                            ) : (
                              <div className="mx-auto flex h-6 min-h-6 w-6 min-w-6 items-center justify-center bg-border-light text-[10px] text-text-dim dark:bg-dark-border dark:text-dark-text-main/60">
                                {repo.owner.charAt(0).toUpperCase()}
                              </div>
                            )}
                          </td>
                          <td className="max-w-xs px-4 py-3">
                            <div className="flex items-center gap-2">
                              <Link
                                to={`/graph?node=${repo.id}`}
                                className="block truncate font-medium text-text-main hover:text-action-primary hover:underline"
                              >
                                {repo.full_name}
                              </Link>
                            </div>
                          </td>
                          <td className="max-w-md px-4 py-3">
                            <p
                              className="line-clamp-2 text-sm text-text-muted"
                              title={repo.ai_summary || repo.description}
                            >
                              {repo.ai_summary || repo.description || (
                                <span className="italic text-text-dim">{t('data.no_summary')}</span>
                              )}
                            </p>
                          </td>
                          <td className="px-4 py-3 text-center">
                            {repo.language ? (
                              <span className="inline-flex items-center rounded-full bg-bg-hover px-2.5 py-1 text-xs font-medium text-text-muted dark:bg-dark-bg-sidebar dark:text-dark-text-main/70">
                                {repo.language}
                              </span>
                            ) : (
                              <span className="italic text-text-muted">-</span>
                            )}
                          </td>
                          <td className="px-4 py-3 text-center font-mono tabular-nums text-text-dim">
                            {repo.stargazers_count.toLocaleString()}
                          </td>
                          <td className="px-4 py-3 text-center">
                            <ClusterBadge
                              cluster={repo.cluster_id != null ? clusterMap.get(repo.cluster_id) : undefined}
                              onClick={() => repo.cluster_id != null && handleClusterFilter(repo.cluster_id)}
                            />
                          </td>
                          <td className="whitespace-nowrap px-4 py-3 text-center text-xs text-text-muted">
                            {formatDate(repo.starred_at)}
                          </td>
                          <td className="whitespace-nowrap px-4 py-3 text-center text-xs text-text-muted">
                            {formatDate(repo.last_commit_time)}
                          </td>
                          <td className="hidden max-w-md px-4 py-3">
                            <p className="truncate text-xs text-text-muted">
                              {repo.description || (
                                <span className="italic text-text-dim">{t('data.no_description')}</span>
                              )}
                            </p>
                          </td>
                        </tr>
                      ))}

                      {repos.length === 0 && (
                        <tr>
                          <td colSpan={8} className="px-4 py-12 text-center text-text-muted">
                            {hasActiveFilters ? t('data.no_results') : t('data.no_data')}
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="space-y-3 sm:hidden">
                {repos.map((repo) => (
                  <div key={repo.id} className="panel-surface p-4">
                    <div className="flex items-start gap-3">
                      {repo.owner_avatar_url ? (
                        <img
                          src={repo.owner_avatar_url}
                          alt={repo.owner}
                          className="h-10 w-10 rounded-xl border border-border-light object-cover"
                          loading="lazy"
                          decoding="async"
                          width={40}
                          height={40}
                        />
                      ) : (
                        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-border-light text-text-dim dark:bg-dark-border dark:text-dark-text-main/60">
                          {repo.owner.charAt(0).toUpperCase()}
                        </div>
                      )}
                      <div className="min-w-0 flex-1">
                        <Link
                          to={`/graph?node=${repo.id}`}
                          className="block truncate font-semibold text-text-main hover:text-action-primary"
                        >
                          {repo.full_name}
                        </Link>
                        <p className="mt-1 line-clamp-2 text-xs text-text-muted dark:text-dark-text-main/70">
                          {repo.ai_summary || repo.description || t('data.no_summary')}
                        </p>
                      </div>
                      <div
                        className="inline-flex items-center gap-1.5 rounded-full bg-bg-sidebar/75 px-2.5 py-1 text-xs font-medium text-text-muted dark:bg-dark-bg-sidebar/75 dark:text-dark-text-main/70"
                        aria-label={`${t('data.stars')}: ${repo.stargazers_count.toLocaleString()}`}
                      >
                        <Star className="h-3.5 w-3.5 fill-current" />
                        <span className="tabular-nums">{repo.stargazers_count.toLocaleString()}</span>
                      </div>
                    </div>

                    <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-text-muted dark:text-dark-text-main/70">
                      {repo.language && (
                        <span className="inline-flex items-center rounded-full bg-bg-hover px-2.5 py-1 text-text-muted dark:bg-dark-bg-sidebar dark:text-dark-text-main/70">
                          {repo.language}
                        </span>
                      )}
                      <ClusterBadge
                        cluster={repo.cluster_id != null ? clusterMap.get(repo.cluster_id) : undefined}
                        onClick={() => repo.cluster_id != null && handleClusterFilter(repo.cluster_id)}
                      />
                      <span>{t('data.starred_date')}: {formatDate(repo.starred_at)}</span>
                      <span>{t('data.last_commit')}: {formatDate(repo.last_commit_time)}</span>
                    </div>
                  </div>
                ))}

                {repos.length === 0 && (
                  <div className="panel-surface p-6 text-center text-sm text-text-muted">
                    {hasActiveFilters ? t('data.no_results') : t('data.no_data')}
                  </div>
                )}
              </div>

              {count > 0 && (
                <div className="flex flex-col gap-3 text-sm sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-center gap-2 text-text-muted">
                    <span>{t('data.rows_per_page')}:</span>
                    <select
                      value={pageSize}
                      onChange={(event) => {
                        setPageSize(Number(event.target.value));
                        setCurrentPage(1);
                      }}
                      className="field-surface h-10 px-3 text-sm"
                    >
                      {PAGE_SIZES.map((size) => (
                        <option key={size} value={size}>
                          {size}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="text-text-muted">
                      {t('data.showing', {
                        start: offset + 1,
                        end: offset + repos.length,
                        total: count,
                      })}
                    </span>

                    <div className="flex items-center gap-1">
                      <button
                        type="button"
                        onClick={() => setCurrentPage((page) => Math.max(1, page - 1))}
                        disabled={currentPage === 1}
                        className="header-action h-10 min-h-0 w-10 px-0 disabled:cursor-not-allowed disabled:opacity-30"
                        aria-label={t('data.previous_page', 'Previous page')}
                      >
                        <ChevronLeft className="h-4 w-4" />
                      </button>

                      <span className="px-3 py-1 font-medium text-text-main">
                        {currentPage} / {totalPages || 1}
                      </span>

                      <button
                        type="button"
                        onClick={() => setCurrentPage((page) => Math.min(totalPages, page + 1))}
                        disabled={currentPage === totalPages || totalPages === 0}
                        className="header-action h-10 min-h-0 w-10 px-0 disabled:cursor-not-allowed disabled:opacity-30"
                        aria-label={t('data.next_page', 'Next page')}
                      >
                        <ChevronRight className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default DataPage;
