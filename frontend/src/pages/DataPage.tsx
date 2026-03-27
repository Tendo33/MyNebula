import { useMemo, useState, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

import { Sidebar } from '../components/layout/Sidebar';
import { LanguageSwitch } from '../components/layout/LanguageSwitch';
import { SearchInput } from '../components/ui/SearchInput';
import { ClusterInfo } from '../types';
import {
  Loader2, ChevronUp, ChevronDown,
  ChevronLeft, ChevronRight, X, Layers, Calendar, Tag
} from 'lucide-react';
import { useSearchParams, Link } from 'react-router-dom';
import { useDataReposQuery } from '../features/data/hooks/useDataReposQuery';

// ============================================================================
// Types
// ============================================================================

type SortField = 'name' | 'language' | 'stargazers_count' | 'starred_at' | 'cluster' | 'summary' | 'last_commit_time';
type SortDirection = 'asc' | 'desc';

interface SortConfig {
  field: SortField;
  direction: SortDirection;
}

// ============================================================================
// Constants
// ============================================================================

const PAGE_SIZES = [25, 50, 100];
const DEFAULT_PAGE_SIZE = 25;

// ============================================================================
// Sub Components
// ============================================================================

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
  const justifyClass = align === 'center' ? 'justify-center' : align === 'right' ? 'justify-end' : 'justify-start';
  const ariaSort: React.AriaAttributes['aria-sort'] = isActive
    ? (currentSort.direction === 'asc' ? 'ascending' : 'descending')
    : 'none';

  return (
    <th
      className={`px-4 py-3 whitespace-nowrap ${className}`}
      scope="col"
      aria-sort={ariaSort}
    >
      <button
        type="button"
        onClick={() => onSort(field)}
        className={`w-full flex items-center gap-1 ${justifyClass} hover:bg-bg-hover transition-colors select-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30 rounded px-1 py-0.5 dark:hover:bg-dark-bg-sidebar/70`}
      >
        <span>{label}</span>
        <span className="flex flex-col">
          <ChevronUp
            className={`w-3 h-3 -mb-1 ${
              isActive && currentSort.direction === 'asc'
                ? 'text-action-primary'
                : 'text-text-dim'
            }`}
          />
          <ChevronDown
            className={`w-3 h-3 ${
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
  cluster: ClusterInfo | undefined;
  onClick?: () => void;
}

const ClusterBadge: React.FC<ClusterBadgeProps> = ({ cluster, onClick }) => {
  const { t } = useTranslation();
  if (!cluster) {
    return <span className="text-text-dim italic text-xs">{t('data.unclustered')}</span>;
  }

  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onClick?.();
      }}
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-medium hover:opacity-80 transition-opacity text-text-main dark:text-dark-text-main focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30"
      style={{
        backgroundColor: cluster.color + '30',
      }}
    >
      <div
        className="w-2 h-2 rounded-full"
        style={{ backgroundColor: cluster.color }}
      />
      {cluster.name || `Cluster ${cluster.id}`}
    </button>
  );
};

// ============================================================================
// Main Component
// ============================================================================

const DataPage = () => {
  const { t } = useTranslation();

  const [searchParams, setSearchParams] = useSearchParams();
  const monthFilter = searchParams.get('month');
  const topicFilter = searchParams.get('topic');

  // Local state
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    field: 'starred_at',
    direction: 'desc'
  });
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [localSearch, setLocalSearch] = useState('');
  const [selectedClusters, setSelectedClusters] = useState<Set<number>>(new Set());
  const offset = (currentPage - 1) * pageSize;

  const {
    repos,
    clusters,
    totalNodes,
    count,
    loading,
    error,
    retry,
  } = useDataReposQuery({
    searchQuery: localSearch,
    clusterIds: Array.from(selectedClusters),
    month: monthFilter,
    topic: topicFilter,
    sortField: sortConfig.field,
    sortDirection: sortConfig.direction,
    limit: pageSize,
    offset,
  });

  // Cluster map for quick lookup
  const clusterMap = useMemo(() => {
    const map = new Map<number, ClusterInfo>();
    clusters.forEach((c: ClusterInfo) => map.set(c.id, c));
    return map;
  }, [clusters]);

  const totalPages = Math.max(1, Math.ceil(count / pageSize));
  const paginatedData = repos;
  const hasActiveFilters = Boolean(
    selectedClusters.size > 0 || localSearch.trim() || monthFilter || topicFilter
  );

  useEffect(() => {
    if (currentPage > totalPages) {
      setCurrentPage(totalPages);
    }
  }, [currentPage, totalPages]);

  // Reset to page 1 when filters change
  const handleSearch = useCallback((query: string) => {
    setLocalSearch(query);
    setCurrentPage(1);
  }, []);

  // Handle sort
  const handleSort = useCallback((field: SortField) => {
    setSortConfig(prev => ({
      field,
      direction: prev.field === field && prev.direction === 'asc' ? 'desc' : 'asc',
    }));
    setCurrentPage(1);
  }, []);

  // Handle cluster filter
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



  // Format date
  const formatDate = (dateStr: string | null | undefined): string => {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
  };

  return (
    <div className="flex min-h-screen bg-bg-main text-text-main dark:bg-dark-bg-main dark:text-dark-text-main">
      <Sidebar />

      <main className="flex-1 flex flex-col min-w-0" style={{ marginLeft: 'var(--sidebar-width, 240px)' }}>
        {/* Header */}
        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between min-h-[3.5rem] px-4 sm:px-8 py-3 sm:py-0 border-b border-border-light sticky top-0 bg-bg-main/95 backdrop-blur-sm z-40 transition-all dark:bg-dark-bg-main/95 dark:border-dark-border">
          <div className="flex items-center gap-3 select-none">
            <h2 className="text-base font-semibold text-text-main tracking-tight">
              {t('sidebar.data')}
            </h2>
            <div className="h-4 w-[1px] bg-border-light mx-1" />
            <span className="text-sm text-text-muted">
              {count} / {totalNodes} {t('common.repositories')}
            </span>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center gap-3 w-full sm:w-auto">
            <LanguageSwitch />

            {/* Search */}
            <div className="w-full sm:w-64">
              <SearchInput
                onSearch={handleSearch}
                value={localSearch}
                placeholder={t('data.search_placeholder')}
              />
            </div>

            {/* Mobile sort */}
            <div className="flex items-center gap-2 sm:hidden">
              <label className="text-xs text-text-muted">{t('data.sort', 'Sort')}:</label>
              <select
                value={sortConfig.field}
                onChange={(e) => {
                  setSortConfig((prev) => ({
                    field: e.target.value as SortField,
                    direction: prev.direction,
                  }));
                  setCurrentPage(1);
                }}
                className="flex-1 border border-border-light rounded px-2 py-1 bg-bg-main text-text-main dark:bg-dark-bg-main dark:text-dark-text-main dark:border-dark-border"
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
                className="px-2 py-1 rounded border border-border-light bg-bg-main text-text-main focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30 dark:bg-dark-bg-main dark:text-dark-text-main dark:border-dark-border"
                aria-label={t('data.sort_direction', 'Toggle sort direction')}
              >
                {sortConfig.direction === 'asc' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
            </div>

            {/* Clear filters */}
            {hasActiveFilters && (
              <button
                onClick={() => {
                  setSelectedClusters(new Set());
                  setLocalSearch('');
                  setSearchParams({});
                  setCurrentPage(1);
                }}
                className="flex items-center gap-1 px-3 py-1.5 text-sm text-text-muted hover:text-text-main hover:bg-bg-hover rounded-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30"
              >
                <X className="w-4 h-4" />
                {t('common.clear_filters')}
              </button>
            )}
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 p-4 sm:p-6 overflow-auto">
          {loading ? (
            <div className="flex items-center justify-center h-64">
              <Loader2 className="animate-spin h-8 w-8 text-text-muted" />
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center gap-3 h-64">
              <p className="text-sm text-red-600">{t('common.load_failed', 'Failed to load data')}</p>
              <button
                type="button"
                onClick={() => {
                  void retry();
                }}
                className="px-3 py-1.5 rounded border border-border-light text-sm hover:bg-bg-hover dark:border-dark-border"
              >
                {t('common.retry')}
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Filter chips */}
              {(clusters.length > 0) || monthFilter || topicFilter ? (
                <div className="flex items-center gap-4 flex-wrap">
                  {/* Month Filter Chip */}
                  {monthFilter && (
                    <div className="flex items-center gap-2">
                       <div className="flex items-center gap-1 text-xs text-text-muted">
                        <Calendar className="w-4 h-4" />
                        <span>{t('data.filter_label')}</span>
                      </div>
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-action-primary/10 text-action-primary ring-1 ring-inset ring-action-primary/20">
                        {monthFilter}
                        <button
                          onClick={() => {
                            const nextParams = new URLSearchParams(searchParams);
                            nextParams.delete('month');
                            setSearchParams(nextParams);
                          }}
                          className="hover:bg-action-primary/20 rounded-full p-0.5"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    </div>
                  )}

                  {/* Topic Filter Chip */}
                  {topicFilter && (
                    <div className="flex items-center gap-2">
                      <div className="flex items-center gap-1 text-xs text-text-muted">
                        <Tag className="w-4 h-4" />
                        <span>{t('data.filter_label')}</span>
                      </div>
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-action-primary/10 text-action-primary ring-1 ring-inset ring-action-primary/20">
                        {t('data.topic', 'Topic')}: {topicFilter}
                        <button
                          onClick={() => {
                            const nextParams = new URLSearchParams(searchParams);
                            nextParams.delete('topic');
                            setSearchParams(nextParams);
                          }}
                          className="hover:bg-action-primary/20 rounded-full p-0.5"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    </div>
                  )}

                  {clusters.length > 0 && (
                    <div className="flex items-center gap-2 flex-wrap">
                      <div className="flex items-center gap-1 text-xs text-text-muted">
                        <Layers className="w-4 h-4" />
                        <span>{t('data.filter_by_cluster')}:</span>
                      </div>
                  {clusters.map((cluster: ClusterInfo) => (
                    <button
                      key={cluster.id}
                      onClick={() => handleClusterFilter(cluster.id)}
                        className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all text-text-main dark:text-dark-text-main ${
                        selectedClusters.has(cluster.id)
                          ? 'ring-2 ring-offset-1'
                          : 'hover:opacity-80'
                      } focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30`}
                      style={{
                        backgroundColor: cluster.color + '30',
                        '--tw-ring-color': cluster.color,
                      } as React.CSSProperties}
                    >
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: cluster.color }}
                      />
                      {cluster.name || `Cluster ${cluster.id}`}
                      <span className="opacity-75">({cluster.repo_count})</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : null}

              {/* Table (desktop) */}
              <div className="hidden sm:block w-full overflow-hidden rounded-lg border border-border-light bg-bg-main shadow-sm dark:bg-dark-bg-main dark:border-dark-border">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-bg-hover text-text-muted font-medium border-b border-border-light dark:bg-dark-bg-sidebar/60 dark:text-dark-text-main/70 dark:border-dark-border">
                      <tr>
                        <th className="px-2 py-3 w-14 text-center text-xs text-text-muted/50">#</th>
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
                        <th className="px-4 py-3 w-20 hidden">{t('data.description')}</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border-light dark:divide-dark-border">
                      {paginatedData.map((repo) => (
                        <tr
                          key={repo.id}
                          className="hover:bg-bg-hover/50 transition-colors dark:hover:bg-dark-bg-sidebar/60"
                        >
                          <td className="px-2 py-3 text-center">
                            {repo.owner_avatar_url ? (
                              <img
                                src={repo.owner_avatar_url}
                                alt={repo.owner}
                                className="w-6 h-6 min-w-6 min-h-6 mx-auto object-cover rounded-none"
                                loading="lazy"
                                decoding="async"
                                width={24}
                                height={24}
                              />
                            ) : (
                               <div className="w-6 h-6 min-w-6 min-h-6 bg-border-light mx-auto flex items-center justify-center text-[10px] text-text-dim rounded-none dark:bg-dark-border dark:text-dark-text-main/60">
                                 {repo.owner.charAt(0).toUpperCase()}
                               </div>
                            )}
                          </td>
                          <td className="px-4 py-3 max-w-xs">
                            <div className="flex items-center gap-2">
                              <Link
                                to={`/graph?node=${repo.id}`}
                                className="font-medium text-text-main hover:text-action-primary truncate block hover:underline"
                              >
                                {repo.full_name}
                              </Link>

                            </div>
                          </td>
                          <td className="px-4 py-3 max-w-md">
                            <p className="line-clamp-2 text-sm text-text-muted" title={repo.ai_summary || repo.description}>
                                {repo.ai_summary || repo.description || <span className="italic text-text-dim">{t('data.no_summary')}</span>}
                            </p>
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-center">
                            {repo.language ? (
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-bg-hover text-text-muted dark:bg-dark-bg-sidebar dark:text-dark-text-main/70">
                                {repo.language}
                              </span>
                            ) : (
                              <span className="text-text-muted italic">-</span>
                            )}
                          </td>
                          <td className="px-4 py-3 text-center font-mono text-text-dim tabular-nums">
                            {repo.stargazers_count.toLocaleString()}
                          </td>
                          <td className="px-4 py-3 text-center">
                            <ClusterBadge
                              cluster={repo.cluster_id != null ? clusterMap.get(repo.cluster_id) : undefined}
                              onClick={() => repo.cluster_id != null && handleClusterFilter(repo.cluster_id)}
                            />
                          </td>
                          <td className="px-4 py-3 text-text-muted text-xs whitespace-nowrap text-center">
                            {formatDate(repo.starred_at)}
                          </td>
                          <td className="px-4 py-3 text-text-muted text-xs whitespace-nowrap text-center">
                            {formatDate(repo.last_commit_time)}
                          </td>
                          <td className="px-4 py-3 max-w-md hidden">
                            <p className="truncate text-text-muted text-xs">
                              {repo.description || <span className="italic text-text-dim">{t('data.no_description')}</span>}
                            </p>
                          </td>
                        </tr>
                      ))}

                      {paginatedData.length === 0 && (
                        <tr>
                          <td colSpan={8} className="px-4 py-12 text-center text-text-muted">
                            {hasActiveFilters
                              ? t('data.no_results')
                              : t('data.no_data')
                            }
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Cards (mobile) */}
              <div className="sm:hidden space-y-3">
                {paginatedData.map((repo) => (
                  <div key={repo.id} className="rounded-lg border border-border-light bg-bg-main p-4 shadow-sm dark:bg-dark-bg-main dark:border-dark-border">
                    <div className="flex items-start gap-3">
                      {repo.owner_avatar_url ? (
                        <img
                          src={repo.owner_avatar_url}
                          alt={repo.owner}
                          className="w-10 h-10 rounded-md border border-border-light object-cover"
                          loading="lazy"
                          decoding="async"
                          width={40}
                          height={40}
                        />
                      ) : (
                        <div className="w-10 h-10 rounded-md bg-border-light flex items-center justify-center text-text-dim dark:bg-dark-border dark:text-dark-text-main/60">
                          {repo.owner.charAt(0).toUpperCase()}
                        </div>
                      )}
                      <div className="flex-1 min-w-0">
                        <Link
                          to={`/graph?node=${repo.id}`}
                          className="font-semibold text-text-main hover:text-action-primary block truncate"
                        >
                          {repo.full_name}
                        </Link>
                        <p className="mt-1 text-xs text-text-muted line-clamp-2 dark:text-dark-text-main/70">
                          {repo.ai_summary || repo.description || t('data.no_summary')}
                        </p>
                      </div>
                      <div className="text-xs text-text-muted tabular-nums whitespace-nowrap dark:text-dark-text-main/70">
                        ⭐ {repo.stargazers_count.toLocaleString()}
                      </div>
                    </div>

                    <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-text-muted dark:text-dark-text-main/70">
                      {repo.language && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded bg-bg-hover text-text-muted dark:bg-dark-bg-sidebar dark:text-dark-text-main/70">
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

                {paginatedData.length === 0 && (
                  <div className="rounded-lg border border-border-light bg-bg-main p-6 text-center text-sm text-text-muted dark:bg-dark-bg-main dark:border-dark-border">
                    {hasActiveFilters
                      ? t('data.no_results')
                      : t('data.no_data')
                    }
                  </div>
                )}
              </div>

              {/* Pagination */}
              {count > 0 && (
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2 text-text-muted">
                    <span>{t('data.rows_per_page')}:</span>
                    <select
                      value={pageSize}
                      onChange={(e) => {
                        setPageSize(Number(e.target.value));
                        setCurrentPage(1);
                      }}
                      className="border border-border-light rounded px-2 py-1 bg-bg-main text-text-main dark:bg-dark-bg-main dark:text-dark-text-main dark:border-dark-border"
                    >
                      {PAGE_SIZES.map(size => (
                        <option key={size} value={size}>{size}</option>
                      ))}
                    </select>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="text-text-muted">
                      {t('data.showing', {
                        start: offset + 1,
                        end: offset + paginatedData.length,
                        total: count,
                      })}
                    </span>

                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                        disabled={currentPage === 1}
                        className="p-1.5 rounded hover:bg-bg-hover disabled:opacity-30 disabled:cursor-not-allowed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30"
                      >
                        <ChevronLeft className="w-4 h-4" />
                      </button>

                      <span className="px-3 py-1 text-text-main font-medium">
                        {currentPage} / {totalPages || 1}
                      </span>

                      <button
                        onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                        disabled={currentPage === totalPages || totalPages === 0}
                        className="p-1.5 rounded hover:bg-bg-hover disabled:opacity-30 disabled:cursor-not-allowed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30"
                      >
                        <ChevronRight className="w-4 h-4" />
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
