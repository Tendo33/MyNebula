import { useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Calendar,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Layers,
  Loader2,
  Tag,
  X,
} from 'lucide-react';

import { Sidebar } from '../components/layout/Sidebar';
import { LanguageSwitch } from '../components/layout/LanguageSwitch';
import { SearchInput } from '../components/ui/SearchInput';
import type { DataClusterInfo } from '../api/v2/data';
import { useDataReposQuery } from '../features/data/hooks/useDataReposQuery';
import { useDataPageUrlState } from './data/hooks/useDataPageUrlState';
import { getClusterAccent } from '../utils/clusterAccent';
import { PAGE_SIZES, type SortField } from './data/dataPageFilters';
import { DataRepoTable } from './data/DataRepoTable';

const DataPage = () => {
  const { t } = useTranslation();
  const {
    monthFilter,
    topicFilter,
    sortConfig,
    setSortConfig,
    currentPage,
    setCurrentPage,
    pageSize,
    setPageSize,
    localSearch,
    selectedClusters,
    handleSearch,
    handleSort,
    handleClusterFilter,
    clampPage,
    clearFilters,
    searchParams,
    setSearchParams,
  } = useDataPageUrlState();

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
    clampPage(totalPages);
  }, [clampPage, totalPages]);

  return (
    <div className="page-shell">
      <Sidebar />

      <main id="main-content" className="page-main">
        <header className="page-header">
          <div className="page-header-inner select-none">
            <div>
              <div className="section-kicker mb-1 px-0">{t('common.repositories')}</div>
              <h1 className="page-title">{t('sidebar.data')}</h1>
            </div>
            <span className="toolbar-badge">
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
              <label htmlFor="data-mobile-sort" className="text-xs text-text-muted">{t('data.sort', 'Sort')}:</label>
              <select
                id="data-mobile-sort"
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
                onClick={clearFilters}
                className="header-action-ghost self-start sm:self-auto"
              >
                <X className="h-4 w-4" />
                {t('common.clear_filters')}
              </button>
            )}
          </div>
        </header>

        <div className="page-content">
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

              <DataRepoTable
                repos={repos}
                clusterMap={clusterMap}
                sortConfig={sortConfig}
                onSort={handleSort}
                onClusterFilter={handleClusterFilter}
                hasActiveFilters={hasActiveFilters}
              />

              {count > 0 && (
                <div className="flex flex-col gap-3 text-sm sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-center gap-2 text-text-muted">
                    <label htmlFor="data-page-size">{t('data.rows_per_page')}:</label>
                    <select
                      id="data-page-size"
                      value={pageSize}
                      onChange={(event) => {
                        setPageSize(Number(event.target.value));
                        setCurrentPage(1);
                      }}
                      className="field-surface h-11 px-3 text-sm"
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
                        className="header-action h-11 w-11 px-0 disabled:cursor-not-allowed disabled:opacity-30"
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
                        className="header-action h-11 w-11 px-0 disabled:cursor-not-allowed disabled:opacity-30"
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
