import { useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Calendar,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Layers,
  Tag,
  X,
} from 'lucide-react';

import { SearchInput } from '../components/ui/SearchInput';
import { EmptyState } from '../components/ui/EmptyState';
import type { DataClusterInfo } from '../api/v2/data';
import { useDataReposQuery } from '../features/data/hooks/useDataReposQuery';
import { useDataPageUrlState } from './data/hooks/useDataPageUrlState';
import { getClusterAccent } from '../utils/clusterAccent';
import { PAGE_SIZES, type SortField } from './data/dataPageFilters';
import { DataRepoTable } from './data/DataRepoTable';
import { Button } from '../components/ui/button';
import { SelectField } from '../components/ui/select';
import { Spinner } from '../components/ui/spinner';

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
    if (!loading && !error) {
      clampPage(totalPages);
    }
  }, [clampPage, error, loading, totalPages]);

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex shrink-0 flex-col gap-3 border-b px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            {!loading && !error && totalNodes > 0 ? (
              <span className="page-subtitle">
                {t('data.shown_of_total', { shown: count, total: totalNodes })}
              </span>
            ) : null}
          </div>
          {hasActiveFilters && (
            <Button type="button" variant="outline" size="sm" onClick={clearFilters}>
              <X data-icon="inline-start" />
              {t('common.clear_filters')}
            </Button>
          )}
        </div>

        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <div className="min-w-0 w-full sm:max-w-sm sm:flex-1">
            <SearchInput
              onSearch={handleSearch}
              value={localSearch}
              placeholder={t('data.search_placeholder')}
            />
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="flex min-w-0 flex-1 items-center gap-2 sm:hidden">
              <label htmlFor="data-mobile-sort" className="shrink-0 text-sm text-text-muted">{t('data.sort', 'Sort')}:</label>
              <SelectField
                id="data-mobile-sort"
                value={sortConfig.field}
                onValueChange={(value) => {
                  setSortConfig((prev) => ({
                    field: value as SortField,
                    direction: prev.direction,
                  }));
                  setCurrentPage(1);
                }}
                className="min-w-0 flex-1"
                options={[
                  { value: 'starred_at', label: t('data.starred_date') },
                  { value: 'name', label: t('data.repository') },
                  { value: 'stargazers_count', label: t('data.stars') },
                  { value: 'language', label: t('data.language') },
                  { value: 'cluster', label: t('data.cluster') },
                  { value: 'summary', label: t('data.summary') },
                  { value: 'last_commit_time', label: t('data.last_commit') },
                ]}
              />
              <Button
                type="button"
                variant="outline"
                size="icon"
                onClick={() => {
                  setSortConfig((prev) => ({
                    field: prev.field,
                    direction: prev.direction === 'asc' ? 'desc' : 'asc',
                  }));
                  setCurrentPage(1);
                }}
                aria-label={t('data.sort_direction', 'Toggle sort direction')}
              >
                {sortConfig.direction === 'asc' ? <ChevronUp /> : <ChevronDown />}
              </Button>
            </div>

            <div className="flex shrink-0 items-center gap-2 text-sm text-text-muted">
              <label htmlFor="data-page-size">{t('data.rows_per_page')}:</label>
              <SelectField
                id="data-page-size"
                value={String(pageSize)}
                onValueChange={(value) => {
                  setPageSize(Number(value));
                  setCurrentPage(1);
                }}
                options={PAGE_SIZES.map((size) => ({ value: String(size), label: String(size) }))}
              />
            </div>
          </div>
        </div>

        {(clusters.length > 0 || monthFilter || topicFilter) && (
          <section aria-label={t('common.filter')} className="min-w-0">
            <div className="flex max-h-28 flex-wrap items-center gap-2 overflow-y-auto p-1">
              {monthFilter && (
                <span className="chip-button max-w-full border-action-primary/10 bg-action-primary/10 text-action-primary">
                  <Calendar className="size-3.5 shrink-0" />
                  <span className="truncate">{monthFilter}</span>
                  <button
                    type="button"
                    onClick={() => {
                      const nextParams = new URLSearchParams(searchParams);
                      nextParams.delete('month');
                      setSearchParams(nextParams);
                    }}
                    className="shrink-0 rounded p-0.5 hover:bg-action-primary/20"
                    aria-label={t('common.clear_filter')}
                  >
                    <X className="size-3" />
                  </button>
                </span>
              )}
              {topicFilter && (
                <span className="chip-button max-w-full border-action-primary/10 bg-action-primary/10 text-action-primary">
                  <Tag className="size-3.5 shrink-0" />
                  <span className="truncate" title={topicFilter}>{t('data.topic', 'Topic')}: {topicFilter}</span>
                  <button
                    type="button"
                    onClick={() => {
                      const nextParams = new URLSearchParams(searchParams);
                      nextParams.delete('topic');
                      setSearchParams(nextParams);
                    }}
                    className="shrink-0 rounded p-0.5 hover:bg-action-primary/20"
                    aria-label={t('common.clear_filter')}
                  >
                    <X className="size-3" />
                  </button>
                </span>
              )}
              {clusters.length > 0 && (
                <>
                  <span className="flex shrink-0 items-center gap-1 text-xs text-text-muted">
                    <Layers className="size-3.5" />
                    {t('data.filter_by_cluster')}:
                  </span>
                  {clusters.map((cluster) => {
                    const accent = getClusterAccent({ id: cluster.id, color: cluster.color });
                    const selected = selectedClusters.has(cluster.id);
                    const name = cluster.name || `Cluster ${cluster.id}`;

                    return (
                      <button
                        key={cluster.id}
                        type="button"
                        onClick={() => handleClusterFilter(cluster.id)}
                        aria-pressed={selected}
                        title={name}
                        className={`chip-button max-w-full sm:max-w-xs ${selected ? 'ring-2 ring-offset-1' : 'hover:opacity-90'}`}
                        style={
                          {
                            backgroundColor: selected ? accent.strongBackground : accent.softBackground,
                            borderColor: selected ? accent.strongBorder : accent.softBorder,
                            ['--chip-text' as string]: accent.text,
                            ['--chip-text-dark' as string]: accent.textOnDark,
                            '--tw-ring-color': accent.base,
                          } as React.CSSProperties
                        }
                      >
                        <span className="size-2 shrink-0 rounded-full" style={{ backgroundColor: accent.dot }} />
                        <span className="truncate">{name}</span>
                        <span className="shrink-0 opacity-75">({cluster.repo_count})</span>
                      </button>
                    );
                  })}
                </>
              )}
            </div>
          </section>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-hidden">
        {loading ? (
          <div className="flex h-full items-center justify-center">
            <Spinner className="size-8 text-text-muted" />
          </div>
        ) : error ? (
          <div className="flex h-full items-center justify-center px-4">
            <EmptyState
              title={t('common.load_failed_data')}
              actionType="button"
              actionLabel={t('common.retry')}
              onAction={() => {
                void retry();
              }}
            />
          </div>
        ) : (
          <DataRepoTable
            repos={repos}
            clusterMap={clusterMap}
            sortConfig={sortConfig}
            onSort={handleSort}
            onClusterFilter={handleClusterFilter}
            hasActiveFilters={hasActiveFilters}
            onClearFilters={clearFilters}
          />
        )}
      </div>

      {!loading && !error && count > 0 && (
        <div className="flex shrink-0 flex-wrap items-center justify-end gap-2 border-t px-4 py-3 text-sm">
          <span className="text-text-muted">
            {t('data.showing', {
              start: offset + 1,
              end: offset + repos.length,
              total: count,
            })}
          </span>
          <div className="flex items-center gap-1">
            <Button
              type="button"
              variant="outline"
              size="icon"
              onClick={() => setCurrentPage((page) => Math.max(1, page - 1))}
              disabled={currentPage === 1}
              aria-label={t('data.previous_page', 'Previous page')}
            >
              <ChevronLeft />
            </Button>
            <span className="px-3 py-1 font-medium text-text-main">
              {currentPage} / {totalPages || 1}
            </span>
            <Button
              type="button"
              variant="outline"
              size="icon"
              onClick={() => setCurrentPage((page) => Math.min(totalPages, page + 1))}
              disabled={currentPage === totalPages || totalPages === 0}
              aria-label={t('data.next_page', 'Next page')}
            >
              <ChevronRight />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataPage;
