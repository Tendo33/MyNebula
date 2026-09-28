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
import { NativeSelect } from '../components/ui/native-select';
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '../components/ui/resizable';
import { ScrollArea } from '../components/ui/scroll-area';
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
    clampPage(totalPages);
  }, [clampPage, totalPages]);

  return (
    <div className="relative flex min-h-0 flex-1">
      <ResizablePanelGroup orientation="horizontal" className="absolute inset-0">
        <ResizablePanel defaultSize="22%" minSize="16%" maxSize="36%">
          <aside
            aria-label={t('common.filter')}
            className="flex h-full min-h-0 flex-col border-r bg-sidebar"
          >
            <div className="shrink-0 border-b px-3 py-3">
              <p className="text-xs font-medium text-muted-foreground">{t('common.filter')}</p>
            </div>
            <ScrollArea className="min-h-0 flex-1">
              <div className="flex flex-col gap-4 p-3">
                {monthFilter && (
                  <div className="flex flex-col items-start gap-2">
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
                        aria-label={t('common.clear_filter')}
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </span>
                  </div>
                )}

                {topicFilter && (
                  <div className="flex flex-col items-start gap-2">
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
                        aria-label={t('common.clear_filter')}
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </span>
                  </div>
                )}

                {clusters.length > 0 && (
                  <div className="flex flex-col items-start gap-2">
                    <div className="flex items-center gap-1 text-xs text-text-muted">
                      <Layers className="h-4 w-4" />
                      <span>{t('data.filter_by_cluster')}:</span>
                    </div>
                    {clusters.map((cluster) => {
                      const accent = getClusterAccent({ id: cluster.id, color: cluster.color });
                      const selected = selectedClusters.has(cluster.id);

                      return (
                        <button
                          key={cluster.id}
                          type="button"
                          onClick={() => handleClusterFilter(cluster.id)}
                          className={`chip-button max-w-full ${selected ? 'ring-2 ring-offset-1 shadow-sm' : 'hover:opacity-90'}`}
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
                          <div
                            className="h-2 w-2 shrink-0 rounded-full"
                            style={{ backgroundColor: accent.dot }}
                          />
                          {cluster.name || `Cluster ${cluster.id}`}
                          <span className="opacity-75">({cluster.repo_count})</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            </ScrollArea>
          </aside>
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel defaultSize="78%" minSize="40%">
          <div className="flex h-full min-h-0 flex-col">
            <div className="flex shrink-0 flex-col gap-3 border-b px-4 py-3">
              <div className="flex items-center justify-between gap-3">
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
                  <div className="flex items-center gap-2 sm:hidden">
                    <label htmlFor="data-mobile-sort" className="text-sm text-text-muted">{t('data.sort', 'Sort')}:</label>
                    <NativeSelect
                      id="data-mobile-sort"
                      value={sortConfig.field}
                      onChange={(event) => {
                        setSortConfig((prev) => ({
                          field: event.target.value as SortField,
                          direction: prev.direction,
                        }));
                        setCurrentPage(1);
                      }}
                      className="flex-1"
                    >
                      <option value="starred_at">{t('data.starred_date')}</option>
                      <option value="name">{t('data.repository')}</option>
                      <option value="stargazers_count">{t('data.stars')}</option>
                      <option value="language">{t('data.language')}</option>
                      <option value="cluster">{t('data.cluster')}</option>
                      <option value="summary">{t('data.summary')}</option>
                      <option value="last_commit_time">{t('data.last_commit')}</option>
                    </NativeSelect>
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

                  <div className="flex items-center gap-2 text-sm text-text-muted">
                    <label htmlFor="data-page-size">{t('data.rows_per_page')}:</label>
                    <NativeSelect
                      id="data-page-size"
                      value={pageSize}
                      onChange={(event) => {
                        setPageSize(Number(event.target.value));
                        setCurrentPage(1);
                      }}
                    >
                      {PAGE_SIZES.map((size) => (
                        <option key={size} value={size}>
                          {size}
                        </option>
                      ))}
                    </NativeSelect>
                  </div>
                </div>
              </div>
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
              <div className="flex shrink-0 items-center justify-end gap-2 border-t px-4 py-3 text-sm">
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
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
};

export default DataPage;
