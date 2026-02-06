import { useMemo, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { Sidebar } from '../components/layout/Sidebar';
import { SearchInput } from '../components/ui/SearchInput';
import { useGraph } from '../contexts/GraphContext';
import { ClusterInfo } from '../types';
import {
  Loader2, ChevronUp, ChevronDown,
  ChevronLeft, ChevronRight, X, Layers, Calendar
} from 'lucide-react';
import { useSearchParams, Link } from 'react-router-dom';

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

  return (
    <th
      className={`px-4 py-3 whitespace-nowrap cursor-pointer hover:bg-gray-100 transition-colors select-none ${className}`}
      onClick={() => onSort(field)}
    >
      <div className={`flex items-center gap-1 ${justifyClass}`}>
        <span>{label}</span>
        <span className="flex flex-col">
          <ChevronUp
            className={`w-3 h-3 -mb-1 ${
              isActive && currentSort.direction === 'asc'
                ? 'text-action-primary'
                : 'text-gray-300'
            }`}
          />
          <ChevronDown
            className={`w-3 h-3 ${
              isActive && currentSort.direction === 'desc'
                ? 'text-action-primary'
                : 'text-gray-300'
            }`}
          />
        </span>
      </div>
    </th>
  );
};

interface ClusterBadgeProps {
  cluster: ClusterInfo | undefined;
  onClick?: () => void;
}

const ClusterBadge: React.FC<ClusterBadgeProps> = ({ cluster, onClick }) => {
  if (!cluster) {
    return <span className="text-text-dim italic text-xs">Unclustered</span>;
  }

  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onClick?.();
      }}
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-medium hover:opacity-80 transition-opacity"
      style={{
        backgroundColor: cluster.color + '30',
        color: '#374151',
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

  const { rawData, filters, toggleCluster, clearClusterFilter, loading } = useGraph();
  const [searchParams, setSearchParams] = useSearchParams();
  const monthFilter = searchParams.get('month');

  // Local state
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    field: 'starred_at',
    direction: 'desc'
  });
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [localSearch, setLocalSearch] = useState('');

  // Cluster map for quick lookup
  const clusterMap = useMemo(() => {
    if (!rawData) return new Map<number, ClusterInfo>();

    const map = new Map<number, ClusterInfo>();
    rawData.clusters.forEach(c => map.set(c.id, c));
    return map;
  }, [rawData]);

  // Filter and sort data
  const processedData = useMemo(() => {
    if (!rawData) return [];

    let filtered = [...rawData.nodes];

    // Apply search filter
    if (localSearch.trim()) {
      const query = localSearch.toLowerCase();
      filtered = filtered.filter(
        node =>
          node.name.toLowerCase().includes(query) ||
          node.full_name.toLowerCase().includes(query) ||
          (node.description?.toLowerCase().includes(query) ?? false) ||
          (node.language?.toLowerCase().includes(query) ?? false)
      );
    }

    // Apply cluster filter
    if (filters.selectedClusters.size > 0) {
      filtered = filtered.filter(
        node => node.cluster_id !== undefined && filters.selectedClusters.has(node.cluster_id)
      );
    }

    // Apply month filter
    if (monthFilter) {
      filtered = filtered.filter(node =>
        node.starred_at && node.starred_at.startsWith(monthFilter)
      );
    }

    // Sort
    filtered.sort((a, b) => {
      let comparison = 0;

      switch (sortConfig.field) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'language':
          comparison = (a.language || '').localeCompare(b.language || '');
          break;
        case 'stargazers_count':
          comparison = a.stargazers_count - b.stargazers_count;
          break;
        case 'starred_at':
          comparison = (a.starred_at || '').localeCompare(b.starred_at || '');
          break;
        case 'cluster':
          const clusterA = a.cluster_id !== undefined ? clusterMap.get(a.cluster_id)?.name || '' : '';
          const clusterB = b.cluster_id !== undefined ? clusterMap.get(b.cluster_id)?.name || '' : '';
          comparison = clusterA.localeCompare(clusterB);
          break;
        case 'summary':
          comparison = (a.ai_summary || '').localeCompare(b.ai_summary || '');
          break;
        case 'last_commit_time':
          comparison = (a.last_commit_time || '').localeCompare(b.last_commit_time || '');
          break;
      }

      return sortConfig.direction === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [rawData, localSearch, filters.selectedClusters, sortConfig, clusterMap]);

  // Pagination
  const totalPages = Math.ceil(processedData.length / pageSize);
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return processedData.slice(start, start + pageSize);
  }, [processedData, currentPage, pageSize]);

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
    toggleCluster(clusterId);
    setCurrentPage(1);
  }, [toggleCluster]);



  // Format date
  const formatDate = (dateStr: string | undefined): string => {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
  };

  return (
    <div className="flex min-h-screen bg-bg-main text-text-main">
      <Sidebar />

      <main className="flex-1 ml-60 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between h-14 px-8 border-b border-border-light sticky top-0 bg-bg-main/95 backdrop-blur-sm z-40 transition-all">
          <div className="flex items-center gap-3 select-none">
            <h2 className="text-base font-semibold text-text-main tracking-tight">
              {t('sidebar.data')}
            </h2>
            <div className="h-4 w-[1px] bg-border-light mx-1" />
            <span className="text-sm text-text-muted">
              {processedData.length} / {rawData?.total_nodes || 0} {t('common.repositories')}
            </span>
          </div>

          <div className="flex items-center gap-3">
            {/* Search */}
            <div className="w-64">
              <SearchInput
                onSearch={handleSearch}
                value={localSearch}
                placeholder={t('data.search_placeholder')}
              />
            </div>

            {/* Clear filters */}
            {(filters.selectedClusters.size > 0 || localSearch.trim() || monthFilter) && (
              <button
                onClick={() => {
                  clearClusterFilter();
                  setLocalSearch('');
                  setSearchParams({});
                  setCurrentPage(1);
                }}
                className="flex items-center gap-1 px-3 py-1.5 text-sm text-text-muted hover:text-text-main hover:bg-bg-hover rounded-md transition-colors"
              >
                <X className="w-4 h-4" />
                {t('common.clear_filters')}
              </button>
            )}
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 p-6 overflow-auto">
          {loading ? (
            <div className="flex items-center justify-center h-64">
              <Loader2 className="animate-spin h-8 w-8 text-text-muted" />
            </div>
          ) : (
            <div className="space-y-4">
              {/* Filter chips */}
              {(rawData && rawData.clusters.length > 0) || monthFilter ? (
                <div className="flex items-center gap-4 flex-wrap">
                  {/* Month Filter Chip */}
                  {monthFilter && (
                    <div className="flex items-center gap-2">
                       <div className="flex items-center gap-1 text-xs text-text-muted">
                        <Calendar className="w-4 h-4" />
                        <span>Filter:</span>
                      </div>
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-action-primary/10 text-action-primary ring-1 ring-inset ring-action-primary/20">
                        {monthFilter}
                        <button
                          onClick={() => {
                            searchParams.delete('month');
                            setSearchParams(searchParams);
                          }}
                          className="hover:bg-action-primary/20 rounded-full p-0.5"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    </div>
                  )}

                  {rawData && rawData.clusters.length > 0 && (
                    <div className="flex items-center gap-2 flex-wrap">
                      <div className="flex items-center gap-1 text-xs text-text-muted">
                        <Layers className="w-4 h-4" />
                        <span>{t('data.filter_by_cluster')}:</span>
                      </div>
                  {rawData.clusters.map(cluster => (
                    <button
                      key={cluster.id}
                      onClick={() => handleClusterFilter(cluster.id)}
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                        filters.selectedClusters.has(cluster.id)
                          ? 'ring-2 ring-offset-1'
                          : 'hover:opacity-80'
                      }`}
                      style={{
                        backgroundColor: cluster.color + '30',
                        color: '#374151',
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

              {/* Table */}
              <div className="w-full overflow-hidden rounded-lg border border-border-light bg-white shadow-sm">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-bg-hover text-text-muted font-medium border-b border-border-light">
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
                    <tbody className="divide-y divide-border-light">
                      {paginatedData.map((repo) => (
                        <tr
                          key={repo.id}
                          className="hover:bg-bg-hover/50 transition-colors"
                        >
                          <td className="px-2 py-3 text-center">
                            {repo.owner_avatar_url ? (
                              <img src={repo.owner_avatar_url} alt={repo.owner} className="w-6 h-6 min-w-6 min-h-6 mx-auto object-cover rounded-none" />
                            ) : (
                               <div className="w-6 h-6 min-w-6 min-h-6 bg-gray-200 mx-auto flex items-center justify-center text-[10px] text-gray-500 rounded-none">
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
                                {repo.ai_summary || repo.description || <span className="italic text-text-dim">No summary</span>}
                            </p>
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-center">
                            {repo.language ? (
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800">
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
                              cluster={repo.cluster_id !== undefined ? clusterMap.get(repo.cluster_id) : undefined}
                              onClick={() => repo.cluster_id !== undefined && handleClusterFilter(repo.cluster_id)}
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
                              {repo.description || <span className="italic text-text-dim">No description</span>}
                            </p>
                          </td>
                        </tr>
                      ))}

                      {paginatedData.length === 0 && (
                        <tr>
                          <td colSpan={7} className="px-4 py-12 text-center text-text-muted">
                            {localSearch || filters.selectedClusters.size > 0
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

              {/* Pagination */}
              {processedData.length > 0 && (
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2 text-text-muted">
                    <span>{t('data.rows_per_page')}:</span>
                    <select
                      value={pageSize}
                      onChange={(e) => {
                        setPageSize(Number(e.target.value));
                        setCurrentPage(1);
                      }}
                      className="border border-border-light rounded px-2 py-1 bg-white text-text-main"
                    >
                      {PAGE_SIZES.map(size => (
                        <option key={size} value={size}>{size}</option>
                      ))}
                    </select>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="text-text-muted">
                      {t('data.showing', {
                        start: (currentPage - 1) * pageSize + 1,
                        end: Math.min(currentPage * pageSize, processedData.length),
                        total: processedData.length,
                      })}
                    </span>

                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                        disabled={currentPage === 1}
                        className="p-1.5 rounded hover:bg-bg-hover disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                      >
                        <ChevronLeft className="w-4 h-4" />
                      </button>

                      <span className="px-3 py-1 text-text-main font-medium">
                        {currentPage} / {totalPages || 1}
                      </span>

                      <button
                        onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                        disabled={currentPage === totalPages || totalPages === 0}
                        className="p-1.5 rounded hover:bg-bg-hover disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
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
