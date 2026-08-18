import { useCallback, useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';

import {
  DEFAULT_PAGE_SIZE,
  buildFilterParams,
  nextSortConfig,
  parseClusterParams,
  toggleClusterSelection,
  type SortConfig,
  type SortField,
} from '../dataPageFilters';

/**
 * URL-backed filter, sort, and pagination state for the Data page.
 *
 * The URL is the source of truth for query and cluster selection so a link can
 * reproduce a view. Sort and page size stay local — they are presentation
 * preferences, not addressable state.
 */
export const useDataPageUrlState = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const monthFilter = searchParams.get('month');
  const topicFilter = searchParams.get('topic');
  const urlSearchQuery = searchParams.get('q') ?? '';

  const [sortConfig, setSortConfig] = useState<SortConfig>({
    field: 'starred_at',
    direction: 'desc',
  });
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [localSearch, setLocalSearch] = useState(urlSearchQuery);
  const [selectedClusters, setSelectedClusters] = useState<Set<number>>(
    () => new Set(parseClusterParams(searchParams))
  );

  // Adopt back-button and external URL changes into local state.
  useEffect(() => {
    const nextClusterIds = parseClusterParams(searchParams);
    const currentClusterIds = Array.from(selectedClusters).sort((left, right) => left - right);
    if (localSearch !== urlSearchQuery) {
      setLocalSearch(urlSearchQuery);
    }
    if (currentClusterIds.join(',') !== nextClusterIds.join(',')) {
      setSelectedClusters(new Set(nextClusterIds));
    }
  }, [localSearch, searchParams, selectedClusters, urlSearchQuery]);

  // Clamp the page when the result count shrinks under an active page. The
  // caller owns this because `totalPages` derives from the query that this
  // hook's own state drives.
  const clampPage = useCallback((totalPages: number) => {
    setCurrentPage((page) => (page > totalPages ? totalPages : page));
  }, []);

  const handleSearch = useCallback(
    (query: string) => {
      setLocalSearch(query);
      setCurrentPage(1);
      setSearchParams(buildFilterParams(searchParams, query, selectedClusters), {
        replace: true,
      });
    },
    [searchParams, selectedClusters, setSearchParams]
  );

  const handleSort = useCallback((field: SortField) => {
    setSortConfig((prev) => nextSortConfig(prev, field));
    setCurrentPage(1);
  }, []);

  const handleClusterFilter = useCallback(
    (clusterId: number) => {
      setSelectedClusters((current) => {
        const next = toggleClusterSelection(current, clusterId);
        setSearchParams(buildFilterParams(searchParams, localSearch, next), { replace: true });
        return next;
      });
      setCurrentPage(1);
    },
    [searchParams, localSearch, setSearchParams]
  );

  /** Reset every filter and return to an unfiltered first page. */
  const clearFilters = useCallback(() => {
    setSelectedClusters(new Set());
    setLocalSearch('');
    setSearchParams({});
    setCurrentPage(1);
  }, [setSearchParams]);

  return {
    searchParams,
    setSearchParams,
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
  };
};
