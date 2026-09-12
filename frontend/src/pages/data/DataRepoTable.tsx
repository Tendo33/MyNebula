import React from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import { Star } from 'lucide-react';

import type { DataClusterInfo, DataRepoItem } from '../../api/v2/data';
import { Card } from '../../components/ui/card';
import { EmptyState } from '../../components/ui/EmptyState';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../../components/ui/table';
import { ClusterBadge, SortableHeader } from './DataTableParts';
import { formatDataDate, type SortConfig, type SortField } from './dataPageFilters';

/**
 * Repository table for the Data page.
 *
 * Renders a sortable table above the `sm` breakpoint and a card list below it;
 * both read from the same data so the two layouts cannot drift apart.
 */
export const DataRepoTable: React.FC<{
  repos: DataRepoItem[];
  clusterMap: Map<number, DataClusterInfo>;
  sortConfig: SortConfig;
  onSort: (field: SortField) => void;
  onClusterFilter: (clusterId: number) => void;
  hasActiveFilters: boolean;
  onClearFilters: () => void;
}> = ({ repos, clusterMap, sortConfig, onSort, onClusterFilter, hasActiveFilters, onClearFilters }) => {
  const { t } = useTranslation();

  return (
    <>
    <Card className="hidden w-full overflow-hidden py-0 sm:block">
        <Table className="text-left">
          <TableHeader className="bg-muted font-medium text-muted-foreground">
            <TableRow>
              <TableHead className="w-14 px-2 py-3 text-center text-xs text-muted-foreground/50">#</TableHead>
              <SortableHeader
                label={t('data.repository')}
                field="name"
                currentSort={sortConfig}
                onSort={onSort}
              />
              <SortableHeader
                label={t('data.summary')}
                field="summary"
                currentSort={sortConfig}
                onSort={onSort}
              />
              <SortableHeader
                label={t('data.language')}
                field="language"
                currentSort={sortConfig}
                onSort={onSort}
                align="center"
              />
              <SortableHeader
                label={t('data.stars')}
                field="stargazers_count"
                currentSort={sortConfig}
                onSort={onSort}
                align="center"
              />
              <SortableHeader
                label={t('data.cluster')}
                field="cluster"
                currentSort={sortConfig}
                onSort={onSort}
                align="center"
              />
              <SortableHeader
                label={t('data.starred_date')}
                field="starred_at"
                currentSort={sortConfig}
                onSort={onSort}
                align="center"
              />
              <SortableHeader
                label={t('data.last_commit')}
                field="last_commit_time"
                currentSort={sortConfig}
                onSort={onSort}
                align="center"
              />
              <TableHead className="hidden w-20 px-4 py-3">{t('data.description')}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {repos.map((repo) => (
              <TableRow key={repo.id}>
                <TableCell className="px-2 py-3 text-center">
                  {repo.owner_avatar_url ? (
                    <img
                      src={repo.owner_avatar_url}
                      alt={repo.owner}
                      className="mx-auto h-7 min-h-7 w-7 min-w-7 rounded-lg object-cover"
                      loading="lazy"
                      decoding="async"
                      width={24}
                      height={24}
                    />
                  ) : (
                    <div className="mx-auto flex h-7 min-h-7 w-7 min-w-7 items-center justify-center rounded-lg bg-border-light text-[10px] text-text-muted dark:bg-dark-border dark:text-dark-text-main/60">
                      {repo.owner.charAt(0).toUpperCase()}
                    </div>
                  )}
                </TableCell>
                <TableCell className="max-w-xs whitespace-normal px-4 py-3">
                  <div className="flex items-center gap-2">
                    <Link
                      to={`/graph?node=${repo.id}`}
                      className="block min-h-[1.75rem] truncate py-1 font-medium text-text-main hover:text-action-primary hover:underline"
                    >
                      {repo.full_name}
                    </Link>
                  </div>
                </TableCell>
                <TableCell className="max-w-md whitespace-normal px-4 py-3">
                  <p
                    className="line-clamp-2 text-sm text-text-muted"
                    title={repo.ai_summary || repo.description}
                  >
                    {repo.ai_summary || repo.description || (
                      <span className="italic text-text-muted">{t('data.no_summary')}</span>
                    )}
                  </p>
                </TableCell>
                <TableCell className="px-4 py-3 text-center">
                  {repo.language ? (
                    <span className="inline-flex items-center rounded-full bg-bg-hover px-2.5 py-1 text-xs font-medium text-text-muted dark:bg-dark-bg-sidebar dark:text-dark-text-main/70">
                      {repo.language}
                    </span>
                  ) : (
                    <span className="italic text-text-muted">-</span>
                  )}
                </TableCell>
                <TableCell className="px-4 py-3 text-center font-mono tabular-nums text-text-muted">
                  {repo.stargazers_count.toLocaleString()}
                </TableCell>
                <TableCell className="px-4 py-3 text-center">
                  <ClusterBadge
                    cluster={repo.cluster_id != null ? clusterMap.get(repo.cluster_id) : undefined}
                    onClick={() => repo.cluster_id != null && onClusterFilter(repo.cluster_id)}
                  />
                </TableCell>
                <TableCell className="px-4 py-3 text-center text-xs text-text-muted">
                  {formatDataDate(repo.starred_at)}
                </TableCell>
                <TableCell className="px-4 py-3 text-center text-xs text-text-muted">
                  {formatDataDate(repo.last_commit_time)}
                </TableCell>
                <TableCell className="hidden max-w-md whitespace-normal px-4 py-3">
                  <p className="truncate text-xs text-text-muted">
                    {repo.description || (
                      <span className="italic text-text-muted">{t('data.no_description')}</span>
                    )}
                  </p>
                </TableCell>
              </TableRow>
            ))}

            {repos.length === 0 && (
              <TableRow className="hover:bg-transparent">
                <TableCell colSpan={8} className="whitespace-normal">
                  <EmptyState
                    title={hasActiveFilters ? t('data.no_results') : t('data.no_data')}
                    description={t('data.empty_hint')}
                    actionTo={hasActiveFilters ? undefined : '/settings'}
                    actionLabel={hasActiveFilters ? t('common.clear_filters') : t('common.sync_now')}
                    actionType={hasActiveFilters ? 'button' : 'link'}
                    onAction={hasActiveFilters ? onClearFilters : undefined}
                  />
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
    </Card>

    <div className="flex flex-col gap-3 sm:hidden">
      {repos.map((repo) => (
        <Card key={repo.id} className="p-4">
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
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-border-light text-text-muted dark:bg-dark-border dark:text-dark-text-main/60">
                {repo.owner.charAt(0).toUpperCase()}
              </div>
            )}
            <div className="min-w-0 flex-1">
              <Link
                to={`/graph?node=${repo.id}`}
                className="block min-h-[1.75rem] truncate py-1 font-semibold text-text-main hover:text-action-primary"
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
              onClick={() => repo.cluster_id != null && onClusterFilter(repo.cluster_id)}
            />
            <span>{t('data.starred_date')}: {formatDataDate(repo.starred_at)}</span>
            <span>{t('data.last_commit')}: {formatDataDate(repo.last_commit_time)}</span>
          </div>
        </Card>
      ))}

      {repos.length === 0 && (
        <EmptyState
          title={hasActiveFilters ? t('data.no_results') : t('data.no_data')}
          description={t('data.empty_hint')}
          actionTo={hasActiveFilters ? undefined : '/settings'}
          actionLabel={hasActiveFilters ? t('common.clear_filters') : t('common.sync_now')}
          actionType={hasActiveFilters ? 'button' : 'link'}
          onAction={hasActiveFilters ? onClearFilters : undefined}
        />
      )}
    </div>
    </>
  );
};
