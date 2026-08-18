import React from 'react';
import { useTranslation } from 'react-i18next';
import { ChevronDown, ChevronUp } from 'lucide-react';

import type { DataClusterInfo } from '../../api/v2/data';
import { getClusterAccent } from '../../utils/clusterAccent';
import type { SortConfig, SortField } from './dataPageFilters';

/**
 * Table primitives for the Data page.
 *
 * `SortableHeader` carries the `aria-sort` state screen readers rely on to
 * announce the current ordering.
 */
interface SortableHeaderProps {
  label: string;
  field: SortField;
  currentSort: SortConfig;
  onSort: (field: SortField) => void;
  className?: string;
  align?: 'left' | 'center' | 'right';
}

export const SortableHeader: React.FC<SortableHeaderProps> = ({
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
              isActive && currentSort.direction === 'asc' ? 'text-action-primary' : 'text-text-muted'
            }`}
          />
          <ChevronDown
            className={`h-3 w-3 ${
              isActive && currentSort.direction === 'desc'
                ? 'text-action-primary'
                : 'text-text-muted'
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

export const ClusterBadge: React.FC<ClusterBadgeProps> = ({ cluster, onClick }) => {
  const { t } = useTranslation();

  if (!cluster) {
    return <span className="text-xs italic text-text-muted">{t('data.unclustered')}</span>;
  }

  const accent = getClusterAccent({ id: cluster.id, color: cluster.color });

  return (
    <button
      type="button"
      onClick={(event) => {
        event.stopPropagation();
        onClick?.();
      }}
      className="chip-button max-w-full"
      style={{
        backgroundColor: accent.softBackground,
        borderColor: accent.softBorder,
        ['--chip-text' as string]: accent.text,
        ['--chip-text-dark' as string]: accent.textOnDark,
      }}
      title={cluster.name || undefined}
    >
      <div className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: accent.dot }} />
      {/* Truncate rather than wrap: a wrapped Chinese cluster name split the
          chip across two lines and made every table row a different height. */}
      <span className="truncate">{cluster.name || `Cluster ${cluster.id}`}</span>
    </button>
  );
};
