import React from 'react';
import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import { Loader2 } from 'lucide-react';

const graphSkeletonDots = [
  { top: '18%', left: '14%' },
  { top: '26%', left: '22%' },
  { top: '33%', left: '36%' },
  { top: '25%', left: '58%' },
  { top: '18%', left: '72%' },
  { top: '40%', left: '17%' },
  { top: '44%', left: '31%' },
  { top: '48%', left: '45%' },
  { top: '42%', left: '61%' },
  { top: '50%', left: '76%' },
  { top: '60%', left: '22%' },
  { top: '64%', left: '37%' },
  { top: '68%', left: '51%' },
  { top: '63%', left: '66%' },
  { top: '74%', left: '30%' },
  { top: '77%', left: '47%' },
  { top: '72%', left: '61%' },
  { top: '32%', left: '80%' },
  { top: '57%', left: '9%' },
  { top: '80%', left: '73%' },
];

// ============================================================================
// Base Skeleton Component
// ============================================================================

interface SkeletonProps {
  className?: string;
  animate?: boolean;
  style?: React.CSSProperties;
}

export const Skeleton: React.FC<SkeletonProps> = ({ className, animate = true, style }) => (
  <div
    style={style}
    className={clsx(
      'rounded bg-bg-hover/90 dark:bg-dark-bg-sidebar/70',
      animate && 'animate-pulse',
      className
    )}
  />
);

// ============================================================================
// Graph Skeleton
// ============================================================================

export const GraphSkeleton: React.FC = () => {
  const { t } = useTranslation();

  return (
    <div className="w-full h-full flex items-center justify-center bg-bg-hover/50 relative overflow-hidden dark:bg-dark-bg-sidebar/60">
      {/* Animated background circles */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="absolute w-48 h-48 rounded-full bg-bg-hover/70 animate-pulse opacity-50 dark:bg-dark-bg-sidebar/70" style={{ top: '15%', left: '20%' }} />
        <div className="absolute w-64 h-64 rounded-full bg-bg-hover/70 animate-pulse opacity-40 dark:bg-dark-bg-sidebar/70" style={{ top: '30%', right: '15%' }} />
        <div className="absolute w-40 h-40 rounded-full bg-bg-hover/70 animate-pulse opacity-60 dark:bg-dark-bg-sidebar/70" style={{ bottom: '20%', left: '30%' }} />
        <div className="absolute w-56 h-56 rounded-full bg-bg-hover/70 animate-pulse opacity-35 dark:bg-dark-bg-sidebar/70" style={{ bottom: '25%', right: '25%' }} />

        {graphSkeletonDots.map((dot, i) => (
          <div
            key={i}
            className="absolute w-3 h-3 rounded-full bg-border-light animate-pulse dark:bg-dark-border"
            style={{
              top: dot.top,
              left: dot.left,
              animationDelay: `${i * 0.1}s`,
            }}
          />
        ))}
      </div>

      {/* Center loading text */}
      <div className="relative z-10 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-bg-main/80 backdrop-blur-sm rounded-lg shadow-sm border border-border-light dark:bg-dark-bg-main/80 dark:border-dark-border">
          <Loader2 className="w-4 h-4 animate-spin text-action-primary" />
          <span className="text-sm text-text-muted dark:text-dark-text-main/70">{t('graph.loading_data')}</span>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Dashboard Stat Card Skeleton
// ============================================================================

export const StatCardSkeleton: React.FC = () => (
  <div className="panel-surface-strong p-6">
    <Skeleton className="h-4 w-24 mb-3" />
    <Skeleton className="h-8 w-16 mb-2" />
    <Skeleton className="h-3 w-20" />
  </div>
);

// ============================================================================
// Dashboard Skeleton
// ============================================================================

export const DashboardSkeleton: React.FC = () => (
  <div className="max-w-6xl mx-auto space-y-8 p-8">
    {/* Stats Grid */}
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {[...Array(4)].map((_, i) => (
        <StatCardSkeleton key={i} />
      ))}
    </div>

    {/* Charts Row */}
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Language Distribution */}
      <div className="panel-surface p-6">
        <Skeleton className="h-5 w-40 mb-6" />
        <div className="space-y-4">
          {[...Array(5)].map((_, i) => (
            <div key={i}>
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <Skeleton className="w-3 h-3 rounded-full" />
                  <Skeleton className="h-4 w-20" />
                </div>
                <Skeleton className="h-3 w-16" />
              </div>
              <Skeleton className="h-2 w-full rounded-full" />
            </div>
          ))}
        </div>
      </div>

      {/* Activity Timeline */}
      <div className="panel-surface p-6">
        <Skeleton className="h-5 w-32 mb-6" />
        <div className="flex items-end gap-1 h-32">
          {[...Array(12)].map((_, i) => (
            <div key={i} className="flex-1">
              <Skeleton
                className="w-full rounded-t"
                style={{ height: `${20 + Math.random() * 80}%` }}
              />
            </div>
          ))}
        </div>
        <div className="flex justify-between mt-2">
          <Skeleton className="h-3 w-16" />
          <Skeleton className="h-3 w-16" />
        </div>
      </div>
    </div>

    {/* Clusters Grid */}
    <div>
      <Skeleton className="h-5 w-28 mb-4" />
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="panel-surface p-4">
            <div className="flex items-start gap-3">
              <Skeleton className="w-4 h-4 rounded-full flex-shrink-0" />
              <div className="flex-1">
                <Skeleton className="h-4 w-32 mb-2" />
                <Skeleton className="h-3 w-20 mb-2" />
                <div className="flex gap-1">
                  {[...Array(3)].map((_, j) => (
                    <Skeleton key={j} className="h-5 w-12 rounded" />
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

// ============================================================================
// Repo Details Panel Skeleton
// ============================================================================

export const RepoDetailsSkeleton: React.FC = () => (
  <div className="panel-surface-strong absolute top-6 right-6 z-30 w-96 overflow-hidden">
    {/* Header */}
    <div className="border-b border-border-light p-5 bg-bg-sidebar">
      <div className="flex items-start gap-3">
        <Skeleton className="w-10 h-10 rounded-lg flex-shrink-0" />
        <div className="flex-1">
          <Skeleton className="h-5 w-32 mb-2" />
          <Skeleton className="h-3 w-20 mb-2" />
          <Skeleton className="h-4 w-full" />
        </div>
      </div>
    </div>

    {/* Content */}
    <div className="p-5 space-y-5">
      {/* Stats */}
      <div className="grid grid-cols-2 gap-3">
        {[...Array(2)].map((_, i) => (
          <div key={i} className="rounded-2xl border border-border-light p-3">
            <Skeleton className="h-6 w-16 mb-1" />
            <Skeleton className="h-3 w-12" />
          </div>
        ))}
      </div>

      {/* Tags */}
      <div>
        <Skeleton className="h-3 w-20 mb-2" />
        <div className="flex flex-wrap gap-1.5">
          {[...Array(5)].map((_, i) => (
            <Skeleton key={i} className="h-6 w-16 rounded-full" />
          ))}
        </div>
      </div>

      {/* AI Summary */}
      <div>
        <Skeleton className="h-3 w-24 mb-2" />
        <Skeleton className="h-20 w-full rounded-md" />
      </div>

      {/* Button */}
      <Skeleton className="h-10 w-full rounded-md" />
    </div>
  </div>
);

// ============================================================================
// Cluster Panel Skeleton
// ============================================================================

export const ClusterPanelSkeleton: React.FC = () => (
  <div className="panel-surface overflow-hidden">
    {/* Header */}
    <div className="flex items-center justify-between px-4 py-3 border-b border-border-light">
      <div className="flex items-center gap-2">
        <Skeleton className="w-4 h-4" />
        <Skeleton className="h-4 w-20" />
      </div>
    </div>

    {/* List */}
    <div className="p-2 space-y-1">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="flex items-center gap-3 px-3 py-2.5">
          <Skeleton className="w-3 h-3 rounded-full flex-shrink-0" />
          <div className="flex-1">
            <Skeleton className="h-4 w-24 mb-1" />
            <div className="flex gap-1">
              {[...Array(2)].map((_, j) => (
                <Skeleton key={j} className="h-4 w-10 rounded" />
              ))}
            </div>
          </div>
          <Skeleton className="w-8 h-4" />
        </div>
      ))}
    </div>
  </div>
);

export default Skeleton;
