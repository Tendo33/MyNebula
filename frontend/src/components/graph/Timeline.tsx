import React, { useMemo, useState, useCallback, useRef, useEffect } from 'react';
import { clsx } from 'clsx';
import { useTranslation } from 'react-i18next';
import { Calendar, X } from 'lucide-react';
import { useGraph } from '../../contexts/GraphContext';
import { Button } from '../ui/button';
import { Card } from '../ui/card';

// ============================================================================
// Types
// ============================================================================

interface TimelineProps {
  className?: string;
}

// ============================================================================
// Constants
// ============================================================================

const BAR_MIN_HEIGHT = 4;
const BAR_MAX_HEIGHT = 48;

// ============================================================================
// Component
// ============================================================================

const Timeline: React.FC<TimelineProps> = ({ className }) => {
  const { t, i18n } = useTranslation();
  const {
    timelineData,
    filters,
    setTimeRange,
  } = useGraph();

  // Local state for drag interaction
  const [isDragging, setIsDragging] = useState(false);
  const [dragHandle, setDragHandle] = useState<'start' | 'end' | 'range' | null>(null);
  const [localRange, setLocalRange] = useState<[number, number] | null>(null);
  const [dragOffset, setDragOffset] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  // Get points from timeline data
  const points = useMemo(() => {
    if (!timelineData || timelineData.points.length === 0) {
      return [];
    }
    return timelineData.points;
  }, [timelineData]);

  // Calculate max count for scaling
  const maxCount = useMemo(() => {
    if (points.length === 0) return 1;
    return Math.max(...points.map(p => p.count), 1);
  }, [points]);

  // Current range (local when dragging, or from filters)
  const currentRange = useMemo((): [number, number] | null => {
    if (isDragging && localRange) return localRange;
    return filters.timeRange;
  }, [isDragging, localRange, filters.timeRange]);

  // Check if a bar is in the selected range
  const isInRange = useCallback((index: number): boolean => {
    if (!currentRange) return true;
    return index >= currentRange[0] && index <= currentRange[1];
  }, [currentRange]);

  // Format date for display
  const monthFormatter = useMemo(
    () => new Intl.DateTimeFormat(i18n.resolvedLanguage ?? i18n.language, {
      month: 'short',
      year: 'numeric',
      timeZone: 'UTC',
    }),
    [i18n.language, i18n.resolvedLanguage]
  );

  const formatDate = useCallback(
    (dateStr: string): string => monthFormatter.format(new Date(`${dateStr}-01T00:00:00Z`)),
    [monthFormatter]
  );

  // Handle pointer down on bar
  const handleBarPointerDown = useCallback((index: number, e: React.PointerEvent) => {
    e.preventDefault();

    if (!currentRange) {
      // Start new selection
      setLocalRange([index, index]);
      setDragHandle('end');
      setDragOffset(0);
    } else if (index === currentRange[0]) {
      setLocalRange(currentRange);
      setDragHandle('start');
      setDragOffset(0);
    } else if (index === currentRange[1]) {
      setLocalRange(currentRange);
      setDragHandle('end');
      setDragOffset(0);
    } else if (index > currentRange[0] && index < currentRange[1]) {
      setLocalRange(currentRange);
      setDragHandle('range');
      setDragOffset(index - currentRange[0]);
    } else {
      // Click outside range - start new selection
      setLocalRange([index, index]);
      setDragHandle('end');
      setDragOffset(0);
    }

    setIsDragging(true);
  }, [currentRange]);

  // Handle pointer move during drag
  const handlePointerMove = useCallback((e: PointerEvent) => {
    if (!isDragging || !containerRef.current || points.length === 0) return;

    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const barWidth = rect.width / points.length;
    const index = Math.max(0, Math.min(points.length - 1, Math.floor(x / barWidth)));

    setLocalRange(prev => {
      if (!prev) return [index, index];

      switch (dragHandle) {
        case 'start':
          return [Math.min(index, prev[1]), prev[1]];
        case 'end':
          return [prev[0], Math.max(index, prev[0])];
        case 'range':
          {
            const rangeWidth = prev[1] - prev[0];
            const maxStart = Math.max(0, points.length - 1 - rangeWidth);
            const nextStart = Math.max(0, Math.min(index - dragOffset, maxStart));
            return [nextStart, nextStart + rangeWidth];
          }
        default:
          return prev;
      }
    });
  }, [isDragging, dragHandle, points.length, dragOffset]);

  // Handle pointer up
  const handlePointerUp = useCallback(() => {
    if (isDragging && localRange) {
      // Commit the range
      if (localRange[0] === 0 && localRange[1] === points.length - 1) {
        // Full range selected = no filter
        setTimeRange(null);
      } else {
        setTimeRange(localRange);
      }
    }
    setIsDragging(false);
    setDragHandle(null);
    setLocalRange(null);
    setDragOffset(0);
  }, [isDragging, localRange, points.length, setTimeRange]);

  // Clear selection
  const handleClearSelection = useCallback(() => {
    setTimeRange(null);
  }, [setTimeRange]);

  // Attach global pointer events for dragging
  useEffect(() => {
    if (isDragging) {
      window.addEventListener('pointermove', handlePointerMove);
      window.addEventListener('pointerup', handlePointerUp);
      return () => {
        window.removeEventListener('pointermove', handlePointerMove);
        window.removeEventListener('pointerup', handlePointerUp);
      };
    }
  }, [isDragging, handlePointerMove, handlePointerUp]);

  // Empty state
  if (points.length === 0) {
    return null;
  }

  return (
    <Card className={clsx('p-4', className)}>
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Calendar className="size-4 text-text-muted" />
          <span className="text-xs font-medium uppercase tracking-wider text-text-muted">
            {t('graph.timeline')}
          </span>
        </div>

        {currentRange && (
          <Button type="button" variant="ghost" className="h-7 px-2 text-xs" onClick={handleClearSelection}>
            <X data-icon="inline-start" />
            {t('common.clear')}
          </Button>
        )}
      </div>

      {/* Bars */}
      <div
        ref={containerRef}
        className="flex items-end gap-0.5 h-14 cursor-crosshair select-none"
        onPointerDown={(e) => {
          // Handle click on empty space
          if (!containerRef.current || points.length === 0) return;

          const rect = containerRef.current.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const barWidth = rect.width / points.length;
          const index = Math.max(0, Math.min(points.length - 1, Math.floor(x / barWidth)));

          handleBarPointerDown(index, e);
        }}
      >
        {points.map((point, idx) => {
          const heightPercent = (point.count / maxCount) * 100;
          const height = BAR_MIN_HEIGHT + (heightPercent / 100) * (BAR_MAX_HEIGHT - BAR_MIN_HEIGHT);
          const inRange = isInRange(idx);
          const isRangeStart = currentRange && idx === currentRange[0];
          const isRangeEnd = currentRange && idx === currentRange[1];

          return (
            <button
              type="button"
              key={point.date}
              aria-pressed={Boolean(currentRange && idx >= currentRange[0] && idx <= currentRange[1])}
              aria-label={`${formatDate(point.date)}: ${t('search.repo_count', '{{count}} repos', { count: point.count })}`}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  setTimeRange([idx, idx]);
                } else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
                  event.preventDefault();
                  const nextIndex = Math.max(
                    0,
                    Math.min(points.length - 1, idx + (event.key === 'ArrowRight' ? 1 : -1))
                  );
                  const nextBar = containerRef.current?.children[nextIndex] as HTMLElement | undefined;
                  nextBar?.focus();
                } else if (event.key === 'Escape') {
                  handleClearSelection();
                }
              }}
              className="flex-1 flex flex-col justify-end items-center group relative rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary focus-visible:ring-offset-1"
            >
              {/* Bar */}
              <div
                className={clsx(
                  'w-full rounded-t-sm transition-all duration-150',
                  inRange
                    ? 'bg-action-primary hover:bg-action-hover'
                    : 'bg-border-light/80 hover:bg-border-light dark:bg-dark-border/60 dark:hover:bg-dark-border',
                  (isRangeStart || isRangeEnd) && 'ring-2 ring-action-primary ring-offset-1'
                )}
                style={{ height: `${height}px` }}
              />

              {/* Tooltip */}
              <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 group-focus:opacity-100 transition-opacity pointer-events-none z-50">
                <div className="bg-text-main text-bg-main px-2.5 py-1.5 rounded-md text-xs whitespace-nowrap shadow-lg">
                  <div className="font-semibold">{t('search.repo_count', '{{count}} repos', { count: point.count })}</div>
                  <div className="text-bg-main/70">{formatDate(point.date)}</div>
                  {point.top_languages.length > 0 && (
                    <div className="mt-1 text-bg-main/60 text-[10px]">
                      {point.top_languages.slice(0, 3).join(', ')}
                    </div>
                  )}
                </div>
                {/* Tooltip arrow */}
                <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1">
                  <div className="border-4 border-transparent border-t-text-main" />
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Date labels */}
      <div className="flex justify-between mt-2 text-[10px] text-text-muted font-mono">
        <span>{points.length > 0 ? formatDate(points[0].date) : ''}</span>
        <span>{points.length > 0 ? formatDate(points[points.length - 1].date) : ''}</span>
      </div>

      {/* Selection info */}
      {currentRange && (
        <div className="mt-2 pt-2 border-t border-border-light">
          <div className="flex items-center justify-between text-xs">
            <span className="text-text-muted">
              {formatDate(points[currentRange[0]].date)} — {formatDate(points[currentRange[1]].date)}
            </span>
            <span className="text-text-main font-medium">
              {t('search.repo_count', '{{count}} repos', {
                count: points
                  .slice(currentRange[0], currentRange[1] + 1)
                  .reduce((sum, p) => sum + p.count, 0),
              })}
            </span>
          </div>
        </div>
      )}
    </Card>
  );
};

export default Timeline;
