import React from 'react';
import { clsx } from 'clsx';
import { ArrowRight } from 'lucide-react';

import type { SearchResult } from './commandPaletteTypes';

/**
 * Result rows for the command palette.
 *
 * Presentational only: the parent owns selection state and keyboard
 * navigation, and reads `data-index` off these rows to scroll the active one
 * into view.
 */
export const CommandPaletteResultList: React.FC<{
  results: SearchResult[];
  selectedIndex: number;
  onSelect: (result: SearchResult) => void;
}> = ({ results, selectedIndex, onSelect }) => (
  <div className="py-2">
      {results.map((result, idx) => (
        <button
          key={`${result.type}-${result.id}`}
          id={`command-palette-option-${idx}`}
          data-index={idx}
          onClick={() => onSelect(result)}
          className={clsx(
            'w-full flex items-center gap-3 px-4 py-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30',
            idx === selectedIndex
              ? 'bg-action-primary/10'
              : 'hover:bg-bg-hover'
          )}
          role="option"
          aria-selected={idx === selectedIndex}
        >
          {/* Icon */}
          <div className="flex-shrink-0">
            {result.icon}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-medium text-text-main truncate">
                {result.title}
              </span>
              <span className={clsx(
                'text-[10px] px-1.5 py-0.5 rounded uppercase',
                result.type === 'repo' && 'bg-bg-hover text-text-muted dark:bg-dark-bg-sidebar dark:text-dark-text-main/70',
                result.type === 'cluster' && 'bg-bg-hover text-text-muted dark:bg-dark-bg-sidebar dark:text-dark-text-main/70',
                result.type === 'language' && 'bg-action-primary/10 text-action-primary',
                result.type === 'tag' && 'bg-action-primary/10 text-action-primary',
              )}>
                {result.type}
              </span>
            </div>
            {result.subtitle && (
              <p className="text-sm text-text-muted truncate mt-0.5">
                {result.subtitle}
              </p>
            )}
          </div>

          {/* Meta */}
          {result.meta && (
            <span className="text-xs text-text-dim flex-shrink-0">
              {result.meta}
            </span>
          )}

          {/* Arrow */}
          <ArrowRight className={clsx(
            'w-4 h-4 flex-shrink-0 transition-opacity',
            idx === selectedIndex ? 'opacity-100 text-action-primary' : 'opacity-0'
          )} />
        </button>
      ))}
  </div>
);
