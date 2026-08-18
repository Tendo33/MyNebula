import React from 'react';

import type { ProcessedNode } from './graph2dTypes';

/**
 * Hover overlay for the graph canvas.
 *
 * Purely presentational: it renders whatever node the canvas reports as
 * hovered. Split out of `Graph2D` so the canvas component holds wiring rather
 * than markup.
 */
export const GraphHoverCard: React.FC<{ node: ProcessedNode }> = ({ node }) => (
  <div className="absolute top-4 right-4 z-10 bg-bg-main px-4 py-3 rounded-lg border border-border-light shadow-lg max-w-sm pointer-events-none dark:bg-dark-bg-main dark:border-dark-border">
    {/* Header with Avatar */}
    <div className="flex items-start gap-3">
      {/* Owner Avatar */}
      {node.owner_avatar_url ? (
        <img
          src={node.owner_avatar_url}
          alt={node.owner || node.name}
          className="w-10 h-10 rounded-md border border-border-light flex-shrink-0"
          loading="lazy"
          decoding="async"
          width={40}
          height={40}
        />
      ) : (
        <div className="w-10 h-10 rounded-md bg-border-light flex items-center justify-center flex-shrink-0 dark:bg-dark-border">
          <span className="text-text-dim text-sm font-medium dark:text-dark-text-main/60">
            {(node.owner || node.name)?.charAt(0).toUpperCase()}
          </span>
        </div>
      )}

      <div className="flex-1 min-w-0">
        <h3 className="text-text-main font-semibold text-sm truncate">{node.name}</h3>
        <div className="flex items-center gap-2 mt-0.5">
          {node.language && (
            <span className="text-[10px] px-1.5 py-0.5 bg-action-primary/10 rounded text-action-primary">
              {node.language}
            </span>
          )}
          <span className="text-xs text-action-primary font-medium">
            ⭐ {node.stargazers_count.toLocaleString()}
          </span>
        </div>
      </div>
    </div>

    {/* Full Description - Not truncated */}
    {(node.description || node.ai_summary) && (
      <p className="text-xs text-text-muted mt-2 leading-relaxed">
        {node.description || node.ai_summary}
      </p>
    )}

    {/* AI Tags Preview */}
    {node.ai_tags && node.ai_tags.length > 0 && (
      <div className="flex flex-wrap gap-1 mt-2">
        {node.ai_tags.slice(0, 4).map((tag: string) => (
          <span
            key={tag}
            className="text-[10px] px-1.5 py-0.5 bg-action-primary/10 text-action-primary rounded"
          >
            {tag}
          </span>
        ))}
        {node.ai_tags.length > 4 && (
          <span className="text-[10px] text-text-dim">+{node.ai_tags.length - 4}</span>
        )}
      </div>
    )}
  </div>
);
