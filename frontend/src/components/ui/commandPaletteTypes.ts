import React from 'react';

import type { ClusterInfo, GraphNode } from '../../types';

export interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectNode?: (node: GraphNode) => void;
  onSelectCluster?: (cluster: ClusterInfo) => void;
  onSelectSearch?: (value: string, facet?: 'search' | 'language' | 'tag') => void;
}

export type FilterType = 'all' | 'repos' | 'clusters' | 'languages' | 'tags';

export interface SearchResultBase {
  id: string | number;
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
  meta?: string;
}

export interface RepoSearchResult extends SearchResultBase {
  type: 'repo';
  data: GraphNode;
  source?: 'local' | 'remote';
}

export interface ClusterSearchResult extends SearchResultBase {
  type: 'cluster';
  data: ClusterInfo;
}

export interface LanguageSearchResult extends SearchResultBase {
  type: 'language';
  data: { language: string };
}

export interface TagSearchResult extends SearchResultBase {
  type: 'tag';
  data: { tag: string };
}

export type SearchResult =
  | RepoSearchResult
  | ClusterSearchResult
  | LanguageSearchResult
  | TagSearchResult;

export const RECENT_SEARCHES_KEY = 'nebula_recent_searches';
export const MAX_RECENT_SEARCHES = 5;
export const MAX_RESULTS = 20;
