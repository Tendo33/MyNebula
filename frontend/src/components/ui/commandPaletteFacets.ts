import type { GraphNode } from '../../types';

/**
 * Facet counting for the command palette.
 *
 * Pure so the ranking and truncation rules are directly testable; inside the
 * component these were inline `useMemo`s over `rawData`.
 */

export const LANGUAGE_FACET_LIMIT = 10;
export const TAG_FACET_LIMIT = 20;

const rankedEntries = (counts: Record<string, number>, limit: number): [string, number][] =>
  Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit);

/** Top languages by repo count. */
export const buildLanguageFacets = (nodes: GraphNode[] | undefined): [string, number][] => {
  if (!nodes) return [];
  const counts: Record<string, number> = {};
  nodes.forEach((node) => {
    if (node.language) {
      counts[node.language] = (counts[node.language] || 0) + 1;
    }
  });
  return rankedEntries(counts, LANGUAGE_FACET_LIMIT);
};

/**
 * Top tags by repo count, pooling AI tags and GitHub topics into one namespace
 * — the palette treats them as a single "tag" facet.
 */
export const buildTagFacets = (nodes: GraphNode[] | undefined): [string, number][] => {
  if (!nodes) return [];
  const counts: Record<string, number> = {};
  nodes.forEach((node) => {
    (node.ai_tags || []).forEach((tag) => {
      counts[tag] = (counts[tag] || 0) + 1;
    });
    (node.topics || []).forEach((topic) => {
      counts[topic] = (counts[topic] || 0) + 1;
    });
  });
  return rankedEntries(counts, TAG_FACET_LIMIT);
};
