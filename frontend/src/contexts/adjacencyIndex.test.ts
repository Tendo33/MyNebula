import { describe, expect, it } from 'vitest';

import type { GraphEdge } from '../types';
import { buildAdjacencyIndex } from './graphFiltering';

const edge = (source: GraphEdge['source'], target: GraphEdge['target']): GraphEdge =>
  ({ source, target, weight: 0.5 }) as GraphEdge;

describe('buildAdjacencyIndex', () => {
  it('returns an empty index for no edges', () => {
    expect(buildAdjacencyIndex([]).size).toBe(0);
  });

  it('records both directions for an undirected edge', () => {
    const index = buildAdjacencyIndex([edge(1, 2)]);

    expect(index.get(1)).toEqual(new Set([2]));
    expect(index.get(2)).toEqual(new Set([1]));
  });

  it('accumulates multiple neighbours per node', () => {
    const index = buildAdjacencyIndex([edge(1, 2), edge(1, 3), edge(2, 3)]);

    expect(index.get(1)).toEqual(new Set([2, 3]));
    expect(index.get(2)).toEqual(new Set([1, 3]));
    expect(index.get(3)).toEqual(new Set([1, 2]));
  });

  it('handles object endpoints, which react-force-graph substitutes after simulation starts', () => {
    // The library rewrites `source`/`target` from ids into node objects once the
    // force simulation runs. Dropping this branch silently breaks neighbour
    // highlighting after the first render.
    const index = buildAdjacencyIndex([
      edge({ id: 1 } as GraphEdge['source'], { id: 2 } as GraphEdge['target']),
    ]);

    expect(index.get(1)).toEqual(new Set([2]));
    expect(index.get(2)).toEqual(new Set([1]));
  });

  it('handles a mix of id and object endpoints', () => {
    const index = buildAdjacencyIndex([
      edge(1, { id: 2 } as GraphEdge['target']),
      edge({ id: 2 } as GraphEdge['source'], 3),
    ]);

    expect(index.get(2)).toEqual(new Set([1, 3]));
  });

  it('records a self-referential edge without duplicating the node', () => {
    const index = buildAdjacencyIndex([edge(1, 1)]);

    expect(index.get(1)).toEqual(new Set([1]));
    expect(index.size).toBe(1);
  });

  it('deduplicates repeated edges between the same pair', () => {
    const index = buildAdjacencyIndex([edge(1, 2), edge(2, 1), edge(1, 2)]);

    expect(index.get(1)).toEqual(new Set([2]));
    expect(index.get(2)).toEqual(new Set([1]));
  });

  it('returns a stable result for the same edge list', () => {
    const edges = [edge(1, 2), edge(2, 3)];

    expect(buildAdjacencyIndex(edges)).toEqual(buildAdjacencyIndex(edges));
  });
});
