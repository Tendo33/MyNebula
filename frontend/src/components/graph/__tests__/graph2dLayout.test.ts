import { describe, expect, it } from 'vitest';

import type { GraphData, GraphEdge, GraphNode } from '../../../types';
import {
  buildClusterGroups,
  buildClusterLayoutData,
  toProcessedLinks,
  toProcessedNodes,
} from '../graph2dLayout';
import { POSITION_SCALE } from '../graph2dTypes';
import { GRAPH_2D_COLORS as COLORS } from '../graph2dUtils';

const node = (overrides: Partial<GraphNode> = {}): GraphNode =>
  ({
    id: 1,
    github_id: 101,
    full_name: 'octo/nebula',
    name: 'nebula',
    language: 'TypeScript',
    html_url: 'https://github.com/octo/nebula',
    owner: 'octo',
    x: 1,
    y: 2,
    z: 0,
    cluster_id: 1,
    color: '#123456',
    size: 1,
    stargazers_count: 10,
    ...overrides,
  }) as GraphNode;

describe('toProcessedNodes', () => {
  it('returns nothing for missing or empty input', () => {
    expect(toProcessedNodes(undefined)).toEqual([]);
    expect(toProcessedNodes([])).toEqual([]);
  });

  it('scales precomputed coordinates', () => {
    const [processed] = toProcessedNodes([node({ x: 2, y: 3 })]);
    expect(processed.x).toBe(2 * POSITION_SCALE);
    expect(processed.y).toBe(3 * POSITION_SCALE);
  });

  it('drops non-finite coordinates instead of propagating NaN', () => {
    // A NaN here reaches the canvas transform and blanks the whole graph.
    const [processed] = toProcessedNodes([
      node({ x: NaN as unknown as number, y: Infinity as unknown as number }),
    ]);
    expect(processed.x).toBeUndefined();
    expect(processed.y).toBeUndefined();
  });

  it('falls back to the default colour', () => {
    const [processed] = toProcessedNodes([node({ color: '' })]);
    expect(processed.color).toBe(COLORS.NODE_DEFAULT);
  });
});

describe('toProcessedLinks', () => {
  it('returns nothing for missing edges', () => {
    expect(toProcessedLinks(undefined)).toEqual([]);
  });

  it('normalises object endpoints back to ids', () => {
    const edges = [
      { source: { id: 1 }, target: 2, weight: 0.5 },
      { source: 3, target: { id: 4 }, weight: 0.7 },
    ] as unknown as GraphEdge[];

    expect(toProcessedLinks(edges)).toEqual([
      { source: 1, target: 2, weight: 0.5 },
      { source: 3, target: 4, weight: 0.7 },
    ]);
  });
});

describe('buildClusterLayoutData', () => {
  it('averages member positions into a centroid', () => {
    const nodes = toProcessedNodes([
      node({ id: 1, cluster_id: 5, x: 0, y: 0 }),
      node({ id: 2, cluster_id: 5, x: 2, y: 4 }),
    ]);

    const { centers, clusterNodes } = buildClusterLayoutData(nodes);

    expect(clusterNodes.get(5)).toHaveLength(2);
    expect(centers.get(5)?.x).toBe(POSITION_SCALE);
    expect(centers.get(5)?.y).toBe(2 * POSITION_SCALE);
  });

  it('ignores unclustered and unpositioned nodes', () => {
    const nodes = toProcessedNodes([
      node({ id: 1, cluster_id: null }),
      node({ id: 2, cluster_id: 5, x: NaN as unknown as number }),
    ]);

    expect(buildClusterLayoutData(nodes).centers.size).toBe(0);
  });

  it('keeps clusters separate', () => {
    const nodes = toProcessedNodes([
      node({ id: 1, cluster_id: 1, x: 0, y: 0 }),
      node({ id: 2, cluster_id: 2, x: 10, y: 10 }),
    ]);

    expect(buildClusterLayoutData(nodes).centers.size).toBe(2);
  });
});

describe('buildClusterGroups', () => {
  it('returns an empty map for no data', () => {
    expect(buildClusterGroups(null).size).toBe(0);
  });

  it('indexes clusters by id', () => {
    const rawData = {
      clusters: [
        { id: 1, name: 'A' },
        { id: 2, name: 'B' },
      ],
    } as unknown as GraphData;

    const groups = buildClusterGroups(rawData);
    expect(groups.get(2)?.name).toBe('B');
    expect(groups.size).toBe(2);
  });
});
