import { describe, expect, it, vi } from 'vitest';

import { GRAPH_2D_COLORS as COLORS } from '../graph2dUtils';
import {
  drawClusterHulls,
  groupNodesByCluster,
  hullSignature,
  paintNodeOnCanvas,
  paintNodePointerArea,
} from '../graph2dPainters';
import type { HullCache, ImageCache, ProcessedNode } from '../graph2dTypes';
import type { ClusterInfo } from '../../../types';

const node = (overrides: Partial<ProcessedNode> = {}): ProcessedNode =>
  ({
    id: 1,
    name: 'nebula',
    full_name: 'octo/nebula',
    cluster_id: 1,
    color: '#123456',
    size: 1,
    stargazers_count: 100,
    x: 10,
    y: 20,
    ...overrides,
  }) as ProcessedNode;

const fakeCtx = () => {
  const calls: string[] = [];
  const ctx = {
    calls,
    globalAlpha: 1,
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 0,
    font: '',
    textAlign: '',
    textBaseline: '',
    save: vi.fn(() => calls.push('save')),
    restore: vi.fn(() => calls.push('restore')),
    beginPath: vi.fn(() => calls.push('beginPath')),
    closePath: vi.fn(() => calls.push('closePath')),
    arc: vi.fn(() => calls.push('arc')),
    clip: vi.fn(() => calls.push('clip')),
    fill: vi.fn(() => calls.push('fill')),
    stroke: vi.fn(() => calls.push('stroke')),
    moveTo: vi.fn(() => calls.push('moveTo')),
    lineTo: vi.fn(() => calls.push('lineTo')),
    drawImage: vi.fn(() => calls.push('drawImage')),
    fillRect: vi.fn(() => calls.push('fillRect')),
    fillText: vi.fn((text: string) => calls.push(`fillText:${text}`)),
    measureText: vi.fn(() => ({ width: 40 })),
  };
  return ctx as unknown as CanvasRenderingContext2D & { calls: string[] };
};

const basePaintOptions = {
  globalScale: 1,
  color: '#123456',
  isVisible: true,
  isSelected: false,
  isHovered: false,
  hqRendering: false,
  onAvatarLoaded: () => {},
};

describe('paintNodeOnCanvas', () => {
  it('draws nothing for a node without coordinates', () => {
    const ctx = fakeCtx();
    paintNodeOnCanvas({
      ...basePaintOptions,
      node: node({ x: undefined, y: undefined }),
      ctx,
      imageCache: new Map() as ImageCache,
    });

    expect(ctx.calls).toEqual([]);
  });

  it('draws a solid circle when there is no avatar', () => {
    const ctx = fakeCtx();
    paintNodeOnCanvas({
      ...basePaintOptions,
      node: node(),
      ctx,
      imageCache: new Map() as ImageCache,
    });

    expect(ctx.calls).toContain('arc');
    expect(ctx.calls).toContain('fill');
    expect(ctx.calls).not.toContain('drawImage');
  });

  it('dims a filtered-out node that is neither selected nor hovered', () => {
    const ctx = fakeCtx();
    paintNodeOnCanvas({
      ...basePaintOptions,
      node: node(),
      ctx,
      isVisible: false,
      imageCache: new Map() as ImageCache,
    });

    expect(ctx.globalAlpha).toBe(0.25);
  });

  it('does not dim a filtered-out node that is selected', () => {
    const ctx = fakeCtx();
    paintNodeOnCanvas({
      ...basePaintOptions,
      node: node(),
      ctx,
      isVisible: false,
      isSelected: true,
      imageCache: new Map() as ImageCache,
    });

    expect(ctx.globalAlpha).toBe(1);
  });

  it('draws a cached avatar inside a circular clip', () => {
    const ctx = fakeCtx();
    const cache = new Map() as ImageCache;
    const image = Object.create(HTMLImageElement.prototype) as HTMLImageElement;
    cache.set('https://avatars/octo', image);

    paintNodeOnCanvas({
      ...basePaintOptions,
      node: node({ owner_avatar_url: 'https://avatars/octo' }),
      ctx,
      imageCache: cache,
    });

    expect(ctx.calls).toContain('clip');
    expect(ctx.calls).toContain('drawImage');
  });

  it('marks an unseen avatar as loading exactly once', () => {
    const ctx = fakeCtx();
    const cache = new Map() as ImageCache;

    paintNodeOnCanvas({
      ...basePaintOptions,
      node: node({ owner_avatar_url: 'https://avatars/octo' }),
      ctx,
      imageCache: cache,
    });

    // The placeholder is what stops a second paint from starting a second fetch.
    expect(cache.get('https://avatars/octo')).toBe('loading');
  });

  it('does not redraw an avatar that previously failed', () => {
    const ctx = fakeCtx();
    const cache = new Map() as ImageCache;
    cache.set('https://avatars/octo', 'error');

    paintNodeOnCanvas({
      ...basePaintOptions,
      node: node({ owner_avatar_url: 'https://avatars/octo' }),
      ctx,
      imageCache: cache,
    });

    expect(ctx.calls).not.toContain('drawImage');
    expect(ctx.calls).toContain('fill');
    expect(cache.get('https://avatars/octo')).toBe('error');
  });

  it('strokes a border for the selected node and a glow only in HQ mode', () => {
    const plain = fakeCtx();
    paintNodeOnCanvas({
      ...basePaintOptions,
      node: node(),
      ctx: plain,
      isSelected: true,
      imageCache: new Map() as ImageCache,
    });
    const plainStrokes = plain.calls.filter((call) => call === 'stroke').length;

    const hq = fakeCtx();
    paintNodeOnCanvas({
      ...basePaintOptions,
      node: node(),
      ctx: hq,
      isSelected: true,
      hqRendering: true,
      imageCache: new Map() as ImageCache,
    });
    const hqStrokes = hq.calls.filter((call) => call === 'stroke').length;

    expect(hqStrokes).toBe(plainStrokes + 1);
  });

  it('labels a hovered node even when it is filtered out', () => {
    const ctx = fakeCtx();
    paintNodeOnCanvas({
      ...basePaintOptions,
      node: node({ name: 'nebula' }),
      ctx,
      isVisible: false,
      isHovered: true,
      imageCache: new Map() as ImageCache,
    });

    expect(ctx.calls).toContain('fillText:nebula');
  });

  it('hides the label for a small, unfocused node at low zoom', () => {
    const ctx = fakeCtx();
    paintNodeOnCanvas({
      ...basePaintOptions,
      node: node({ stargazers_count: 1 }),
      ctx,
      globalScale: 1,
      imageCache: new Map() as ImageCache,
    });

    expect(ctx.calls.some((call) => call.startsWith('fillText'))).toBe(false);
  });

  it('whitens a dimmed avatar node so it reads as out of focus', () => {
    const ctx = fakeCtx();
    const cache = new Map() as ImageCache;
    cache.set('https://avatars/octo', Object.create(HTMLImageElement.prototype));

    paintNodeOnCanvas({
      ...basePaintOptions,
      node: node({ owner_avatar_url: 'https://avatars/octo' }),
      ctx,
      color: COLORS.NODE_DIM,
      imageCache: cache,
    });

    expect(ctx.calls.filter((call) => call === 'fill').length).toBeGreaterThan(0);
  });

  it('balances every save with a restore', () => {
    const ctx = fakeCtx();
    const cache = new Map() as ImageCache;
    cache.set('https://avatars/octo', Object.create(HTMLImageElement.prototype));

    paintNodeOnCanvas({
      ...basePaintOptions,
      node: node({ owner_avatar_url: 'https://avatars/octo' }),
      ctx,
      isSelected: true,
      hqRendering: true,
      imageCache: cache,
    });

    // An unbalanced save leaks clip/alpha state into every later node.
    expect(ctx.calls.filter((c) => c === 'save').length).toBe(
      ctx.calls.filter((c) => c === 'restore').length
    );
  });
});

describe('paintNodePointerArea', () => {
  it('paints a hit area larger than the visual radius', () => {
    const ctx = fakeCtx();
    paintNodePointerArea(node(), '#ff0000', ctx);

    expect(ctx.calls).toEqual(['beginPath', 'arc', 'fill']);
    expect(ctx.fillStyle).toBe('#ff0000');
  });

  it('skips a node without coordinates', () => {
    const ctx = fakeCtx();
    paintNodePointerArea(node({ x: undefined, y: undefined }), '#ff0000', ctx);

    expect(ctx.calls).toEqual([]);
  });
});

describe('groupNodesByCluster', () => {
  it('excludes filtered-out, unclustered, and unpositioned nodes', () => {
    const nodes = [
      node({ id: 1, cluster_id: 5 }),
      node({ id: 2, cluster_id: 5 }),
      node({ id: 3, cluster_id: null }),
      node({ id: 4, cluster_id: 5, x: undefined }),
      node({ id: 5, cluster_id: 5 }),
    ];

    const grouped = groupNodesByCluster(nodes, new Set([1, 2, 3, 4]));

    expect(grouped.get(5)?.map((n) => n.id)).toEqual([1, 2]);
    expect(grouped.has(null as unknown as number)).toBe(false);
  });
});

describe('hullSignature', () => {
  it('rounds coordinates so sub-pixel jitter does not invalidate the cache', () => {
    expect(hullSignature([{ x: 10.4, y: 20.4 }])).toBe(hullSignature([{ x: 10.2, y: 20.1 }]));
  });

  it('changes when a point moves by a whole pixel', () => {
    expect(hullSignature([{ x: 10, y: 20 }])).not.toBe(hullSignature([{ x: 11, y: 20 }]));
  });
});

describe('drawClusterHulls', () => {
  const cluster = (id: number): ClusterInfo =>
    ({ id, name: `Cluster ${id}`, color: '#445566', repo_count: 3, keywords: [] }) as ClusterInfo;

  it('skips clusters with fewer than three visible nodes', () => {
    const ctx = fakeCtx();
    drawClusterHulls({
      nodes: [node({ id: 1, cluster_id: 5 }), node({ id: 2, cluster_id: 5, x: 30 })],
      ctx,
      globalScale: 1,
      visibleNodeIds: new Set([1, 2]),
      clusterGroups: new Map([[5, cluster(5)]]),
      hullCache: new Map() as HullCache,
    });

    expect(ctx.calls).toEqual([]);
  });

  it('draws and labels a hull for a cluster with enough nodes', () => {
    const ctx = fakeCtx();
    drawClusterHulls({
      nodes: [
        node({ id: 1, cluster_id: 5, x: 0, y: 0 }),
        node({ id: 2, cluster_id: 5, x: 50, y: 0 }),
        node({ id: 3, cluster_id: 5, x: 25, y: 40 }),
      ],
      ctx,
      globalScale: 1,
      visibleNodeIds: new Set([1, 2, 3]),
      clusterGroups: new Map([[5, cluster(5)]]),
      hullCache: new Map() as HullCache,
    });

    expect(ctx.calls).toContain('closePath');
    expect(ctx.calls).toContain('fillText:Cluster 5');
  });

  it('reuses a cached hull when positions have not moved', () => {
    const cache = new Map() as HullCache;
    const nodes = [
      node({ id: 1, cluster_id: 5, x: 0, y: 0 }),
      node({ id: 2, cluster_id: 5, x: 50, y: 0 }),
      node({ id: 3, cluster_id: 5, x: 25, y: 40 }),
    ];
    const options = {
      nodes,
      globalScale: 1,
      visibleNodeIds: new Set([1, 2, 3]),
      clusterGroups: new Map([[5, cluster(5)]]),
      hullCache: cache,
    };

    drawClusterHulls({ ...options, ctx: fakeCtx() });
    const firstHull = cache.get(5)?.hull;
    drawClusterHulls({ ...options, ctx: fakeCtx() });

    expect(cache.get(5)?.hull).toBe(firstHull);
  });

  it('skips a cluster with no metadata', () => {
    const ctx = fakeCtx();
    drawClusterHulls({
      nodes: [
        node({ id: 1, cluster_id: 9, x: 0, y: 0 }),
        node({ id: 2, cluster_id: 9, x: 50, y: 0 }),
        node({ id: 3, cluster_id: 9, x: 25, y: 40 }),
      ],
      ctx,
      globalScale: 1,
      visibleNodeIds: new Set([1, 2, 3]),
      clusterGroups: new Map(),
      hullCache: new Map() as HullCache,
    });

    expect(ctx.calls).toEqual([]);
  });

  it('omits the cluster label when zoomed far out', () => {
    const ctx = fakeCtx();
    drawClusterHulls({
      nodes: [
        node({ id: 1, cluster_id: 5, x: 0, y: 0 }),
        node({ id: 2, cluster_id: 5, x: 50, y: 0 }),
        node({ id: 3, cluster_id: 5, x: 25, y: 40 }),
      ],
      ctx,
      globalScale: 0.3,
      visibleNodeIds: new Set([1, 2, 3]),
      clusterGroups: new Map([[5, cluster(5)]]),
      hullCache: new Map() as HullCache,
    });

    expect(ctx.calls.some((call) => call.startsWith('fillText'))).toBe(false);
  });
});
