/**
 * Clusters as communities of the similarity graph.
 *
 * The edges the site draws decide the groups, so colour and visible structure
 * always agree. Louvain picks the number of groups; resolution only drops when
 * there are more sizable groups than `maxClusters`.
 */

import type { Edge } from './graph.ts';

const MIN_GROUP_SIZE = 3;
const RESOLUTIONS = [1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05];
const MATCH_JACCARD = 0.5;

/** Small seeded PRNG so node visiting order (and thus the result) is reproducible. */
const mulberry32 = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

/** Weighted Louvain: local moves, then aggregate communities into nodes, repeat. */
export const louvain = (ids: number[], edges: Edge[], resolution = 1, seed = 42): number[][] => {
  const index = new Map(ids.map((id, i) => [id, i]));
  let nbrs: Map<number, number>[] = ids.map(() => new Map());
  let strength = new Float64Array(ids.length);
  let total = 0;
  for (const [s, t, w] of edges) {
    const a = index.get(s);
    const b = index.get(t);
    if (a === undefined || b === undefined || a === b) continue;
    nbrs[a].set(b, (nbrs[a].get(b) ?? 0) + w);
    nbrs[b].set(a, (nbrs[b].get(a) ?? 0) + w);
    strength[a] += w;
    strength[b] += w;
    total += w;
  }
  let members: number[][] = ids.map((_, i) => [i]);
  if (total === 0) return members.map((m) => m.map((i) => ids[i]));
  const random = mulberry32(seed);
  const twoM = 2 * total;

  for (;;) {
    const n = nbrs.length;
    const comm = Array.from({ length: n }, (_, i) => i);
    const tot = Float64Array.from(strength);
    const order = comm.map((i) => [random(), i]).sort((a, b) => a[0] - b[0]).map(([, i]) => i);
    let improved = false;
    for (let moved = true; moved; ) {
      moved = false;
      for (const i of order) {
        const own = comm[i];
        const links = new Map<number, number>();
        for (const [j, w] of nbrs[i]) links.set(comm[j], (links.get(comm[j]) ?? 0) + w);
        tot[own] -= strength[i];
        let best = own;
        let bestGain = (links.get(own) ?? 0) - (resolution * tot[own] * strength[i]) / twoM;
        for (const [c, w] of links) {
          const gain = w - (resolution * tot[c] * strength[i]) / twoM;
          if (gain > bestGain + 1e-12) {
            best = c;
            bestGain = gain;
          }
        }
        tot[best] += strength[i];
        if (best !== own) {
          comm[i] = best;
          moved = improved = true;
        }
      }
    }
    if (!improved) break;

    // Aggregate: each community becomes one node of the next level.
    const renumber = new Map<number, number>();
    comm.forEach((c) => renumber.has(c) || renumber.set(c, renumber.size));
    const size = renumber.size;
    const nextNbrs: Map<number, number>[] = Array.from({ length: size }, () => new Map());
    const nextStrength = new Float64Array(size);
    const nextMembers: number[][] = Array.from({ length: size }, () => []);
    for (let i = 0; i < n; i += 1) {
      const ci = renumber.get(comm[i])!;
      nextStrength[ci] += strength[i];
      nextMembers[ci].push(...members[i]);
      for (const [j, w] of nbrs[i]) {
        const cj = renumber.get(comm[j])!;
        if (ci !== cj) nextNbrs[ci].set(cj, (nextNbrs[ci].get(cj) ?? 0) + w);
      }
    }
    nbrs = nextNbrs;
    strength = nextStrength;
    members = nextMembers;
  }
  return members.map((m) => m.map((i) => ids[i]));
};

/** Louvain groups, largest first; stragglers from tiny communities join their strongest neighbour group. */
export const detectCommunities = (ids: number[], edges: Edge[], maxClusters: number): Set<number>[] => {
  if (!edges.length) return [];
  let groups: Set<number>[] = [];
  for (const resolution of RESOLUTIONS) {
    groups = louvain(ids, edges, resolution)
      .filter((c) => c.length >= MIN_GROUP_SIZE)
      .map((c) => new Set(c));
    if (groups.length <= maxClusters) break;
  }
  groups.sort((a, b) => b.size - a.size || Math.min(...a) - Math.min(...b));

  const label = new Map<number, number>();
  groups.forEach((g, i) => g.forEach((id) => label.set(id, i)));
  const adjacency = new Map<number, [number, number][]>();
  for (const [s, t, w] of edges) {
    adjacency.set(s, [...(adjacency.get(s) ?? []), [t, w]]);
    adjacency.set(t, [...(adjacency.get(t) ?? []), [s, w]]);
  }
  let pending = ids.filter((id) => !label.has(id) && adjacency.has(id));
  while (pending.length) {
    const still = pending.filter((id) => {
      const pull = new Map<number, number>();
      for (const [other, w] of adjacency.get(id)!) {
        const g = label.get(other);
        if (g !== undefined) pull.set(g, (pull.get(g) ?? 0) + w);
      }
      if (!pull.size) return true;
      const best = [...pull.entries()].sort((a, b) => b[1] - a[1])[0][0];
      label.set(id, best);
      groups[best].add(id);
      return false;
    });
    if (still.length === pending.length) break;
    pending = still;
  }
  return groups;
};

export interface ClusterInfo {
  name: string;
  description: string | null;
  keywords: string[];
  members: number[];
  named_by: 'llm' | 'tags';
}

/** Reuse last run's cluster id (and colour, name) for groups that mostly persist. */
export const matchPrevious = (
  groups: Set<number>[],
  previous: Record<string, ClusterInfo>,
): [number, ClusterInfo | null][] => {
  const taken = new Set<string>();
  let nextId = Math.max(-1, ...Object.keys(previous).map(Number)) + 1;
  return groups.map((group) => {
    let bestKey: string | null = null;
    let bestScore = 0;
    for (const [key, info] of Object.entries(previous)) {
      if (taken.has(key)) continue;
      const members = new Set(info.members);
      const inter = [...group].filter((id) => members.has(id)).length;
      const score = inter / Math.max(1, group.size + members.size - inter);
      if (score > bestScore) [bestKey, bestScore] = [key, score];
    }
    if (bestKey !== null && bestScore >= MATCH_JACCARD) {
      taken.add(bestKey);
      return [Number(bestKey), previous[bestKey]];
    }
    return [nextId++, null];
  });
};

/** Python-style str.title() for ASCII tags ("vector-db" -> "Vector-Db"); CJK stays as is. */
const display = (tag: string) =>
  /^[\x00-\x7f]*$/.test(tag) ? tag.toLowerCase().replace(/(^|[^a-z])([a-z])/g, (_, p, c) => p + c.toUpperCase()) : tag;

/** Name each group by its most distinctive tags (frequency x inverse group frequency). */
export const tagNames = (groupsTags: string[][][], languages: string[][]): [string, string, string[]][] => {
  const counts = groupsTags.map((group) => {
    const c = new Map<string, number>();
    group.forEach((tags) => new Set(tags).forEach((t) => c.set(t, (c.get(t) ?? 0) + 1)));
    return c;
  });
  const spread = new Map<string, number>();
  counts.forEach((c) => c.forEach((_, t) => spread.set(t, (spread.get(t) ?? 0) + 1)));
  return groupsTags.map((group, g) => {
    const size = Math.max(1, group.length);
    const score = (t: string) => (counts[g].get(t)! / size) * Math.log(1 + groupsTags.length / spread.get(t)!);
    const keywords = [...counts[g].keys()].sort((a, b) => score(b) - score(a) || a.localeCompare(b)).slice(0, 5);
    const name = keywords.slice(0, 2).map(display).join(' / ') || `${size} repos`;
    const langCounts = new Map<string, number>();
    languages[g].filter(Boolean).forEach((l) => langCounts.set(l, (langCounts.get(l) ?? 0) + 1));
    const topLangs = [...langCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 2).map(([l]) => l);
    return [sanitizeClusterName(name), `${size} repositories${topLangs.length ? ` · ${topLangs.join(', ')}` : ''}`, keywords];
  });
};

export const sanitizeClusterName = (name: string | null | undefined): string => {
  if (!name) return '未分类';
  const cleaned = name
    .trim()
    .split(/\s+/)
    .join(' ')
    .replaceAll('工具集', '工具')
    .replaceAll('集合', '集')
    .replace(/\s+[&+/,，、]\s+/g, ' / ')
    .replace(/^[-_/ ]+|[-_/ ]+$/g, '');
  return cleaned || '未分类';
};

/** Duplicate names get a keyword suffix; the largest cluster keeps the plain name. */
export const deduplicateNames = <T extends { name: string; keywords: string[]; members: number[] }>(entries: T[]): T[] => {
  const used = new Set<string>();
  for (const entry of [...entries].sort((a, b) => b.members.length - a.members.length)) {
    if (!used.has(entry.name)) {
      used.add(entry.name);
      continue;
    }
    const keywords = entry.keywords.filter(Boolean);
    const suffix = (keywords.at(-1) ?? String(entry.members.length)).replaceAll(' ', '').slice(0, 12);
    let candidate = `${entry.name} · ${suffix}`;
    for (let seq = 2; used.has(candidate); seq += 1) candidate = `${entry.name} · ${suffix}${seq}`;
    entry.name = candidate;
    used.add(candidate);
  }
  return entries;
};
