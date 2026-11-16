/** Similarity scoring, graph edges, timeline, and the JSON the site reads. */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { TAG_SYNONYMS } from './synonyms.ts';
import type { Repo, State } from './sync.ts';

// Fixed weights (locked by product decision).
const SEMANTIC_WEIGHT = 0.65;
const TAG_OVERLAP_WEIGHT = 0.2;
const STAR_LIST_WEIGHT = 0.1;
const LANGUAGE_WEIGHT = 0.05;

const EDGE_K = 8;
/** Repos in no star list. Must match `UNCATEGORIZED_LIST_ID` in frontend/src/types.ts. */
const UNCATEGORIZED_LIST_ID = 'uncategorized';

export const CLUSTER_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD',
  '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8B500', '#00CED1',
];

export type Edge = [number, number, number];

// ---- tags ----------------------------------------------------------------

export const normalizeTag = (token: string): string => {
  const cleaned = token
    .trim()
    .toLowerCase()
    .replace(/_/g, '-')
    .replace(/[^a-z0-9\-\u4e00-\u9fff]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-+|-+$/g, '');
  return cleaned ? (TAG_SYNONYMS[cleaned] ?? cleaned) : '';
};

/** Merge tag sources into unique canonical tokens, keeping first-seen order. */
export const normalizeTags = (...sources: (string[] | null | undefined)[]): string[] => {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const source of sources) {
    for (const raw of source ?? []) {
      if (!raw) continue;
      const parts = /[,/|;，、 ]/.test(raw) ? raw.split(/[,\s/|;，、]+/).filter(Boolean) : [raw];
      for (const part of parts) {
        const token = normalizeTag(part);
        if (token && !seen.has(token)) {
          seen.add(token);
          out.push(token);
        }
      }
    }
  }
  return out;
};

/** Weighted Jaccard: tags are unique per repo, so shared ones weigh 1/2, the rest 1. */
export const tagOverlap = (left: string[], right: string[], leftSet: ReadonlySet<string> = new Set(left)): number => {
  if (!left.length || !right.length) return 0;
  let inter = 0;
  for (const tag of right) if (leftSet.has(tag)) inter += 1;
  if (!inter) return 0;
  const symDiff = left.length + right.length - 2 * inter;
  return inter / (inter + 2 * symDiff);
};

// ---- pairwise scores -------------------------------------------------------

export interface Scores {
  n: number;
  score: Float32Array;
}

/** One-slot buffer so a streamed score rounds the same way as a Float32Array store. */
const scoreBits = new Float32Array(1);
const f32 = (value: number) => {
  scoreBits[0] = value;
  return scoreBits[0];
};

interface Neighbour {
  j: number;
  w: number;
}

/**
 * Keep the K neighbours a stable descending sort would have kept. Equal scores
 * prefer the earlier index, matching a stable sort of 0..n-1.
 */
const consider = (row: Neighbour[], j: number, w: number) => {
  if (row.length < EDGE_K) {
    row.push({ j, w });
    return;
  }
  let worst = 0;
  for (let k = 1; k < row.length; k += 1) {
    const candidate = row[k];
    const held = row[worst];
    if (candidate.w < held.w || (candidate.w === held.w && candidate.j > held.j)) worst = k;
  }
  const held = row[worst];
  if (w > held.w || (w === held.w && j < held.j)) row[worst] = { j, w };
};

const prepare = (repos: Repo[]) => {
  const n = repos.length;
  const dim = repos[0]?.embedding?.length ?? 0;
  const unit = new Float32Array(n * dim);
  const mean = new Float32Array(dim);
  repos.forEach((repo, i) => {
    const vec = repo.embedding!;
    let sum = 0;
    for (const value of vec) sum += value * value;
    const norm = Math.sqrt(sum) || 1;
    for (let d = 0; d < dim; d += 1) mean[d] += (unit[i * dim + d] = vec[d] / norm) / n;
  });
  for (let i = 0; i < n; i += 1) {
    let norm = 0;
    for (let d = 0; d < dim; d += 1) norm += (unit[i * dim + d] -= mean[d]) ** 2;
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < dim; d += 1) unit[i * dim + d] /= norm;
  }
  const tags = repos.map((r) => normalizeTags(r.ai_tags, r.topics));
  const tagSets = tags.map((row) => new Set(row));
  const listSets = repos.map((r) => new Set(r.star_list_ids));
  const weight = (i: number, j: number) => {
    let dot = 0;
    for (let d = 0, a = i * dim, b = j * dim; d < dim; d += 1) dot += unit[a + d] * unit[b + d];
    let sharedList = 0;
    const lists = listSets[i];
    const right = repos[j].star_list_ids;
    if (lists.size && right.length) {
      for (const id of right) {
        if (lists.has(id)) {
          sharedList = 1;
          break;
        }
      }
    }
    const sameLanguage = Boolean(repos[i].language) && repos[i].language === repos[j].language;
    return f32(
      Math.min(
        1,
        SEMANTIC_WEIGHT * Math.max(0, dot) +
          TAG_OVERLAP_WEIGHT * tagOverlap(tags[i], tags[j], tagSets[i]) +
          STAR_LIST_WEIGHT * sharedList +
          LANGUAGE_WEIGHT * +sameLanguage,
      ),
    );
  };
  return { n, weight };
};

/**
 * Blend of embedding similarity, tag overlap, star list, and language for every
 * pair. Embeddings share a dominant direction (raw cosines bunch around
 * 0.5-0.65), so they are mean-centered first to separate neighbours from bystanders.
 */
export const scorePairs = (repos: Repo[]): Scores => {
  const { n, weight } = prepare(repos);
  const score = new Float32Array(n * n);
  for (let i = 0; i < n; i += 1) {
    score[i * n + i] = -1;
    for (let j = i + 1; j < n; j += 1) score[i * n + j] = score[j * n + i] = weight(i, j);
  }
  return { n, score };
};

const quantile = (values: number[], q: number): number => {
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (pos - lo);
};

/**
 * Link-score cutoff that yields roughly `EDGE_K` edges per node, sampled on 200
 * repos. Purely rank-based: centered scores have no fixed scale to anchor a floor.
 */
const thresholdFromUpper = (upper: number[], m: number): number => {
  if (m < 3) return 0.5;
  const degree = Math.max(1, Math.min(EDGE_K, m - 1));
  return quantile(upper, Math.max(0.05, Math.min(0.98, 1 - degree / (m - 1))));
};

export const adaptiveThreshold = ({ n, score }: Scores, sample = 200): number => {
  const m = Math.min(n, sample);
  const upper: number[] = [];
  for (let i = 0; i < m; i += 1) for (let j = i + 1; j < m; j += 1) upper.push(score[i * n + j]);
  return thresholdFromUpper(upper, m);
};

const commitRow = (ids: number[], i: number, row: Neighbour[], threshold: number, pairs: Map<string, Edge>) => {
  const ordered = row.slice().sort((a, b) => b.w - a.w || a.j - b.j);
  for (let rank = 0; rank < ordered.length; rank += 1) {
    const { j, w } = ordered[rank];
    if (w <= 0 || (w < threshold && rank > 0)) return;
    const [s, t] = ids[i] < ids[j] ? [ids[i], ids[j]] : [ids[j], ids[i]];
    pairs.set(`${s}:${t}`, [s, t, Math.round(w * 1e4) / 1e4]);
  }
};

const edgesFromRows = (ids: number[], rows: Neighbour[][], threshold: number): Edge[] => {
  const pairs = new Map<string, Edge>();
  rows.forEach((row, i) => commitRow(ids, i, row, threshold, pairs));
  return [...pairs.values()].sort((a, b) => b[2] - a[2]);
};

/**
 * Undirected per-node k-NN edges above the adaptive threshold. Every node also
 * keeps its best edge, so the force layout never flings a weakly related repo
 * off into space. Each row keeps only its top K, with the same tie break as a
 * stable sort of the full row.
 */
export const buildEdges = (ids: number[], scores: Scores): Edge[] => {
  if (ids.length < 2) return [];
  const rows: Neighbour[][] = Array.from({ length: scores.n }, () => []);
  for (let i = 0; i < scores.n; i += 1) {
    const base = i * scores.n;
    for (let j = 0; j < scores.n; j += 1) if (j !== i) consider(rows[i], j, scores.score[base + j]);
  }
  return edgesFromRows(ids, rows, adaptiveThreshold(scores));
};

/**
 * Same edges as `buildEdges(scorePairs(...))` without retaining the n×n matrix.
 * The sync path uses this; the matrix stays available for the threshold helper.
 */
const linkRepos = (repos: Repo[]): Edge[] => {
  const { n, weight } = prepare(repos);
  if (n < 2) return [];
  const rows: Neighbour[][] = Array.from({ length: n }, () => []);
  const m = Math.min(n, 200);
  const upper: number[] = [];
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const w = weight(i, j);
      if (i < m && j < m) upper.push(w);
      consider(rows[i], j, w);
      consider(rows[j], i, w);
    }
  }
  return edgesFromRows(
    repos.map((repo) => repo.id),
    rows,
    thresholdFromUpper(upper, m),
  );
};

const topCounts = (values: string[], limit: number) => {
  const counts = new Map<string, number>();
  values.forEach((v) => counts.set(v, (counts.get(v) ?? 0) + 1));
  return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, limit).map(([v]) => v);
};

export const buildTimeline = (repos: Repo[]) => {
  const months = new Map<string, Repo[]>();
  [...repos]
    .filter((r) => r.starred_at)
    .sort((a, b) => a.starred_at!.localeCompare(b.starred_at!))
    .forEach((r) => {
      const key = r.starred_at!.slice(0, 7);
      months.set(key, [...(months.get(key) ?? []), r]);
    });
  return [...months.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, items]) => ({
      date,
      count: items.length,
      repos: items.slice(0, 20).map((r) => r.full_name),
      top_languages: topCounts(items.flatMap((r) => (r.language ? [r.language] : [])), 3),
      top_topics: topCounts(items.flatMap((r) => r.topics ?? []), 5),
    }));
};

// ---- analysis + output ----------------------------------------------------

export interface Analysis {
  repos: Repo[];
  edges: Edge[];
}

/** Score every embedded repo pair once and link each repo to its closest neighbours. */
export const analyze = (state: State): Analysis => {
  const repos = Object.values(state.repos)
    .filter((r) => r.embedding?.length)
    .sort((a, b) => (b.starred_at ?? '').localeCompare(a.starred_at ?? ''));
  if (!repos.length) return { repos, edges: [] };
  return { repos, edges: linkRepos(repos) };
};

const NODE_FIELDS = [
  'id', 'full_name', 'name', 'owner', 'owner_avatar_url', 'description', 'language', 'html_url',
  'stargazers_count', 'ai_summary', 'ai_tags', 'topics', 'starred_at', 'last_commit_time',
] as const;

export const writeSite = (state: State, analysis: Analysis, outputDir: string) => {
  const { repos, edges } = analysis;
  const clusterOf = new Map<number, number>();
  for (const [cid, info] of Object.entries(state.clusters)) for (const m of info.members) clusterOf.set(m, Number(cid));

  const clusters = Object.entries(state.clusters)
    .map(([cid, info]) => ({
      id: Number(cid),
      name: info.name,
      description: info.description,
      keywords: info.keywords,
      color: CLUSTER_COLORS[Number(cid) % CLUSTER_COLORS.length],
      repo_count: repos.filter((r) => clusterOf.get(r.id) === Number(cid)).length,
    }))
    .filter((c) => c.repo_count > 0)
    .sort((a, b) => a.id - b.id);

  const starLists = state.star_lists.map((list) => ({
    ...list,
    repo_count: repos.filter((r) => r.star_list_ids.includes(list.id)).length,
  }));
  const uncategorized = repos.filter((r) => r.star_list_ids.length === 0).length;
  if (starLists.length && uncategorized) {
    starLists.push({ id: UNCATEGORIZED_LIST_ID, name: 'Uncategorized', description: null, repo_count: uncategorized });
  }

  const nodes = repos.map((repo) => {
    const node: Record<string, unknown> = Object.fromEntries(NODE_FIELDS.map((f) => [f, repo[f] ?? null]));
    node.cluster_id = clusterOf.get(repo.id) ?? null;
    node.star_list_ids = repo.star_list_ids.length
      ? repo.star_list_ids
      : starLists.length
        ? [UNCATEGORIZED_LIST_ID]
        : [];
    return node;
  });

  const now = new Date();
  const stamp = now.toISOString().replace(/\D/g, '').slice(0, 14);
  const graph = {
    version: `${stamp}-n${nodes.length}-e${edges.length}`,
    generated_at: now.toISOString(),
    total_repos: Object.keys(state.repos).length,
    nodes,
    edges,
    clusters,
    star_lists: starLists,
    timeline: buildTimeline(Object.values(state.repos)),
  };
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'graph.json'), JSON.stringify(graph));
  return { version: graph.version, nodes: nodes.length, edges: edges.length, clusters: clusters.length };
};
