/** Similarity scoring, graph edges, related repos, timeline, and the JSON the site reads. */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { TAG_SYNONYMS } from './prompts.ts';
import type { Repo, State } from './sync.ts';

// Fixed weights (locked by product decision).
const SEMANTIC_WEIGHT = 0.65;
const TAG_OVERLAP_WEIGHT = 0.2;
const STAR_LIST_WEIGHT = 0.1;
const LANGUAGE_WEIGHT = 0.05;

const EDGE_K = 8;
/** Floor for the single "best neighbour" edge that keeps a node attached. */
const MIN_ANCHOR_SCORE = 0.3;
const RELATED_LIMIT = 30;
const RELATED_MIN_SCORE = 0.4;
/** Lowest value of the "related strictness" slider; the client filters upward. */
const RELATED_MIN_SEMANTIC = 0.5;
const UNCATEGORIZED_LIST_ID = -1;

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
export const tagOverlap = (left: string[], right: string[]): number => {
  if (!left.length || !right.length) return 0;
  const set = new Set(left);
  const inter = right.filter((t) => set.has(t)).length;
  if (!inter) return 0;
  const symDiff = left.length + right.length - 2 * inter;
  return inter / (inter + 2 * symDiff);
};

// ---- pairwise scores -------------------------------------------------------

export interface Scores {
  n: number;
  score: Float32Array;
  semantic: Float32Array;
  overlap: Float32Array;
  sameList: (i: number, j: number) => boolean;
  sameLanguage: (i: number, j: number) => boolean;
}

export const scorePairs = (repos: Repo[]): Scores => {
  const n = repos.length;
  const dim = repos[0]?.embedding?.length ?? 0;
  const unit = new Float32Array(n * dim);
  repos.forEach((repo, i) => {
    const vec = repo.embedding!;
    const norm = Math.hypot(...vec) || 1;
    for (let d = 0; d < dim; d += 1) unit[i * dim + d] = vec[d] / norm;
  });
  const tags = repos.map((r) => normalizeTags(r.ai_tags, r.topics));
  const sameList = (i: number, j: number) => repos[i].star_list_id != null && repos[i].star_list_id === repos[j].star_list_id;
  const sameLanguage = (i: number, j: number) => Boolean(repos[i].language) && repos[i].language === repos[j].language;

  const score = new Float32Array(n * n);
  const semantic = new Float32Array(n * n);
  const overlap = new Float32Array(n * n);
  for (let i = 0; i < n; i += 1) {
    score[i * n + i] = -1;
    semantic[i * n + i] = 1;
    for (let j = i + 1; j < n; j += 1) {
      let dot = 0;
      for (let d = 0, a = i * dim, b = j * dim; d < dim; d += 1) dot += unit[a + d] * unit[b + d];
      const sem = Math.min(1, Math.max(0, dot));
      const tag = tagOverlap(tags[i], tags[j]);
      const s = Math.min(
        1,
        SEMANTIC_WEIGHT * sem + TAG_OVERLAP_WEIGHT * tag + STAR_LIST_WEIGHT * +sameList(i, j) + LANGUAGE_WEIGHT * +sameLanguage(i, j),
      );
      semantic[i * n + j] = semantic[j * n + i] = sem;
      overlap[i * n + j] = overlap[j * n + i] = tag;
      score[i * n + j] = score[j * n + i] = s;
    }
  }
  return { n, score, semantic, overlap, sameList, sameLanguage };
};

const quantile = (values: number[], q: number): number => {
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (pos - lo);
};

/** Score cutoff that yields roughly `EDGE_K` edges per node, sampled on 200 repos. */
export const adaptiveThreshold = ({ n, score }: Scores, sample = 200): number => {
  const m = Math.min(n, sample);
  if (m < 3) return 0.5;
  const upper: number[] = [];
  for (let i = 0; i < m; i += 1) for (let j = i + 1; j < m; j += 1) upper.push(score[i * n + j]);
  const degree = Math.max(1, Math.min(EDGE_K, m - 1));
  const q = Math.max(0.05, Math.min(0.98, 1 - degree / (m - 1)));
  return Math.max(0.35, Math.min(0.85, quantile(upper, q)));
};

const rankRow = (scores: Scores, i: number) =>
  Array.from({ length: scores.n }, (_, j) => j)
    .filter((j) => j !== i)
    .sort((a, b) => scores.score[i * scores.n + b] - scores.score[i * scores.n + a]);

/**
 * Undirected per-node k-NN edges above the adaptive threshold. Every node also
 * keeps its best edge (if it clears MIN_ANCHOR_SCORE), so the force layout never
 * flings a weakly related repo off into space.
 */
export const buildEdges = (ids: number[], scores: Scores): Edge[] => {
  if (ids.length < 2) return [];
  const threshold = adaptiveThreshold(scores);
  const pairs = new Map<string, Edge>();
  for (let i = 0; i < scores.n; i += 1) {
    rankRow(scores, i)
      .slice(0, EDGE_K)
      .every((j, rank) => {
        const w = scores.score[i * scores.n + j];
        if (w < threshold && !(rank === 0 && w >= MIN_ANCHOR_SCORE)) return false;
        const [s, t] = ids[i] < ids[j] ? [ids[i], ids[j]] : [ids[j], ids[i]];
        pairs.set(`${s}:${t}`, [s, t, Math.round(w * 1e4) / 1e4]);
        return true;
      });
  }
  return [...pairs.values()].sort((a, b) => b[2] - a[2]);
};

/** Per anchor: [id, score, semantic, tag_overlap, flags(bit0 same list, bit1 same language)]. */
export const buildRelated = (repos: Repo[], scores: Scores): Record<string, number[][]> => {
  const { n, score, semantic, overlap } = scores;
  const round = (v: number) => Math.round(v * 1e4) / 1e4;
  const related: Record<string, number[][]> = {};
  repos.forEach((repo, i) => {
    related[String(repo.id)] = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i && semantic[i * n + j] >= RELATED_MIN_SEMANTIC && score[i * n + j] >= RELATED_MIN_SCORE)
      .sort(
        (a, b) =>
          score[i * n + b] - score[i * n + a] ||
          semantic[i * n + b] - semantic[i * n + a] ||
          repos[b].stargazers_count - repos[a].stargazers_count,
      )
      .slice(0, RELATED_LIMIT)
      .map((j) => [
        repos[j].id,
        round(score[i * n + j]),
        round(semantic[i * n + j]),
        round(overlap[i * n + j]),
        +scores.sameList(i, j) | (+scores.sameLanguage(i, j) << 1),
      ]);
  });
  return related;
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
  related: Record<string, number[][]>;
}

/** Score every embedded repo pair once; derive edges and related repos from it. */
export const analyze = (state: State): Analysis => {
  const repos = Object.values(state.repos)
    .filter((r) => r.embedding?.length)
    .sort((a, b) => (b.starred_at ?? '').localeCompare(a.starred_at ?? ''));
  if (!repos.length) return { repos, edges: [], related: {} };
  const scores = scorePairs(repos);
  return {
    repos,
    edges: buildEdges(
      repos.map((r) => r.id),
      scores,
    ),
    related: buildRelated(repos, scores),
  };
};

const NODE_FIELDS = [
  'id', 'full_name', 'name', 'owner', 'owner_avatar_url', 'description', 'language', 'html_url',
  'stargazers_count', 'ai_summary', 'ai_tags', 'topics', 'starred_at', 'last_commit_time', 'star_list_id',
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
    repo_count: repos.filter((r) => r.star_list_id === list.id).length,
  }));
  const uncategorized = repos.filter((r) => r.star_list_id == null).length;
  if (starLists.length && uncategorized) {
    starLists.push({ id: UNCATEGORIZED_LIST_ID, name: 'Uncategorized', description: null, repo_count: uncategorized });
  }

  const nodes = repos.map((repo) => {
    const node: Record<string, unknown> = Object.fromEntries(NODE_FIELDS.map((f) => [f, repo[f] ?? null]));
    node.cluster_id = clusterOf.get(repo.id) ?? null;
    if (starLists.length && node.star_list_id == null) node.star_list_id = UNCATEGORIZED_LIST_ID;
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
  writeFileSync(join(outputDir, 'related.json'), JSON.stringify(analysis.related));
  return { version: graph.version, nodes: nodes.length, edges: edges.length, clusters: clusters.length };
};
