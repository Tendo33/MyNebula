import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

import { deduplicateNames, detectCommunities, louvain, matchPrevious, sanitizeClusterName, tagNames } from './cluster.ts';
import { adaptiveThreshold, analyze, buildEdges, normalizeTag, normalizeTags, scorePairs, tagOverlap, writeSite, type Edge } from './graph.ts';
import { embeddingText, fallbackTags, parseSummaryJson, readmeSignal } from './remote.ts';
import { assignClusters, contentHash, loadSettings, loadState, markForEnrichment, saveState, type Repo, type State } from './sync.ts';

const repo = (id: number, patch: Partial<Repo> = {}): Repo => ({
  id,
  full_name: `o/r${id}`,
  owner: 'o',
  name: `r${id}`,
  description: null,
  language: null,
  topics: [],
  html_url: '',
  owner_avatar_url: null,
  stargazers_count: 100 - id,
  starred_at: null,
  last_commit_time: null,
  star_list_id: null,
  ...patch,
});

/** Three well-separated themes of ten repos each. */
const syntheticState = (): State => {
  let seed = 1;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 0.6;
  const repos: Record<string, Repo> = {};
  for (let i = 0; i < 30; i += 1) {
    const theme = Math.floor(i / 10);
    const embedding = Array.from({ length: 16 }, (_, d) => (d === theme ? 5 : 0) + rand());
    repos[String(i + 1)] = repo(i + 1, {
      language: ['Go', 'Rust', 'Python'][theme],
      topics: [`t${theme}`],
      starred_at: `2025-0${theme + 1}-01T00:00:00Z`,
      star_list_id: i < 5 ? 1 : null,
      embedding,
    });
  }
  return { version: 1, repos, clusters: {}, star_lists: [{ id: 1, name: 'Fav', description: null }] };
};

const settings = () => ({ ...loadSettings(), llmApiKey: '', maxClusters: 12, llmClusterNames: false });

describe('tags and scores', () => {
  it('normalizes tags with synonyms and splitting', () => {
    expect(normalizeTag(' AI_Agents ')).toBe('agent');
    expect(normalizeTag('检索增强生成')).toBe('rag');
    expect(normalizeTag('!!!')).toBe('');
    expect(normalizeTags(['RAG, agents'], ['rag', 'Vector DB'])).toEqual(['rag', 'agent', 'vector', 'db']);
  });

  it('weights rare shared tags like the reference formula', () => {
    expect(tagOverlap(['rag', 'agent', 'llm'], ['rag', 'vector'])).toBeCloseTo(0.5 / (0.5 + 3));
    expect(tagOverlap([], ['x'])).toBe(0);
  });

  it('keeps edges undirected and the threshold clamped', () => {
    const scores = scorePairs(Object.values(syntheticState().repos));
    const threshold = adaptiveThreshold(scores);
    expect(threshold).toBeGreaterThanOrEqual(0.35);
    expect(threshold).toBeLessThanOrEqual(0.85);
    const edges = buildEdges(Array.from({ length: 30 }, (_, i) => i + 1), scores);
    expect(edges.every(([s, t]) => s < t)).toBe(true);
    expect(new Set(edges.map(([s, t]) => `${s}:${t}`)).size).toBe(edges.length);
  });
});

describe('text helpers', () => {
  it('parses model JSON with fences or garbage', () => {
    expect(parseSummaryJson('```json\n{"summary": "Fast", "tags": ["a", "b"]}\n```')).toEqual(['Fast', ['a', 'b']]);
    expect(parseSummaryJson('noise "summary": "S" , "tags": ["x", "y"')[0]).toBe('S');
  });

  it('never returns empty fallback tags', () => {
    expect(fallbackTags(repo(1, { name: 'my-tool' }), 'en')).toEqual(['my', 'open-source']);
    expect(fallbackTags(repo(1, { language: 'Go' }), 'zh')).toEqual(['Go', '开源项目']);
  });

  it('prefers feature sections and AI fields', () => {
    expect(readmeSignal('# Title\nintro text\n## Features\nfast things\n## License\nMIT')).toBe('features: fast things');
    const text = embeddingText(repo(1, { full_name: 'o/r', language: 'Python', description: 'raw', ai_summary: 'smart', ai_tags: ['AI Agents'], topics: ['x'] }), null);
    expect(text).toBe('o/r\nLanguage: Python\nsmart\nTags: agent, AI Agents');
  });

  it('re-embeds without re-summarizing when only the embedding is missing', () => {
    const digest = contentHash({ description: 'd', topics: [] });
    const failedBatch = repo(1, { hash: digest, ai_summary: 'paid', ai_tags: ['x'] });
    expect(markForEnrichment(failedBatch, digest, false)).toBe(true);
    expect(failedBatch.ai_summary).toBe('paid');

    const changed = repo(2, { hash: 'old', ai_summary: 'stale', embedding: [1] });
    expect(markForEnrichment(changed, digest, false)).toBe(true);
    expect(changed.ai_summary).toBeUndefined();

    const fresh = repo(3, { hash: digest, ai_summary: 's', embedding: [1] });
    expect(markForEnrichment(fresh, digest, false)).toBe(false);
    expect(markForEnrichment(fresh, digest, true)).toBe(true);
    expect(fresh.embedding).toBeUndefined();
  });

  it('hashes topics order-independently', () => {
    expect(contentHash({ description: 'd', topics: ['a', 'b'] })).toBe(contentHash({ description: 'd', topics: ['b', 'a'] }));
  });
});

describe('clusters', () => {
  it('finds dense communities and folds stragglers', () => {
    const edges: Edge[] = [[1, 2, 0.9], [2, 3, 0.9], [1, 3, 0.9], [4, 5, 0.9], [5, 6, 0.9], [4, 6, 0.9], [1, 7, 0.5], [3, 4, 0.1]];
    const groups = detectCommunities([1, 2, 3, 4, 5, 6, 7, 8], edges, 8);
    expect(groups.map((g) => [...g].sort())).toEqual([[1, 2, 3, 7], [4, 5, 6]]);
  });

  it('lowers resolution to respect maxClusters', () => {
    const edges: Edge[] = [];
    for (let a = 0; a < 6; a += 1) {
      for (let i = 0; i < 3; i += 1) for (let j = 0; j < i; j += 1) edges.push([a * 10 + i, a * 10 + j, 0.9]);
      if (a < 5) edges.push([a * 10, (a + 1) * 10, 0.2]);
    }
    const ids = [...new Set(edges.flatMap(([s, t]) => [s, t]))];
    expect(detectCommunities(ids, edges, 8)).toHaveLength(6);
    expect(detectCommunities(ids, edges, 3).length).toBeLessThanOrEqual(3);
    expect(louvain([1, 2], [])).toEqual([[1], [2]]);
  });

  it('reuses previous ids by overlap', () => {
    const previous = { '4': { name: 'x', description: null, keywords: [], members: [1, 2, 3, 4], named_by: 'tags' as const } };
    const matched = matchPrevious([new Set([1, 2, 3]), new Set([20, 21, 22])], previous);
    expect(matched[0]).toEqual([4, previous['4']]);
    expect(matched[1]).toEqual([5, null]);
  });

  it('names groups by distinctive tags and de-duplicates', () => {
    const named = tagNames([[['llm', 'rag'], ['llm', 'rag', 'vector-db']], [['llm', 'cli'], ['cli', 'rust']]], [['Python', 'Python'], ['Rust', 'Go']]);
    expect(named[0][0]).toBe('Rag / Llm');
    expect(named[1][0].startsWith('Cli')).toBe(true);
    expect(sanitizeClusterName('  Dev 工具集 ')).toBe('Dev 工具');
    const entries = deduplicateNames([
      { name: 'Tools', keywords: ['cli'], members: [1, 2, 3] },
      { name: 'Tools', keywords: [], members: Array.from({ length: 9 }, (_, i) => i) },
    ]);
    expect(entries.map((e) => e.name)).toEqual(['Tools · cli', 'Tools']);
  });
});

describe('end to end', () => {
  it('writes a consistent site payload and keeps cluster ids across runs', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'nebula-'));
    const state = syntheticState();
    await assignClusters(state, analyze(state), settings(), { reset: false, llmNames: false });
    const meta = writeSite(state, analyze(state), dir);
    const graph = JSON.parse(readFileSync(join(dir, 'graph.json'), 'utf8'));
    const related = JSON.parse(readFileSync(join(dir, 'related.json'), 'utf8'));

    expect(meta).toMatchObject({ nodes: 30, clusters: 3 });
    const ids = new Set(graph.nodes.map((n: { id: number }) => n.id));
    expect(graph.edges.every(([s, t]: Edge) => ids.has(s) && ids.has(t))).toBe(true);
    const languagesByCluster = new Map<number, Set<string>>();
    for (const n of graph.nodes) languagesByCluster.set(n.cluster_id, (languagesByCluster.get(n.cluster_id) ?? new Set()).add(n.language));
    expect([...languagesByCluster.values()].every((s) => s.size === 1)).toBe(true);
    expect(graph.star_lists.map((l: { id: number }) => l.id)).toEqual([1, -1]);
    expect(graph.timeline.map((p: { date: string }) => p.date)).toEqual(['2025-01', '2025-02', '2025-03']);
    expect(related['1'].every((item: number[]) => item[0] !== 1 && item[2] >= 0.5)).toBe(true);

    const before = Object.keys(state.clusters).sort();
    state.repos['99'] = { ...state.repos['1'], id: 99, full_name: 'o/new' };
    await assignClusters(state, analyze(state), settings(), { reset: false, llmNames: false });
    expect(Object.keys(state.clusters).sort()).toEqual(before);
    expect(Object.values(state.clusters).some((c) => c.members.includes(99))).toBe(true);

    saveState(state, join(dir, 'cache.json.gz'));
    expect(loadState(join(dir, 'cache.json.gz')).repos['99'].full_name).toBe('o/new');
  });
});
