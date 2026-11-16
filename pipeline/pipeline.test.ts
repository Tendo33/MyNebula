import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';

import { deduplicateNames, detectCommunities, GROUP_ALGO, louvain, matchPrevious, sanitizeClusterName, tagNames } from './cluster.ts';
import { adaptiveThreshold, analyze, buildEdges, graphKey, normalizeTag, normalizeTags, scorePairs, tagOverlap, writeSite, type Edge } from './graph.ts';
import { embeddingText, fallbackTags, parseSummaryJson, pickReadmeOid, readmeSignal } from './remote.ts';
import { applyContentHash, assertAllowedApiHost, assignClusters, checkApiHosts, contentHash, legacyContentHash, loadSettings, loadState, markForEnrichment, saveState, shouldReuseClusters, type Repo, type State } from './sync.ts';

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
  star_list_ids: [],
  ...patch,
});

/**
 * Neighbours a full stable sort would keep. Independent of the pipeline's
 * top-k heap: score, rank, then the same cutoff (best edge always, then
 * anything at or above the sampled threshold, at most 8).
 */
const stableEdges = (repos: Repo[]): Edge[] => {
  const scores = scorePairs(repos);
  const { n, score } = scores;
  const threshold = adaptiveThreshold(scores);
  const ids = repos.map((item) => item.id);
  const pairs = new Map<string, Edge>();
  for (let i = 0; i < n; i += 1) {
    const ranked: { j: number; w: number }[] = [];
    for (let j = 0; j < n; j += 1) if (j !== i) ranked.push({ j, w: score[i * n + j] });
    ranked.sort((a, b) => b.w - a.w || a.j - b.j);
    for (let rank = 0; rank < Math.min(8, ranked.length); rank += 1) {
      const { j, w } = ranked[rank];
      if (w <= 0 || (w < threshold && rank > 0)) break;
      const [s, t] = ids[i] < ids[j] ? [ids[i], ids[j]] : [ids[j], ids[i]];
      pairs.set(`${s}:${t}`, [s, t, Math.round(w * 1e4) / 1e4]);
    }
  }
  return [...pairs.values()];
};

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
      star_list_ids: i < 5 ? ['fav'] : [],
      embedding,
    });
  }
  return { version: 1, repos, clusters: {}, star_lists: [{ id: 'fav', name: 'Fav', description: null }] };
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

  it('keeps edges undirected and every node attached', () => {
    const scores = scorePairs(Object.values(syntheticState().repos));
    expect(adaptiveThreshold(scores)).toBeGreaterThan(0);
    const edges = buildEdges(Array.from({ length: 30 }, (_, i) => i + 1), scores);
    expect(edges.every(([s, t]) => s < t)).toBe(true);
    expect(new Set(edges.map(([s, t]) => `${s}:${t}`)).size).toBe(edges.length);
    expect(new Set(edges.flatMap(([s, t]) => [s, t])).size).toBe(30);
  });

  it('streams the same edges as the full score matrix', () => {
    const state = syntheticState();
    const repos = Object.values(state.repos)
      .filter((repo) => repo.embedding?.length)
      .sort((a, b) => (b.starred_at ?? '').localeCompare(a.starred_at ?? ''));
    expect(analyze(state).edges).toEqual(buildEdges(repos.map((repo) => repo.id), scorePairs(repos)));
  });

  it('keeps the neighbours a full stable sort would keep', () => {
    const state = syntheticState();
    const repos = Object.values(state.repos).filter((item) => item.embedding?.length);
    const byStar = repos.slice().sort((a, b) => (b.starred_at ?? '').localeCompare(a.starred_at ?? ''));
    const key = (edges: Edge[]) => edges.map(([s, t, w]) => `${s}:${t}:${w}`).sort();
    expect(key(buildEdges(repos.map((item) => item.id), scorePairs(repos)))).toEqual(key(stableEdges(repos)));
    expect(key(analyze(state).edges)).toEqual(key(stableEdges(byStar)));
  });

  it('prefers the earlier repo when every score ties', () => {
    const repos = Array.from({ length: 10 }, (_, i) =>
      repo(i + 1, { embedding: [1, 0], language: 'Go', starred_at: '2024-01-01T00:00:00Z' }),
    );
    const scores = scorePairs(repos);
    const off: number[] = [];
    for (let i = 0; i < 10; i += 1) for (let j = i + 1; j < 10; j += 1) off.push(scores.score[i * 10 + j]);
    expect(new Set(off).size).toBe(1);
    expect(off[0]).toBeGreaterThan(0);
    const edges = buildEdges(repos.map((item) => item.id), scores);
    const key = (rows: Edge[]) => rows.map(([s, t, w]) => `${s}:${t}:${w}`).sort();
    expect(key(edges)).toEqual(key(stableEdges(repos)));
    expect(edges.some(([s, t]) => s === 9 && t === 10)).toBe(false);
    expect(edges.some(([s, t]) => s === 1 && t === 2)).toBe(true);
    const state: State = {
      version: 1,
      repos: Object.fromEntries(repos.map((item) => [String(item.id), item])),
      clusters: {},
      star_lists: [],
    };
    expect(key(analyze(state).edges)).toEqual(key(stableEdges(repos)));
  });

  it('reuses edges when only display fields change', () => {
    const state = syntheticState();
    const repos = Object.values(state.repos)
      .filter((item) => item.embedding?.length)
      .sort((a, b) => (b.starred_at ?? '').localeCompare(a.starred_at ?? ''));
    const first = analyze(state);
    expect(first.reused).toBe(false);
    expect(state.graph_key).toBe(graphKey(repos));
    state.repos['1'].stargazers_count = 1;
    state.repos['1'].ai_summary = 'rewritten blurb';
    const again = analyze(state);
    expect(again.reused).toBe(true);
    expect(again.edges).toEqual(first.edges);

    state.repos['1'].language = 'Zig';
    expect(analyze(state).reused).toBe(false);
    state.repos['1'].language = 'Go';
    expect(analyze(state).reused).toBe(false);
    expect(analyze(state).reused).toBe(true);
    state.repos['1'].star_list_ids = ['other'];
    expect(analyze(state).reused).toBe(false);
  });

  it('changes the scoring key when the cutoff sample changes order', () => {
    const state = syntheticState();
    analyze(state);
    const before = state.graph_key;
    const early = state.repos['1'];
    const late = state.repos['30'];
    [early.starred_at, late.starred_at] = [late.starred_at, early.starred_at];
    expect(analyze(state).reused).toBe(false);
    expect(state.graph_key).not.toBe(before);
  });

  it('keeps list ids and tags from sliding across the scoring key', () => {
    const state = syntheticState();
    const repo1 = state.repos['1'];
    repo1.star_list_ids = ['agent', 'llm'];
    repo1.topics = ['tool'];
    repo1.ai_tags = [];
    analyze(state);
    const split = state.graph_key;
    repo1.star_list_ids = ['llm', 'agent'];
    expect(analyze(state).reused).toBe(true);

    repo1.star_list_ids = ['agent'];
    repo1.topics = ['llm', 'tool'];
    expect(analyze(state).reused).toBe(false);
    expect(state.graph_key).not.toBe(split);
  });

  it('keeps the scoring key when tags normalize to the same tokens', () => {
    const state = syntheticState();
    state.repos['1'].topics = ['agent'];
    analyze(state);
    const before = state.graph_key;
    state.repos['1'].topics = ['AI_Agents'];
    expect(analyze(state).reused).toBe(true);
    expect(state.graph_key).toBe(before);
  });

  it('reuses groups only when the edges and the naming inputs still match', () => {
    const state = syntheticState();
    analyze(state);
    state.graph_max_clusters = 12;
    state.graph_group_algo = GROUP_ALGO;
    state.clusters = { '0': { name: 'A', description: null, keywords: [], members: [1], named_by: 'tags' } };
    expect(shouldReuseClusters(true, state, 12, false, false)).toBe(true);
    expect(shouldReuseClusters(false, state, 12, false, false)).toBe(false);
    expect(shouldReuseClusters(true, state, 8, false, false)).toBe(false);
    expect(shouldReuseClusters(true, state, 12, true, false)).toBe(false);
    expect(shouldReuseClusters(true, state, 12, false, true)).toBe(false);
    state.graph_group_algo = 0;
    expect(shouldReuseClusters(true, state, 12, false, false)).toBe(false);
    state.graph_group_algo = GROUP_ALGO;
    state.clusters['0'].named_by = 'llm';
    expect(shouldReuseClusters(true, state, 12, false, true)).toBe(true);
  });

  it('separates neighbours that share a dominant embedding direction', () => {
    // Every vector leans on one shared dimension, as real embeddings do; themes differ only slightly.
    const crowded = Object.values(syntheticState().repos).map((r) =>
      ({ ...r, language: null, topics: [], star_list_ids: [], embedding: r.embedding!.map((x, d) => (d === 15 ? 40 : x / 5)) }));
    const scores = scorePairs(crowded);
    const theme = (i: number) => Math.floor(i / 10);
    const edges = buildEdges(crowded.map((r) => r.id), scores);
    expect(edges.every(([s, t]) => theme(s - 1) === theme(t - 1))).toBe(true);
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
    expect(contentHash({ description: 'd', topics: [], readme_sha: 'abc' })).not.toBe(legacyContentHash({ description: 'd', topics: [] }));
  });

  it('records a README id without discarding the summary already paid for', () => {
    const legacy = legacyContentHash({ description: 'd', topics: [] });
    const kept = repo(1, { description: 'd', hash: legacy, ai_summary: 'paid', embedding: [1], readme_sha: 'abc' });
    expect(applyContentHash(kept, true, false)).toBe(false);
    expect(kept.ai_summary).toBe('paid');
    expect(kept.hash).toBe(contentHash(kept));

    kept.readme_sha = 'def';
    expect(applyContentHash(kept, true, false)).toBe(true);
    expect(kept.ai_summary).toBeUndefined();
  });

  it('leaves the old fingerprint alone when the README lookup does not run', () => {
    const legacy = legacyContentHash({ description: 'd', topics: [] });
    const kept = repo(1, { description: 'd', hash: legacy, ai_summary: 'paid', embedding: [1] });
    expect(applyContentHash(kept, false, false)).toBe(false);
    expect(kept.hash).toBe(legacy);
    expect(kept.ai_summary).toBe('paid');
  });

  it('adopts the README id after a description edit made while the lookup was down', () => {
    const kept = repo(1, { description: 'd', hash: legacyContentHash({ description: 'd', topics: [] }), ai_summary: 'paid', embedding: [1] });
    kept.description = 'e';
    expect(applyContentHash(kept, false, false)).toBe(true);
    Object.assign(kept, { ai_summary: 'fresh', embedding: [1] });

    kept.readme_sha = 'abc';
    expect(applyContentHash(kept, true, false)).toBe(false);
    expect(kept.ai_summary).toBe('fresh');
  });

  it('picks the README GitHub would serve', () => {
    expect(pickReadmeOid([
      { name: 'README', oid: 'plain', type: 'blob' },
      { name: 'README.md', oid: 'md', type: 'blob' },
      { name: 'docs', oid: 'dir', type: 'tree' },
    ])).toBe('md');
    expect(pickReadmeOid([{ name: 'lib', oid: 'x', type: 'blob' }])).toBeNull();
  });

  it('sends API keys only to hosts the workflow names', () => {
    const allow = ['api.siliconflow.cn', 'api.deepseek.com'];
    expect(() => assertAllowedApiHost('EMBEDDING_BASE_URL', 'https://api.siliconflow.cn/v1', allow)).not.toThrow();
    expect(() => assertAllowedApiHost('LLM_BASE_URL', 'https://api.deepseek.com', allow)).not.toThrow();
    expect(() => assertAllowedApiHost('EMBEDDING_BASE_URL', 'https://evil.example/v1', allow)).toThrow(/ALLOWED_API_HOSTS/);
    expect(() => assertAllowedApiHost('EMBEDDING_BASE_URL', 'http://api.siliconflow.cn/v1', allow)).toThrow(/https/);
    expect(() => assertAllowedApiHost('EMBEDDING_BASE_URL', 'https://user:k@api.siliconflow.cn/v1', allow)).toThrow(/credentials/);
    const settings = {
      embeddingApiKey: 'k',
      embeddingBaseUrl: 'http://localhost:11434/v1',
      llmApiKey: '',
      llmBaseUrl: 'http://localhost:11434/v1',
    };
    expect(() => checkApiHosts(settings, { ci: false, allowlist: [] })).not.toThrow();
    expect(() => checkApiHosts(settings, { ci: true, allowlist: [] })).toThrow(/ALLOWED_API_HOSTS/);
    expect(() => checkApiHosts(settings, { ci: true, allowlist: allow })).toThrow(/https/);
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

    expect(meta).toMatchObject({ nodes: 30, clusters: 3 });
    const ids = new Set(graph.nodes.map((n: { id: number }) => n.id));
    expect(graph.edges.every(([s, t]: Edge) => ids.has(s) && ids.has(t))).toBe(true);
    const languagesByCluster = new Map<number, Set<string>>();
    for (const n of graph.nodes) languagesByCluster.set(n.cluster_id, (languagesByCluster.get(n.cluster_id) ?? new Set()).add(n.language));
    expect([...languagesByCluster.values()].every((s) => s.size === 1)).toBe(true);
    expect(graph.star_lists.map((l: { id: string }) => l.id)).toEqual(['fav', 'uncategorized']);
    expect(graph.nodes.find((n: { id: number }) => n.id === 1).star_list_ids).toEqual(['fav']);
    expect(graph.nodes.find((n: { id: number }) => n.id === 6).star_list_ids).toEqual(['uncategorized']);
    expect(graph.timeline.map((p: { date: string }) => p.date)).toEqual(['2025-01', '2025-02', '2025-03']);

    const before = Object.keys(state.clusters).sort();
    state.repos['99'] = { ...state.repos['1'], id: 99, full_name: 'o/new' };
    await assignClusters(state, analyze(state), settings(), { reset: false, llmNames: false });
    expect(Object.keys(state.clusters).sort()).toEqual(before);
    expect(Object.values(state.clusters).some((c) => c.members.includes(99))).toBe(true);

    saveState(state, join(dir, 'cache.json.gz'));
    const loaded = loadState(join(dir, 'cache.json.gz'));
    expect(loaded.repos['99'].full_name).toBe('o/new');
    expect(loaded.graph_key).toBe(state.graph_key);
    expect(loaded.graph_edges).toEqual(state.graph_edges);
    expect(loaded.graph_max_clusters).toBe(12);
    expect(loaded.graph_group_algo).toBe(GROUP_ALGO);
  });

  it('keeps a repo on every star list and reads a legacy numeric cache', () => {
    const dir = mkdtempSync(join(tmpdir(), 'nebula-'));
    const state = syntheticState();
    state.star_lists.push({ id: 'beta', name: 'Beta', description: null });
    state.repos['1'].star_list_ids = ['fav', 'beta'];
    writeSite(state, analyze(state), dir);
    const written = JSON.parse(readFileSync(join(dir, 'graph.json'), 'utf8'));
    expect(written.nodes.find((n: { id: number }) => n.id === 1).star_list_ids).toEqual(['fav', 'beta']);
    expect(written.star_lists.find((list: { id: string }) => list.id === 'fav').repo_count).toBe(5);
    expect(written.star_lists.find((list: { id: string }) => list.id === 'beta').repo_count).toBe(1);

    const legacy = repo(1);
    delete (legacy as { star_list_ids?: string[] }).star_list_ids;
    (legacy as { star_list_id?: number }).star_list_id = 3;
    const cache = join(dir, 'legacy.json.gz');
    writeFileSync(cache, gzipSync(JSON.stringify({
      version: 1,
      repos: { '1': legacy },
      clusters: {},
      star_lists: [{ id: 3, name: 'Old', description: null }],
    })));
    const loaded = loadState(cache);
    expect(loaded.repos['1'].star_list_ids).toEqual(['3']);
    expect(loaded.star_lists[0].id).toBe('3');
    expect('star_list_id' in loaded.repos['1']).toBe(false);
  });
});
