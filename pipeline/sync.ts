/**
 * MyNebula pipeline.
 *
 *   node pipeline/sync.ts sync     fetch stars, enrich, link, group, write site data
 *   node pipeline/sync.ts export   rebuild links, groups, and site data from the cache
 *
 * State lives in a gzip JSON cache, so reruns only pay for new or changed repos.
 */

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { parseArgs } from 'node:util';
import { gunzipSync, gzipSync } from 'node:zlib';

import { deduplicateNames, detectCommunities, matchPrevious, tagNames, type ClusterInfo } from './cluster.ts';
import { analyze, normalizeTags, writeSite, type Analysis } from './graph.ts';
import type { Lang } from './prompts.ts';
import { ai, embeddingText, github, pool } from './remote.ts';

const CACHE_VERSION = 1;
const log = (...args: unknown[]) => console.log(new Date().toISOString(), ...args);

export interface Repo {
  id: number;
  full_name: string;
  owner: string;
  name: string;
  description: string | null;
  language: string | null;
  topics: string[];
  html_url: string;
  owner_avatar_url: string | null;
  stargazers_count: number;
  starred_at: string | null;
  last_commit_time: string | null;
  star_list_id: number | null;
  hash?: string;
  ai_summary?: string;
  ai_tags?: string[];
  embedding?: number[];
}

export interface State {
  version: number;
  repos: Record<string, Repo>;
  clusters: Record<string, ClusterInfo>;
  star_lists: { id: number; name: string; description: string | null }[];
}

export interface Settings {
  githubToken: string;
  embeddingApiKey: string;
  embeddingBaseUrl: string;
  embeddingModel: string;
  llmApiKey: string;
  llmBaseUrl: string;
  llmModel: string;
  outputLanguage: Lang;
  batchSize: number;
  readmeMaxLength: number;
  readmeConcurrency: number;
  llmConcurrency: number;
  maxClusters: number;
  llmClusterNames: boolean;
  cachePath: string;
  outputDir: string;
}

export const loadSettings = (): Settings => {
  if (existsSync('.env')) process.loadEnvFile('.env');
  const env = (name: string, fallback = '') => process.env[name]?.trim() || fallback;
  const int = (name: string, fallback: number) => Number.parseInt(env(name), 10) || fallback;
  return {
    githubToken: env('GITHUB_TOKEN'),
    embeddingApiKey: env('EMBEDDING_API_KEY'),
    embeddingBaseUrl: env('EMBEDDING_BASE_URL', 'https://api.siliconflow.cn/v1'),
    embeddingModel: env('EMBEDDING_MODEL', 'BAAI/bge-large-zh-v1.5'),
    llmApiKey: env('LLM_API_KEY'),
    llmBaseUrl: env('LLM_BASE_URL', 'https://api.siliconflow.cn/v1'),
    llmModel: env('LLM_MODEL', 'Qwen/Qwen2.5-7B-Instruct'),
    outputLanguage: env('LLM_OUTPUT_LANGUAGE', 'zh') === 'en' ? 'en' : 'zh',
    batchSize: int('SYNC_BATCH_SIZE', 100),
    readmeMaxLength: int('SYNC_README_MAX_LENGTH', 10000),
    readmeConcurrency: int('SYNC_README_FETCH_CONCURRENCY', 6),
    llmConcurrency: int('SYNC_LLM_ENHANCEMENT_CONCURRENCY', 4),
    maxClusters: int('GRAPH_MAX_CLUSTERS', 12),
    llmClusterNames: ['1', 'true', 'yes'].includes(env('GRAPH_LLM_NAMES').toLowerCase()),
    cachePath: env('NEBULA_CACHE', 'data/cache.json.gz'),
    outputDir: env('NEBULA_OUTPUT', 'frontend/public/data'),
  };
};

export const loadState = (path: string): State => {
  if (existsSync(path)) {
    const state = JSON.parse(gunzipSync(readFileSync(path)).toString('utf8'));
    if (state.version === CACHE_VERSION) return state;
    log(`Ignoring cache with unknown version: ${path}`);
  }
  return { version: CACHE_VERSION, repos: {}, clusters: {}, star_lists: [] };
};

export const saveState = (state: State, path: string) => {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(`${path}.tmp`, gzipSync(JSON.stringify(state)));
  renameSync(`${path}.tmp`, path);
};

/** Description + topics fingerprint; a change triggers re-summarize and re-embed. */
export const contentHash = (repo: Pick<Repo, 'description' | 'topics'>) =>
  createHash('md5')
    .update(`${repo.description ?? ''}\x00${[...(repo.topics ?? [])].sort().join(',')}`)
    .digest('hex');

/** Refresh metadata from GitHub; returns keys of repos that need enrichment. */
const fetchStars = async (state: State, settings: Settings, full: boolean): Promise<string[]> => {
  if (!settings.githubToken) throw new Error('GITHUB_TOKEN is not set');
  const gh = github(settings.githubToken);
  const [starred, lists] = [await gh.starred(), await gh.starLists()];

  const previous = state.repos;
  const repos: Record<string, Repo> = {};
  const dirty: string[] = [];
  for (const meta of starred) {
    const key = String(meta.id);
    const repo = { ...previous[key], ...meta } as Repo;
    const digest = contentHash(repo);
    if (full || repo.hash !== digest || !repo.embedding?.length) {
      delete repo.ai_summary;
      delete repo.ai_tags;
      delete repo.embedding;
      dirty.push(key);
    }
    repo.hash = digest;
    repos[key] = repo;
  }
  const removed = Object.keys(previous).filter((k) => !(k in repos)).length;
  state.repos = repos;

  const memberOf = new Map<number, number>();
  state.star_lists = lists.map((list, i) => {
    list.repo_ids.forEach((id) => memberOf.set(id, i + 1));
    return { id: i + 1, name: list.name, description: list.description };
  });
  for (const repo of Object.values(repos)) repo.star_list_id = memberOf.get(repo.id) ?? null;

  log(`Stars: ${starred.length} total, ${dirty.length} need enrichment, ${removed} removed, ${lists.length} star lists`);
  return dirty;
};

/** README -> LLM summary/tags (if configured) -> embedding, saved per batch so a crash resumes. */
const enrich = async (state: State, settings: Settings, keys: string[]): Promise<number> => {
  if (!keys.length) return 0;
  if (!settings.embeddingApiKey) throw new Error('EMBEDDING_API_KEY is not set');
  const gh = github(settings.githubToken);
  const model = ai(settings);
  let failed = 0;
  for (let start = 0; start < keys.length; start += settings.batchSize) {
    const batch = keys.slice(start, start + settings.batchSize).map((k) => state.repos[k]);
    const readmes = await gh.readmes(
      batch.map((r) => r.full_name),
      settings.readmeMaxLength,
      settings.readmeConcurrency,
    );
    if (settings.llmApiKey) {
      await pool(batch, settings.llmConcurrency, async (repo) => {
        [repo.ai_summary, repo.ai_tags] = await model.summarize(repo, readmes.get(repo.full_name));
      });
    }
    try {
      const vectors = await model.embed(batch.map((r) => embeddingText(r, readmes.get(r.full_name))));
      batch.forEach((repo, i) => (repo.embedding = vectors[i].map((v) => Math.round(v * 1e6) / 1e6)));
    } catch (err) {
      failed += batch.length;
      log(`Embedding batch failed (${batch.length} repos):`, String(err));
    }
    saveState(state, settings.cachePath);
    log(`Enriched ${Math.min(start + batch.length, keys.length)}/${keys.length}`);
  }
  if (failed === keys.length) throw new Error('Every embedding batch failed; check EMBEDDING_* settings');
  return failed;
};

/** Group repos by graph community; keep ids, colours, and LLM names stable across runs. */
export const assignClusters = async (
  state: State,
  analysis: Analysis,
  settings: Settings,
  { reset, llmNames }: { reset: boolean; llmNames: boolean },
) => {
  const groups = detectCommunities(
    analysis.repos.map((r) => r.id),
    analysis.edges,
    settings.maxClusters,
  );
  const matched = matchPrevious(groups, reset ? {} : state.clusters);
  const byId = new Map(analysis.repos.map((r) => [r.id, r]));
  const membersOf = groups.map((g) =>
    [...g].map((id) => byId.get(id)!).sort((a, b) => b.stargazers_count - a.stargazers_count || a.full_name.localeCompare(b.full_name)),
  );
  const fallback = tagNames(
    membersOf.map((m) => m.map((r) => normalizeTags(r.ai_tags, r.topics))),
    membersOf.map((m) => m.map((r) => r.language ?? '')),
  );
  const model = llmNames && settings.llmApiKey ? ai(settings) : null;

  const entries: (ClusterInfo & { id: number })[] = [];
  for (const [i, [id, prior]] of matched.entries()) {
    let named: [string, string | null, string[]] | null = null;
    if (model) {
      named = prior?.named_by === 'llm' ? [prior.name, prior.description, prior.keywords] : await model.nameCluster(membersOf[i], entries.map((e) => e.name));
    }
    const [name, description, keywords] = named ?? fallback[i];
    entries.push({
      id,
      name,
      description,
      keywords: keywords ?? [],
      members: membersOf[i].map((r) => r.id).sort((a, b) => a - b),
      named_by: named ? 'llm' : 'tags',
    });
  }
  state.clusters = Object.fromEntries(deduplicateNames(entries).map(({ id, ...info }) => [String(id), info]));
  log(`Grouped ${analysis.repos.length} repos into ${entries.length} clusters (${matched.filter(([, p]) => p).length} reused ids)`);
};

const main = async () => {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      full: { type: 'boolean', default: false },
      recluster: { type: 'boolean', default: false },
      'llm-names': { type: 'boolean', default: false },
      'max-clusters': { type: 'string' },
    },
  });
  const command = positionals[0];
  if (command !== 'sync' && command !== 'export') {
    console.error('usage: node pipeline/sync.ts <sync|export> [--full] [--recluster] [--llm-names] [--max-clusters N]');
    process.exit(2);
  }
  const settings = loadSettings();
  if (values['max-clusters']) settings.maxClusters = Number(values['max-clusters']);
  const state = loadState(settings.cachePath);

  let failed = 0;
  if (command === 'sync') {
    const dirty = await fetchStars(state, settings, values.full);
    saveState(state, settings.cachePath);
    failed = await enrich(state, settings, dirty);
  }
  const analysis = analyze(state);
  await assignClusters(state, analysis, settings, { reset: values.recluster, llmNames: values['llm-names'] || settings.llmClusterNames });
  saveState(state, settings.cachePath);
  log('Exported', writeSite(state, analysis, settings.outputDir), 'to', settings.outputDir);
  if (failed) log(`${failed} repos failed to embed and will be retried next run`);
};

if (import.meta.main) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
