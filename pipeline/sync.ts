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
  /** GitHub star-list ids. A repo may belong to several lists. */
  star_list_ids: string[];
  hash?: string;
  /** Blob id of the README GitHub would serve. Cache-only; absent until a lookup succeeds. */
  readme_sha?: string | null;
  ai_summary?: string;
  ai_tags?: string[];
  embedding?: number[];
}

export interface State {
  version: number;
  repos: Record<string, Repo>;
  clusters: Record<string, ClusterInfo>;
  star_lists: { id: string; name: string; description: string | null }[];
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
    embeddingModel: env('EMBEDDING_MODEL', 'BAAI/bge-m3'),
    llmApiKey: env('LLM_API_KEY'),
    llmBaseUrl: env('LLM_BASE_URL', 'https://api.siliconflow.cn/v1'),
    llmModel: env('LLM_MODEL', 'Qwen/Qwen3-30B-A3B-Instruct-2507'),
    outputLanguage: env('LLM_OUTPUT_LANGUAGE', 'zh') === 'en' ? 'en' : 'zh',
    batchSize: int('SYNC_BATCH_SIZE', 100),
    readmeMaxLength: int('SYNC_README_MAX_LENGTH', 10000),
    readmeConcurrency: 6,
    llmConcurrency: int('SYNC_LLM_CONCURRENCY', 4),
    maxClusters: int('GRAPH_MAX_CLUSTERS', 12),
    llmClusterNames: ['1', 'true', 'yes'].includes(env('GRAPH_LLM_NAMES').toLowerCase()),
    cachePath: env('NEBULA_CACHE', 'data/cache.json.gz'),
    outputDir: env('NEBULA_OUTPUT', 'frontend/public/data'),
  };
};

/** Older caches stored one numeric list id per repo, assigned by list order. */
const migrateLists = (state: State & { repos: Record<string, Repo & { star_list_id?: number | null }> }): State => {
  state.star_lists = state.star_lists.map((list) => ({ ...list, id: String(list.id) }));
  for (const repo of Object.values(state.repos)) {
    if (!Array.isArray(repo.star_list_ids)) {
      const legacy = repo.star_list_id;
      repo.star_list_ids = legacy != null && legacy !== -1 ? [String(legacy)] : [];
    }
    delete repo.star_list_id;
  }
  return state;
};

export const loadState = (path: string): State => {
  if (existsSync(path)) {
    const state = JSON.parse(gunzipSync(readFileSync(path)).toString('utf8'));
    if (state.version === CACHE_VERSION) return migrateLists(state);
    log(`Ignoring cache with unknown version: ${path}`);
  }
  return { version: CACHE_VERSION, repos: {}, clusters: {}, star_lists: [] };
};

export const saveState = (state: State, path: string) => {
  mkdirSync(dirname(path), { recursive: true });
  // A private temp name so two local syncs cannot overwrite one half-written file.
  const tmp = `${path}.${process.pid}.tmp`;
  writeFileSync(tmp, gzipSync(JSON.stringify(state)));
  renameSync(tmp, path);
};

const fingerprint = (repo: Pick<Repo, 'description' | 'topics'>, readmeSha: string) =>
  createHash('md5')
    .update(`${repo.description ?? ''}\x00${[...(repo.topics ?? [])].sort().join(',')}${readmeSha}`)
    .digest('hex');

/** Description + topics fingerprint already stored in caches from before README tracking. */
export const legacyContentHash = (repo: Pick<Repo, 'description' | 'topics'>) => fingerprint(repo, '');

/** Description + topics + README blob id. A README edit retriggers summarize and embed. */
export const contentHash = (repo: Pick<Repo, 'description' | 'topics' | 'readme_sha'>) =>
  fingerprint(repo, `\x00${repo.readme_sha ?? ''}`);

/**
 * Record the content fingerprint. The first successful README lookup adopts the
 * new fingerprint without discarding a summary that was already paid for.
 * A lookup that did not run leaves the previous fingerprint alone.
 */
export const applyContentHash = (repo: Repo, observed: boolean, full: boolean): boolean => {
  const legacy = legacyContentHash(repo);
  // Until one lookup succeeds, the old fingerprint stays, even across description edits.
  const tracked = observed || repo.readme_sha !== undefined;
  const digest = tracked ? contentHash(repo) : legacy;
  if (!full && observed && repo.hash === legacy && digest !== legacy) {
    repo.hash = digest;
    return !repo.embedding?.length;
  }
  return markForEnrichment(repo, digest, full);
};

const hostAllowlist = (raw: string | undefined): string[] =>
  (raw ?? '').split(',').map((item) => item.trim().toLowerCase()).filter(Boolean);

/** Refuse to send a key to a host the workflow did not name. Local runs stay open for Ollama. */
export const assertAllowedApiHost = (label: string, baseUrl: string, allowlist: string[]) => {
  let url: URL;
  try {
    url = new URL(baseUrl);
  } catch {
    throw new Error(`${label} is not a valid URL`);
  }
  if (url.protocol !== 'https:' || url.username || url.password) {
    throw new Error(`${label} must be https and must not contain credentials`);
  }
  const host = url.hostname.toLowerCase().replace(/\.$/, '');
  if (!allowlist.includes(host)) throw new Error(`${label} host "${host}" is not in ALLOWED_API_HOSTS`);
};

export const checkApiHosts = (
  settings: Pick<Settings, 'embeddingApiKey' | 'embeddingBaseUrl' | 'llmApiKey' | 'llmBaseUrl'>,
  opts: { ci: boolean; allowlist: string[] },
) => {
  if (!opts.ci && opts.allowlist.length === 0) return;
  if (opts.allowlist.length === 0) throw new Error('ALLOWED_API_HOSTS is empty; refusing to send API keys');
  if (settings.embeddingApiKey) assertAllowedApiHost('EMBEDDING_BASE_URL', settings.embeddingBaseUrl, opts.allowlist);
  if (settings.llmApiKey) assertAllowedApiHost('LLM_BASE_URL', settings.llmBaseUrl, opts.allowlist);
};

/**
 * Decide whether a repo needs enrichment, clearing only what is stale.
 * Changed content (or `--full`) drops the summary, tags, and embedding; a repo
 * that merely lost its embedding (a failed batch) keeps its paid summary.
 */
export const markForEnrichment = (repo: Repo, digest: string, full: boolean): boolean => {
  const changed = full || repo.hash !== digest;
  if (changed) {
    delete repo.ai_summary;
    delete repo.ai_tags;
    delete repo.embedding;
  }
  repo.hash = digest;
  return changed || !repo.embedding?.length;
};

/** Refresh metadata from GitHub; returns keys of repos that need enrichment. */
const fetchStars = async (state: State, settings: Settings, full: boolean): Promise<string[]> => {
  if (!settings.githubToken) throw new Error('GITHUB_TOKEN is not set');
  const gh = github(settings.githubToken);
  const [starred, lists] = [await gh.starred(), await gh.starLists()];

  const previous = state.repos;
  const repos: Record<string, Repo> = {};
  const dirty: string[] = [];
  const names = starred.flatMap((meta) => (meta.full_name?.includes('/') ? [meta.full_name] : []));
  let identities: Map<string, string | null> | null = null;
  try {
    identities = await gh.readmeShas(names);
  } catch (err) {
    log('README identities unavailable, hashing description and topics only:', String(err));
  }
  for (const meta of starred) {
    const key = String(meta.id);
    const prior = previous[key];
    const repo = { ...prior, ...meta, star_list_ids: prior?.star_list_ids ?? [] } as Repo;
    const observed = identities?.has(repo.full_name) ?? false;
    if (observed) repo.readme_sha = identities?.get(repo.full_name) ?? null;
    if (applyContentHash(repo, observed, full)) dirty.push(key);
    repos[key] = repo;
  }
  const removed = Object.keys(previous).filter((k) => !(k in repos)).length;
  state.repos = repos;

  // A failed lists request must not wipe memberships; repos keep their previous ones.
  if (lists) {
    const memberOf = new Map<number, string[]>();
    state.star_lists = lists.map((list) => {
      list.repo_ids.forEach((id) => {
        const membership = memberOf.get(id) ?? [];
        if (!membership.includes(list.id)) membership.push(list.id);
        memberOf.set(id, membership);
      });
      return { id: list.id, name: list.name, description: list.description };
    });
    for (const repo of Object.values(repos)) repo.star_list_ids = memberOf.get(repo.id) ?? [];
  }

  log(`Stars: ${starred.length} total, ${dirty.length} need enrichment, ${removed} removed, ${state.star_lists.length} star lists`);
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
      const unsummarized = batch.filter((repo) => !repo.ai_summary);
      await pool(unsummarized, settings.llmConcurrency, async (repo) => {
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
    // Some runners forward a literal `--`; drop it so flags after it still parse.
    args: process.argv.slice(2).filter((arg) => arg !== '--'),
    allowPositionals: true,
    options: {
      full: { type: 'boolean', default: false },
      recluster: { type: 'boolean', default: false },
      'llm-names': { type: 'boolean', default: false },
      'max-clusters': { type: 'string' },
    },
  });
  const command = positionals[0];
  if ((command !== 'sync' && command !== 'export') || positionals.length > 1) {
    console.error('usage: node pipeline/sync.ts <sync|export> [--full] [--recluster] [--llm-names] [--max-clusters N]');
    process.exit(2);
  }
  const settings = loadSettings();
  checkApiHosts(settings, { ci: process.env.GITHUB_ACTIONS === 'true', allowlist: hostAllowlist(process.env.ALLOWED_API_HOSTS) });
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
