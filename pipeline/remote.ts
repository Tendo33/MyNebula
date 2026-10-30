/** GitHub reads (stars, READMEs, star lists) and OpenAI-compatible LLM/embedding calls. */

import { CLUSTER_PROMPT, CLUSTER_SYSTEM, EXCLUSION, README_HEADINGS, SUMMARY_PROMPT, SUMMARY_SYSTEM, type Lang } from './prompts.ts';
import { normalizeTag, normalizeTags } from './graph.ts';
import { sanitizeClusterName } from './cluster.ts';
import type { Repo, Settings } from './sync.ts';

const log = (...args: unknown[]) => console.log(new Date().toISOString(), ...args);

/** Run `fn` over items with at most `limit` in flight. */
export const pool = async <T>(items: T[], limit: number, fn: (item: T) => Promise<void>) => {
  const queue = [...items];
  await Promise.all(Array.from({ length: Math.min(limit, queue.length) }, async () => {
    while (queue.length) await fn(queue.shift()!);
  }));
};

// ---- GitHub ----------------------------------------------------------------

export const github = (token: string) => {
  const headers: Record<string, string> = { Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28' };
  if (token) headers.Authorization = `Bearer ${token}`;
  const get = (path: string, accept?: string) =>
    fetch(`https://api.github.com${path}`, { headers: accept ? { ...headers, Accept: accept } : headers, signal: AbortSignal.timeout(30_000) });
  const graphql = async (query: string, variables: object) => {
    const response = await fetch('https://api.github.com/graphql', {
      method: 'POST',
      headers,
      body: JSON.stringify({ query, variables }),
      signal: AbortSignal.timeout(30_000),
    });
    if (!response.ok) throw new Error(`GraphQL HTTP ${response.status}`);
    const payload = await response.json();
    if (payload.errors) throw new Error(JSON.stringify(payload.errors));
    return payload.data ?? {};
  };

  return {
    /** Every starred repo, newest first, flattened to the fields the graph uses. */
    async starred(): Promise<Partial<Repo>[]> {
      const repos: Partial<Repo>[] = [];
      for (let page = 1; ; page += 1) {
        const response = await get(`/user/starred?per_page=100&page=${page}`, 'application/vnd.github.star+json');
        if (!response.ok) throw new Error(`GitHub /user/starred HTTP ${response.status}`);
        const items = await response.json();
        if (!items.length) break;
        for (const item of items) {
          const repo = item.repo ?? item;
          repos.push({
            id: repo.id,
            full_name: repo.full_name,
            owner: repo.owner.login,
            name: repo.name,
            description: repo.description,
            language: repo.language,
            topics: repo.topics ?? [],
            html_url: repo.html_url,
            owner_avatar_url: repo.owner.avatar_url,
            stargazers_count: repo.stargazers_count ?? 0,
            starred_at: item.starred_at ?? null,
            last_commit_time: repo.pushed_at ?? null,
          });
        }
        log(`Fetched starred page ${page} (${repos.length} repos)`);
      }
      return repos;
    },

    async readmes(names: string[], maxLength: number, concurrency: number): Promise<Map<string, string | null>> {
      const out = new Map<string, string | null>();
      await pool(names, concurrency, async (name) => {
        try {
          const response = await get(`/repos/${name}/readme`, 'application/vnd.github.raw+json');
          out.set(name, response.ok ? (await response.text()).slice(0, maxLength) || null : null);
        } catch {
          out.set(name, null);
        }
      });
      return out;
    },

    /** User-defined star lists with member repo ids; empty when unavailable. */
    async starLists(): Promise<{ name: string; description: string | null; repo_ids: number[] }[]> {
      try {
        const lists = [];
        for (let cursor: string | null = null; ; ) {
          const data = await graphql(
            'query($cursor: String) { viewer { lists(first: 100, after: $cursor) { pageInfo { hasNextPage endCursor } nodes { id name description } } } }',
            { cursor },
          );
          const payload = data.viewer?.lists ?? {};
          for (const node of payload.nodes ?? []) {
            if (!node) continue;
            const repoIds: number[] = [];
            for (let itemsCursor: string | null = null; ; ) {
              const items: any = (
                await graphql(
                  'query($listId: ID!, $cursor: String) { node(id: $listId) { ... on UserList { items(first: 100, after: $cursor) { pageInfo { hasNextPage endCursor } nodes { ... on Repository { databaseId } } } } } }',
                  { listId: node.id, cursor: itemsCursor },
                )
              ).node?.items ?? {};
              repoIds.push(...(items.nodes ?? []).flatMap((i: any) => (i?.databaseId ? [i.databaseId] : [])));
              if (!items.pageInfo?.hasNextPage) break;
              itemsCursor = items.pageInfo.endCursor;
            }
            lists.push({ name: node.name, description: node.description ?? null, repo_ids: repoIds });
          }
          if (!payload.pageInfo?.hasNextPage) break;
          cursor = payload.pageInfo.endCursor;
        }
        return lists;
      } catch (err) {
        log('Star lists unavailable:', String(err));
        return [];
      }
    },
  };
};

// ---- text building ---------------------------------------------------------

/** Compact README excerpt that favours feature/usage sections. */
export const readmeSignal = (readme: string, maxChars = 1200): string => {
  const lines = readme.replace(/\r\n/g, '\n').trim().split('\n').map((l) => l.trim());
  const sections: string[] = [];
  let current: string[] = [];
  let heading = '';
  const flush = () => {
    const body = current.filter(Boolean).join(' ');
    if (body) sections.push(heading ? `${heading}: ${body}` : body);
    current = [];
  };
  for (const line of lines) {
    if (!line) continue;
    if (line.startsWith('#')) {
      flush();
      heading = line.replace(/^#+/, '').trim().toLowerCase();
      continue;
    }
    current.push(line);
    if (current.join(' ').length > 400) flush();
  }
  flush();
  const selected = sections.filter((s) => README_HEADINGS.some((k) => s.toLowerCase().includes(k)));
  const result = sections.length ? (selected.length ? selected : sections.slice(0, 3)).join(' | ') : lines.slice(0, 40).join(' ');
  return result.slice(0, maxChars);
};

/** Canonical tag plus the original spelling, deduplicated case-insensitively. */
const semanticTags = (tags: string[]) => {
  const seen = new Set<string>();
  return tags.flatMap((raw) =>
    [normalizeTag(raw), raw.trim()].filter((c) => c && !seen.has(c.toLowerCase()) && seen.add(c.toLowerCase())),
  );
};

/** Text used for the repo embedding; LLM output wins over raw metadata. */
export const embeddingText = (repo: Repo, readme: string | null | undefined): string => {
  const parts = [repo.full_name];
  if (repo.language) parts.push(`Language: ${repo.language}`);
  if (repo.ai_summary || repo.description) parts.push((repo.ai_summary || repo.description)!);
  if (repo.ai_tags?.length) parts.push(`Tags: ${semanticTags(repo.ai_tags).join(', ')}`);
  else if (repo.topics?.length) parts.push(`Topics: ${semanticTags(repo.topics).join(', ')}`);
  const signal = readme ? readmeSignal(readme) : '';
  if (signal) parts.push(`Readme: ${signal}`);
  return parts.join('\n');
};

/** Parse `{"summary", "tags"}` from model output, tolerating fences and junk. */
export const parseSummaryJson = (text: string): [string, string[]] => {
  let body = text.trim();
  if (body.startsWith('```')) {
    const lines = body.split('\n');
    body = (lines.at(-1)!.trim() === '```' ? lines.slice(1, -1) : lines.slice(1)).join('\n');
  }
  let summary: string;
  let rawTags: unknown[];
  try {
    const data = JSON.parse(body.match(/\{[\s\S]*\}/)?.[0] ?? body);
    summary = String(data.summary ?? '').trim();
    rawTags = Array.isArray(data.tags) ? data.tags : [];
  } catch {
    summary = body.match(/"summary"\s*:\s*"([^"]+)"/)?.[1] ?? body.slice(0, 200);
    rawTags = body.match(/"tags"\s*:\s*\[([^\]]+)\]/)?.[1].split(',') ?? [];
  }
  const tags = rawTags.map((t) => String(t).trim().replace(/^["']+|["']+$/g, '')).filter((t) => t && t.length <= 20);
  return [summary, tags.slice(0, 7)];
};

export const fallbackTags = (repo: Repo, lang: Lang): string[] => {
  const generic = lang === 'zh' ? '开源项目' : 'open-source';
  const topics = (repo.topics ?? []).slice(0, 5).filter((t) => t.length <= 15);
  if (topics.length) return topics;
  if (repo.language) return [repo.language, generic];
  return [...repo.name.replace(/[-_]/g, ' ').split(/\s+/).filter(Boolean).slice(0, 1), generic];
};

const ranked = (values: string[]) => {
  const counts = new Map<string, number>();
  values.forEach((v) => counts.set(v, (counts.get(v) ?? 0) + 1));
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([v]) => v);
};

// ---- model calls -------------------------------------------------------------

export const ai = (settings: Settings) => {
  const lang = settings.outputLanguage;
  const post = async (baseUrl: string, apiKey: string, path: string, body: object) => {
    for (let attempt = 0, delay = 1000; ; attempt += 1, delay *= 2) {
      try {
        const response = await fetch(`${baseUrl.replace(/\/$/, '')}${path}`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(120_000),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`);
        return await response.json();
      } catch (err) {
        if (attempt === 3) throw err;
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  };
  const complete = async (system: string, prompt: string, maxTokens = 2048): Promise<string> => {
    const data = await post(settings.llmBaseUrl, settings.llmApiKey, '/chat/completions', {
      model: settings.llmModel,
      messages: [{ role: 'system', content: system }, { role: 'user', content: prompt }],
      max_tokens: maxTokens,
      temperature: 0.3,
    });
    return data.choices?.[0]?.message?.content ?? '';
  };

  return {
    /** Embed in provider-sized batches; each batch retries on its own. */
    async embed(texts: string[]): Promise<number[][]> {
      const out: number[][] = [];
      for (let i = 0; i < texts.length; i += settings.batchSize) {
        const batch = texts.slice(i, i + settings.batchSize).map((t) => (t.trim() ? t : ' '));
        const data = await post(settings.embeddingBaseUrl, settings.embeddingApiKey, '/embeddings', { model: settings.embeddingModel, input: batch });
        out.push(...[...data.data].sort((a: any, b: any) => a.index - b.index).map((d: any) => d.embedding));
      }
      return out;
    },

    /** Summary and tags for one repo. Tags are never empty. */
    async summarize(repo: Repo, readme: string | null | undefined): Promise<[string, string[]]> {
      const zh = lang === 'zh';
      const context = [zh ? `仓库名称: ${repo.full_name}` : `Repository: ${repo.full_name}`];
      if (repo.language) context.push((zh ? '主要语言: ' : 'Primary language: ') + repo.language);
      if (repo.description) context.push((zh ? '官方描述: ' : 'Description: ') + repo.description);
      if (repo.topics?.length) context.push((zh ? 'GitHub 标签: ' : 'GitHub topics: ') + repo.topics.slice(0, 10).join(', '));
      if (readme) context.push((zh ? 'README 全文:\n' : 'README full content:\n') + readme);
      let summary = '';
      let tags: string[] = [];
      try {
        [summary, tags] = parseSummaryJson(await complete(SUMMARY_SYSTEM[lang], SUMMARY_PROMPT[lang].replace('{context}', () => context.join('\n'))));
      } catch (err) {
        log(`Summary failed for ${repo.full_name}:`, String(err));
        summary = repo.description || (zh ? `${repo.name} - 开源项目` : `${repo.name} - open source project`);
      }
      return [summary, tags.length ? tags : fallbackTags(repo, lang)];
    },

    /** LLM name/description/keywords for one group; null if the call fails. */
    async nameCluster(repos: Repo[], existing: string[]): Promise<[string, string, string[]] | null> {
      const topics = repos.map((r) => normalizeTags(r.topics));
      const languages = repos.map((r) => r.language ?? '');
      try {
        const lines = repos.slice(0, 10).map((r, i) => `- ${r.full_name}: ${r.description ?? ''} [${languages[i]}] (${topics[i].slice(0, 5).join(', ')})`);
        const fill: Record<string, string> = {
          count: String(repos.length),
          repos: lines.join('\n'),
          languages: ranked(languages.filter(Boolean)).slice(0, 5).join(', '),
          topics: ranked(topics.flat()).slice(0, 10).join(', '),
          exclusion: existing.length ? EXCLUSION[lang].replace('{names}', () => existing.map((n) => `"${n}"`).join(', ')) : '',
        };
        const prompt = CLUSTER_PROMPT[lang].replace(/\{(count|repos|languages|topics|exclusion)\}/g, (_, key) => fill[key]);
        const text = (await complete(CLUSTER_SYSTEM[lang], prompt, 2560)).trim();
        const line = text.split('\n').find((l) => l.includes('|'))?.trim() ?? text.split('\n')[0];
        const parts = line.split('|');
        if (parts.length >= 3) {
          const keywords = parts[2].split(',').map((k) => k.trim()).filter(Boolean).slice(0, 5);
          return [sanitizeClusterName(parts[0]), parts[1].trim(), keywords];
        }
      } catch (err) {
        log('Cluster naming failed:', String(err));
      }
      return null;
    },
  };
};
