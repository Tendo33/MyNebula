import { expect, test, type Page } from '@playwright/test';

/**
 * Rendered contrast contract, light and dark.
 *
 * `paletteContrast.test.ts` checks the tokens in isolation; this checks what
 * actually lands on screen, which is where the real failures were. Before this
 * suite existed, forcing `html.dark` produced text at 1.14:1 across the app —
 * the ~275 `dark:` variants had never been rendered because nothing ever added
 * the class.
 *
 * Gated behind RUN_E2E=1 like the rest of the e2e directory.
 */
const runE2E = process.env.RUN_E2E === '1';

const META = { version: 'v1', generated_at: '2026-08-18T00:00:00Z', request_id: 'r1' };

const nodes = Array.from({ length: 12 }, (_, index) => {
  const i = index + 1;
  return {
    id: i,
    github_id: 900 + i,
    full_name: `octo/repo-${i}`,
    name: `repo-${i}`,
    description: '示例仓库描述。',
    language: ['TypeScript', 'Python', 'Rust'][i % 3],
    html_url: 'https://example.com',
    owner: 'octo',
    owner_avatar_url: null,
    x: (i % 5) * 4,
    y: Math.floor(i / 5) * 4,
    z: 0,
    cluster_id: (i % 3) + 1,
    color: ['#3b82f6', '#ef4444', '#10b981'][i % 3],
    size: 12,
    star_list_id: null,
    stargazers_count: 1200 * i,
    ai_summary: 'AI 摘要。',
    ai_tags: ['机器学习'],
    topics: ['graph'],
    starred_at: '2026-03-01T00:00:00Z',
    last_commit_time: '2026-03-02T00:00:00Z',
  };
});

const clusters = [1, 2, 3].map((id) => ({
  id,
  name: ['前端可视化', '机器学习基础设施', 'Rust 系统工具'][id - 1],
  description: '聚类描述',
  keywords: ['kw'],
  color: ['#3b82f6', '#ef4444', '#10b981'][id - 1],
  repo_count: 4,
  center_x: 0,
  center_y: 0,
  center_z: 0,
}));

const ROUTES: [string, unknown][] = [
  ['/api/v2/graph/edges', { edges: [], next_cursor: null, total_edges: 0, ...META }],
  ['/api/v2/graph/nodes', { nodes, next_cursor: null, ...META }],
  [
    '/api/v2/graph/timeline',
    {
      points: [
        { date: '2026-01', count: 12, repos: [], top_languages: ['TypeScript'], top_topics: ['graph'] },
      ],
      total_stars: 12,
      date_range: ['2026-01', '2026-03'],
      ...META,
    },
  ],
  [
    '/api/v2/graph',
    {
      nodes,
      edges: [],
      clusters,
      star_lists: [],
      total_nodes: 12,
      total_edges: 0,
      total_clusters: 3,
      total_star_lists: 0,
      ...META,
    },
  ],
  [
    '/api/v2/data/repos',
    {
      items: nodes,
      clusters: clusters.map((c) => ({
        id: c.id,
        name: c.name,
        color: c.color,
        repo_count: c.repo_count,
        keywords: c.keywords,
      })),
      count: 12,
      total_repos: 12,
      limit: 25,
      offset: 0,
      ...META,
    },
  ],
  [
    '/api/v2/dashboard',
    {
      summary: {
        total_repos: 12,
        embedded_repos: 12,
        total_topics: 8,
        total_clusters: 3,
        total_edges: 20,
      },
      top_languages: [{ language: 'TypeScript', count: 5 }],
      top_topics: [{ topic: 'graph', count: 7 }],
      top_clusters: clusters.map((c) => ({
        id: c.id,
        name: c.name,
        color: c.color,
        repo_count: c.repo_count,
        keywords: c.keywords,
      })),
      ...META,
    },
  ],
  ['/api/v2/auth/config', { enabled: true }],
  ['/api/v2/auth/me', { username: 'admin' }],
  [
    '/api/v2/settings',
    {
      schedule: {
        is_enabled: true,
        schedule_hour: 9,
        schedule_minute: 0,
        timezone: 'Asia/Shanghai',
        last_run_at: null,
        last_run_status: null,
        last_run_error: null,
      },
      sync_info: {
        total_repos: 12,
        synced_repos: 12,
        embedded_repos: 12,
        summarized_repos: 12,
        github_token_configured: true,
      },
      graph_defaults: { max_clusters: 8, min_clusters: 3 },
      ...META,
    },
  ],
];

const stubApi = async (page: Page) => {
  await page.route(
    (url) => url.pathname.startsWith('/api/v2/'),
    (route, request) => {
      const path = new URL(request.url()).pathname;
      const hit =
        ROUTES.find(([prefix]) => path === prefix) ??
        ROUTES.find(([prefix]) => path.startsWith(prefix));
      return route.fulfill({ json: hit ? hit[1] : {} });
    }
  );
};

/** WCAG 1.4.3 contrast of every rendered text node against its backdrop. */
const COLLECT_FAILURES = () => {
  const parse = (colour: string) => {
    const m = colour.match(/rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)/);
    return m ? { r: +m[1], g: +m[2], b: +m[3], a: m[4] === undefined ? 1 : +m[4] } : null;
  };
  const luminance = ({ r, g, b }: { r: number; g: number; b: number }) => {
    const f = (v: number) => {
      const c = v / 255;
      return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
  };
  const backdrop = (el: Element) => {
    let node: Element | null = el;
    while (node && node !== document.documentElement) {
      const c = parse(getComputedStyle(node).backgroundColor);
      if (c && c.a > 0.5) return c;
      node = node.parentElement;
    }
    return (
      parse(getComputedStyle(document.documentElement).backgroundColor) ?? {
        r: 255,
        g: 255,
        b: 255,
        a: 1,
      }
    );
  };

  const failures: { text: string; ratio: number; floor: number; colour: string }[] = [];
  for (const el of Array.from(document.querySelectorAll('body *'))) {
    if (el.children.length) continue;
    const text = el.textContent?.trim() ?? '';
    if (text.length < 2) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || Number(cs.opacity) < 0.2) continue;
    // Disabled controls are exempt under WCAG 1.4.3.
    if (el.closest('[disabled], .cursor-not-allowed')) continue;
    const rect = el.getBoundingClientRect();
    if (!rect.width || !rect.height) continue;
    const fg = parse(cs.color);
    if (!fg || fg.a < 0.5) continue;
    const bg = backdrop(el);
    const l1 = luminance(fg);
    const l2 = luminance(bg);
    const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
    const size = Number.parseFloat(cs.fontSize);
    const large = size >= 24 || (size >= 18.66 && Number(cs.fontWeight) >= 700);
    const floor = large ? 3 : 4.5;
    if (ratio < floor) {
      failures.push({ text: text.slice(0, 32), ratio: Number(ratio.toFixed(2)), floor, colour: cs.color });
    }
  }
  return failures;
};

const PAGES: [string, string][] = [
  ['dashboard', '/'],
  ['graph', '/graph'],
  ['data', '/data'],
  ['settings', '/settings'],
];

test.describe('rendered contrast', () => {
  test.skip(!runE2E, 'Set RUN_E2E=1 to execute browser contrast checks.');

  for (const theme of ['light', 'dark'] as const) {
    for (const [name, path] of PAGES) {
      test(`${name} meets WCAG AA in ${theme} mode`, async ({ page }) => {
        await stubApi(page);
        await page.addInitScript((mode) => {
          localStorage.setItem('nebula_theme', mode);
        }, theme);
        await page.goto(path);

        // Guard the guard: an unrendered page would report zero failures.
        await expect(page.locator('main')).toBeVisible();
        await expect
          .poll(async () => page.evaluate(() => document.querySelectorAll('body *').length))
          .toBeGreaterThan(40);
        await expect
          .poll(async () => page.evaluate(() => document.documentElement.classList.contains('dark')))
          .toBe(theme === 'dark');

        const failures = await page.evaluate(COLLECT_FAILURES);
        expect(
          failures,
          failures
            .map((f) => `"${f.text}" ${f.ratio}:1 (needs ${f.floor}) colour=${f.colour}`)
            .join('\n')
        ).toEqual([]);
      });
    }
  }
});
