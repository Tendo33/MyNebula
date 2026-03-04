import { test, expect, type Page } from '@playwright/test';

const runE2E = process.env.RUN_E2E === '1';

const graphPayload = {
  nodes: [
    {
      id: 1,
      github_id: 101,
      full_name: 'demo-org/nebula-ui',
      name: 'nebula-ui',
      description: 'UI demo repo',
      language: 'TypeScript',
      html_url: 'https://github.com/demo-org/nebula-ui',
      owner: 'demo-org',
      owner_avatar_url: 'https://example.com/avatar.png',
      x: 10,
      y: 20,
      z: 0,
      cluster_id: 1,
      color: '#3b82f6',
      size: 18,
      star_list_id: 1,
      star_list_name: 'Favorites',
      stargazers_count: 123,
      ai_summary: 'Demo summary',
      ai_tags: ['demo'],
      topics: ['graph', 'viz'],
      starred_at: '2026-03-01T00:00:00Z',
      last_commit_time: '2026-03-02T00:00:00Z',
    },
    {
      id: 2,
      github_id: 102,
      full_name: 'demo-org/nebula-api',
      name: 'nebula-api',
      description: 'API demo repo',
      language: 'Python',
      html_url: 'https://github.com/demo-org/nebula-api',
      owner: 'demo-org',
      owner_avatar_url: 'https://example.com/avatar.png',
      x: 30,
      y: 40,
      z: 0,
      cluster_id: 1,
      color: '#3b82f6',
      size: 16,
      star_list_id: 1,
      star_list_name: 'Favorites',
      stargazers_count: 99,
      ai_summary: 'API summary',
      ai_tags: ['backend'],
      topics: ['api', 'python'],
      starred_at: '2026-03-01T00:00:00Z',
      last_commit_time: '2026-03-03T00:00:00Z',
    },
  ],
  edges: [],
  clusters: [
    {
      id: 1,
      name: 'Core',
      description: 'Core repos',
      keywords: ['graph', 'api'],
      color: '#3b82f6',
      repo_count: 2,
      center_x: 20,
      center_y: 30,
      center_z: 0,
    },
  ],
  star_lists: [{ id: 1, name: 'Favorites', description: 'Top picks', repo_count: 2 }],
  total_nodes: 2,
  total_edges: 1,
  total_clusters: 1,
  total_star_lists: 1,
  version: 'v-test',
  generated_at: '2026-03-04T00:00:00Z',
  request_id: 'req-e2e-graph',
};

const timelinePayload = {
  points: [
    {
      date: '2026-03',
      count: 2,
      repos: ['demo-org/nebula-ui', 'demo-org/nebula-api'],
      top_languages: ['TypeScript', 'Python'],
      top_topics: ['graph', 'api'],
    },
  ],
  total_stars: 2,
  date_range: ['2026-03', '2026-03'],
  version: 'v-test',
  generated_at: '2026-03-04T00:00:00Z',
  request_id: 'req-e2e-timeline',
};

const dataPayload = {
  items: graphPayload.nodes,
  count: 2,
  limit: 2000,
  offset: 0,
  version: 'v-test',
  generated_at: '2026-03-04T00:00:00Z',
  request_id: 'req-e2e-data',
};

const dashboardPayload = {
  summary: {
    total_repos: 2,
    embedded_repos: 2,
    total_clusters: 1,
    total_edges: 1,
  },
  top_clusters: [{ id: 1, name: 'Core', repo_count: 2, color: '#3b82f6' }],
  version: 'v-test',
  generated_at: '2026-03-04T00:00:00Z',
  request_id: 'req-e2e-dashboard',
};

const edgesPayload = {
  edges: [{ source: 1, target: 2, weight: 0.82 }],
  next_cursor: null,
  version: 'v-test',
  generated_at: '2026-03-04T00:00:00Z',
  request_id: 'req-e2e-edges',
};

const syncStartPayload = {
  pipeline_run_id: 9001,
  status: 'running',
  phase: 'sync',
  message: 'pipeline started',
  version: 'v-test',
  generated_at: '2026-03-04T00:00:00Z',
  request_id: 'req-e2e-sync',
};

const installApiMocks = async (page: Page) => {
  await page.route('**/*', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const pathname = url.pathname;
    if (!pathname.startsWith('/api/v2/')) {
      await route.continue();
      return;
    }
    if (pathname.endsWith('/api/v2/graph')) {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(graphPayload) });
      return;
    }
    if (pathname.endsWith('/api/v2/graph/timeline')) {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(timelinePayload) });
      return;
    }
    if (pathname.endsWith('/api/v2/graph/edges')) {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(edgesPayload) });
      return;
    }
    if (pathname.endsWith('/api/v2/data/repos')) {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(dataPayload) });
      return;
    }
    if (pathname.endsWith('/api/v2/dashboard')) {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(dashboardPayload) });
      return;
    }
    if (pathname.endsWith('/api/v2/sync/start') && request.method() === 'POST') {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(syncStartPayload) });
      return;
    }
    await route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ detail: 'mock not found' }) });
  });
};

test.describe('graph + sync critical flows', () => {
  test.skip(!runE2E, 'Set RUN_E2E=1 to execute browser e2e flow tests.');

  test.beforeEach(async ({ page }) => {
    await installApiMocks(page);
  });

  test('first graph load renders main viewport', async ({ page }) => {
    await page.goto('/graph');
    await expect(page).toHaveURL(/\/graph/);
    await expect(page.locator('canvas')).toBeVisible();
  });

  test('data filter links can navigate to graph detail route', async ({ page }) => {
    await page.goto('/data');
    await expect(page).toHaveURL(/\/data/);

    const repoLink = page.locator('a[href^="/graph?node="]').first();
    await repoLink.waitFor({ state: 'visible' });
    await repoLink.click();

    await expect(page).toHaveURL(/\/graph\?node=\d+/);
  });

  test('sync endpoint can be triggered through API in e2e env', async ({ page }) => {
    await page.goto('/graph');
    const payload = await page.evaluate(async () => {
      const response = await fetch('/api/v2/sync/start?mode=incremental', { method: 'POST' });
      return {
        ok: response.ok,
        payload: await response.json(),
      };
    });
    expect(payload.ok).toBeTruthy();
    const body = payload.payload as { pipeline_run_id?: number };
    expect(body.pipeline_run_id).toBeTruthy();
  });
});
