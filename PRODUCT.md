# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: developers who star a lot of GitHub repositories and want that collection to behave like a personal knowledge space. They fork MyNebula, add their own keys, and deploy their own nebula; the maintainer is the first such user.

Secondary: people the owner shares the deployed site with, who browse someone else's stars to see what that person follows.

Jobs: find a repository again, see what a group of interests contains, open a repository and its neighbours, and keep the map fresh without curating a bookmark list.

## Product Purpose

MyNebula turns GitHub Stars into a semantic graph published as a static site. Success is: set it up once, let the scheduled sync keep it current, and have related repositories visibly sit together so exploring by connection beats scrolling a list.

## Positioning

Connections, not folders: semantic similarity links decide both the layout and the groups, so related repositories cluster without manual tagging. It runs as a static site plus an incremental offline pipeline, deployable for free on GitHub Pages, with an optional zero-cost mode that skips chat-model calls.

## Operating Context

- Setup: fork the repository, add `STARS_TOKEN` and `EMBEDDING_API_KEY` (optionally `LLM_API_KEY`) as Actions secrets, enable Pages. A daily GitHub Actions run syncs and redeploys; `pnpm sync` does the same locally.
- Use: the owner opens the site to search (Cmd/Ctrl+K), filter by group, star list, or time, and read a repository's summary and related repos. Shared visitors arrive by link, often a deep link to a node or filter.
- The site is public by default; private access belongs to the hosting layer (for example Cloudflare Access).

## Capabilities and Constraints

- Pages: Graph (home), Data (sortable, filterable table), Settings (appearance, data snapshot, sync instructions). English and Chinese UI; light, dark, and system themes.
- Static only: no server, no login, no write actions in the browser. Syncing, re-summarizing, and regrouping happen in the pipeline (`--full`, `--recluster`, `--llm-names`, `--max-clusters`).
- One TypeScript package on Node 24; the pipeline has no runtime dependencies and must stay incremental, so only new or changed repositories reach paid APIs.
- Terminology: "groups" (clusters) are Louvain communities of the similarity graph; "links" are similarity edges; "star lists" are the owner's GitHub lists.
- Search is keyword-based; there is no semantic query search in a static build.

## Brand Commitments

- Name: MyNebula. Logo and banner in `doc/images/`.
- Personality: precise, exploratory, calm. Credible and engineered, not a sci-fi toy.
- Visual language is Vercel Geist as recorded in `DESIGN.md`.

## Anti-references

Ivory/indigo “research desk” leftovers, gimmicky space visuals, Inter/Roboto marketing templates, dashboard metric-card grids, and mixing Vercel marketing pills into in-app chrome.

## Evidence on Hand

- `doc/images/logo.png` and `doc/images/banner.png`. There are no current interface screenshots.
- No testimonials, user counts, benchmarks, or case studies exist. Do not invent them.

## Product Principles

- Connection first: structure comes from similarity links; the graph is home and every other view serves it.
- Free to run, cheap to keep fresh: the setup and the daily sync must stay within free tiers, and nothing re-runs paid work for unchanged repositories.
- Owner's map, shareable by link: deep links (node, group, filters) must keep working for visitors who did not set it up.
- Calm under density: hundreds to thousands of repositories stay scannable, and loading, empty, and error states are complete.

## Accessibility & Inclusion

Public Operate UI targets WCAG AA, including shared visitors. The graph canvas is not an accessible node tree; the repository list is the equivalent path and must stay keyboard-complete. Honour reduced motion: the graph renders its resting layout without drift.
