---
name: MyNebula
description: Geist / Vercel ink-on-canvas for a GitHub Stars graph app
colors:
  primary: "#171717"
  on-primary: "#ffffff"
  canvas: "#fafafa"
  hover: "#f2f2f2"
  elevated: "#ffffff"
  ink: "#171717"
  body: "#4d4d4d"
  mute: "#737373"
  hairline: "#ebebeb"
  link: "#0070f3"
  dark-canvas: "#0a0a0a"
  dark-hover: "#171717"
  dark-elevated: "#111111"
  dark-ink: "#ededed"
  dark-body: "#a1a1a1"
  dark-mute: "#8f8f8f"
  dark-hairline: "#2e2e2e"
  dark-link: "#47a8ff"
typography:
  body:
    fontFamily: "Geist"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  small:
    fontFamily: "Geist"
    fontSize: "12px"
    fontWeight: 400
  heading:
    fontFamily: "Geist"
    fontSize: "16px"
    fontWeight: 600
    letterSpacing: "-0.02em"
  mono:
    fontFamily: "ui-monospace"
    fontSize: "14px"
rounded:
  xs: "4px"
  sm: "6px"
  md: "12px"
  full: "999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
---

# MyNebula Design Source

The visual language is Vercel's Geist app chrome: near-white canvas, one near-black ink, 1px hairlines, 6px square controls. Tokens live as CSS variables at the top of `frontend/src/styles.css`; `html.dark` swaps them.

This is an **Operate** surface, not a marketing page: no hero mesh, no pill CTAs.

## Product Context

People explore their own or a shared GitHub Stars graph. The shell must stay out of the way: scan, filter, open a repository.

## Type

- Geist Sans, self-hosted Latin subset only (`frontend/src/assets/geist-latin.woff2`); Chinese and other scripts use the system face. Monospace is the system mono and only for code.
- Three sizes: 12px (meta, captions, rail headings), 14px (body, controls, table), 16px (page and repository names). Weight 400 / 500 for labels / 600 for names.
- Numbers in counts, tables, and dates use tabular figures.

## Visual Language

- Canvas `#fafafa`, ink `#171717`, hairline `#ebebeb`; elevated white only for floating layers.
- Primary actions fill with ink; ghost actions are white + 1px hairline.
- In-app controls are 6px squares. Do not use marketing pills in Graph / Data / Settings.
- Color is reserved for links (`#0070f3`), graph selection, cluster hues, and semantic error. No ivory, no indigo chrome, no nebula gradients.

## Dark

Same system inverted: canvas `#0a0a0a`, ink `#ededed`, hairline `#2e2e2e`, primary button light-on-dark. Canvas-drawn labels and the hover ring switch to the dark ink too.

## Focus and touch

- Every control shows a 2px link-blue outline on keyboard focus; framed inputs show it on the frame, not the inner field.
- On coarse pointers, controls and rail rows grow to 40px.

## Components

In-app chrome is hand-written CSS in `frontend/src/styles.css`, mapped to these tokens:

- `.btn` primary / outline / ghost / icon: 6px squares, 32px tall (primary ink fill, outline hairline)
- Structure comes from 1px hairlines, not cards. Only floating layers (hover card, node list, palette) are 12px elevated surfaces.
- `.search-input`, `.select`, `.switch`, `.range`, `.toggle-group`, `.table`, `.chip`. Native selects keep the same 6px corners and hairlines.
- The shell is one 48px top bar: wordmark, Graph / Data / Settings tabs with an ink underline, and the search trigger. No sidebar, no breadcrumb.
- Graph is the home page: a flat filter rail (Clusters, My Lists, Timeline as hairline sections, rows without checkboxes) and a full-width canvas. Selecting a repository adds a ~23rem details panel: name, a two-column facts grid, summary, tags, text links, then related repos in one scroll with no tabs.
- Graph layout is a live force simulation on the similarity edges: after avatars start loading, nodes burst out one wedge per cluster, settle into communities, then keep a slow drift. Dragging reheats the layout; dim and highlight changes fade. Reduced motion computes the resting layout up front with no drift. The overview shows a sparse strongest-edge backbone, while hover and selection reveal related edges.
- Data is one toolbar row (search, count, clear), one horizontally scrolling row of cluster chips, a full-bleed hairline table, and a pager holding rows-per-page.
- Settings is one centred hairline list: Appearance, Data, Sync.
- Unselected graph nodes are neutral and unlabeled. Hover or selection keeps hue and labels on that cluster and drops the others to about 0.18 opacity. No nebula background.

Do not add a UI framework or component library. Cluster chips take their hue inline because colours come from the graph.

## Motion

150–220ms color/border only, plus two product motions: the graph's burst-then-drift layout (pauses in hidden tabs), and a circular view-transition wipe when the user switches light/dark in Settings (Great UI circular-theme-provider, duration 220ms). No lift-on-hover. Honor `prefers-reduced-motion` by skipping the wipe and without stripping focus rings.
