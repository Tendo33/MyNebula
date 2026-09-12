---
name: MyNebula
description: Geist / Vercel ink-on-canvas for a GitHub Stars graph app
colors:
  primary: "#171717"
  on-primary: "#ffffff"
  canvas: "#fafafa"
  elevated: "#ffffff"
  body: "#4d4d4d"
  mute: "#8f8f8f"
  hairline: "#ebebeb"
  link: "#0070f3"
  error: "#ee0000"
  warning: "#f5a623"
  warning-soft: "#ffefcf"
  warning-deep: "#ab570a"
typography:
  body:
    fontFamily: "Geist Variable"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 20px
  heading:
    fontFamily: "Geist Variable"
    fontWeight: 600
    letterSpacing: "-0.02em"
  mono:
    fontFamily: "Geist Mono Variable"
rounded:
  sm: "6px"
  md: "12px"
  lg: "16px"
spacing:
  sm: "8px"
  md: "16px"
  lg: "24px"
---

# MyNebula Design Source

Visual authority is the Vercel Geist system installed at `frontend/DESIGN.md` (`npx getdesign@latest add vercel`). Tokens in `frontend/src/index.css` map that system onto the app.

This is an **Operate** surface, not a marketing page. Use the app/nav chrome from that file, not the hero mesh or pill CTAs.

## Product Context

People explore a personal GitHub Stars graph. The shell must stay out of the way: scan, filter, open a repo.

## Visual Language

- Canvas `#fafafa`, ink `#171717`, hairline `#ebebeb`, elevated white cards.
- Primary actions fill with ink; ghost actions are white + 1px hairline.
- In-app controls are 6px squares. Do not use marketing pills in Dashboard / Graph / Data / Settings.
- Geist Sans for UI; Geist Mono only for code, counts, and spec-like labels.
- Color is reserved for links (`#0070f3`), graph selection, cluster hues, and semantic error. No ivory, no indigo chrome, no nebula gradients.

## Dark

Same system inverted: canvas `#0a0a0a`, ink `#ededed`, hairline `#2e2e2e`, primary button light-on-dark.

## Components

In-app chrome comes from shadcn on Base UI in `frontend/src/components/ui`, mapped to these tokens:

- `Button` default / outline / icon — 6px squares, 40px tall (primary ink fill, ghost hairline)
- `Card` default (12–16px elevated) and `muted` (settings rows)
- `Input`, `InputGroup`, `NativeSelect`, `Switch`, `Slider`, `Table`, `Sheet`, `Alert`, `EmptyState`

Do not add a second component library (Mantine, Ant, HeroUI, Radix Themes) on top of this stack. Cluster chips stay as `chip-button` because hues come from the graph.

## Motion

150–220ms color/border only, plus one product motion: a circular view-transition wipe when the user switches light/dark in Settings (Great UI circular-theme-provider, duration 220ms). No lift-on-hover. Honor `prefers-reduced-motion` by skipping the wipe and without stripping focus rings.
