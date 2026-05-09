# DESIGN.md Workflow

Use this before UI or visual-system work in the Vite frontend.

## Source Priority

1. Project root `DESIGN.md`
2. Existing project design-system docs
3. This spec's frontend rules
4. General taste or ad hoc inspiration

`DESIGN.md` is the visual source of truth. It describes the intended look and
feel for AI agents. It does not override accessibility, responsiveness, type
safety, or the project's technical stack.

## When No DESIGN.md Exists

Before major UI work, choose a starting point from
`https://github.com/VoltAgent/awesome-design-md` or `https://getdesign.md`.

If there is no better product-specific direction, start with a restrained
product style such as Linear. For branded, venue, product, portfolio, or
object-focused pages, choose a design-md reference that matches the subject
matter rather than a generic SaaS dashboard.

Example:

```bash
npx getdesign@latest add linear.app
```

## Implementation Rules

- Read `DESIGN.md` before editing UI files.
- Translate the visual direction into Tailwind v4 semantic tokens and reusable
  components.
- Do not blindly copy a brand; adapt the mood, spacing, typography, and
  interaction patterns to the target project.
- If `DESIGN.md` conflicts with existing accessible behavior, keep the behavior
  accessible and document the adjustment.
- Do not overwrite an existing `DESIGN.md` without explicit user approval.

## Handoff

For visible UI changes, mention which `DESIGN.md` was used and what browser
viewports were checked.
