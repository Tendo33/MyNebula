# Frontend Quality

## Mandatory Rules

- UI work follows the project root `DESIGN.md` when present.
- TypeScript remains strict.
- No `any`, non-null assertions, or ignored TypeScript errors in new code.
- Components must be responsive and accessible.
- Visible focus styles must remain visible.
- Every page has one `h1`, a `main#main-content` skip target, and language
  changes synchronize `<html lang>`.
- Modal progress surfaces manage focus entry/return, allow Escape only when
  closable, and expose live/progress semantics.
- Modal focus lifecycle effects depend only on open/closed state. Changing an
  `onClose` callback or closeability while open must use current-value refs and
  must not restore/re-enter focus mid-interaction.
- Pointer-driven timeline and resize controls require equivalent keyboard and
  value semantics.
- Styling should use semantic tokens.
- Theme or visual-system changes should update the project design docs.

## Testing

- Use Vitest + Testing Library.
- Prefer `userEvent` for interactions.
- Test visible behavior, accessible names, and state changes.

## Static Mount Checks

When the backend serves the Vite build:

- Build the frontend.
- Test backend static serving or fallback behavior.
- Confirm unknown API routes do not return `index.html`.

## Visual Checks

- Confirm the implemented typography, spacing, color roles, and component states
  match `DESIGN.md`.
- Check at least one mobile and one desktop viewport for visible UI changes.
