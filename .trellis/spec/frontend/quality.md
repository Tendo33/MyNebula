# Frontend Quality

## Mandatory Rules

- UI work follows the project root `DESIGN.md` when present.
- TypeScript remains strict.
- No `any`, non-null assertions, or ignored TypeScript errors in new code.
- Components must be responsive and accessible.
- Visible focus styles must remain visible.
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
