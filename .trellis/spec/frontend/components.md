# Components

## Component Rules

- Components use TypeScript and explicit props.
- Shared components live under `frontend/src/components`.
- Feature-specific components should live with the feature once real features
  exist.
- UI primitives should be small, composable, and tested when behavior is not
  trivial.

## Styling

- Prefer Tailwind CSS v4 semantic tokens.
- Keep global CSS focused on theme tokens, reset-level rules, and app-wide
  primitives.
- Avoid one-off hard-coded colors unless the design system explicitly calls for
  them.
- Preserve visible focus styles.

## Interaction Testing

- Use Vitest + Testing Library.
- Prefer `userEvent` for user interactions.
- Test visible behavior and accessible state, not private component internals.

## Design Docs

If a UI change expands the visual language, update the project's design-system
doc or `DESIGN.md` in the same change.
