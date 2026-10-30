/**
 * Minimal safe templating.
 *
 * Every `${value}` interpolated into `html` is HTML-escaped unless it is a
 * `Raw` fragment produced by `html` itself or by `raw()`. Repo descriptions and
 * summaries come from third parties, so escaping is the default, not opt-in.
 */
export class Raw {
  readonly value: string;
  constructor(value: string) {
    this.value = value;
  }
  toString(): string {
    return this.value;
  }
}

const ESCAPES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

export const escapeHtml = (value: unknown): string =>
  String(value ?? '').replace(/[&<>"']/g, (ch) => ESCAPES[ch]);

const render = (value: unknown): string => {
  if (value instanceof Raw) return value.value;
  if (Array.isArray(value)) return value.map(render).join('');
  if (value === null || value === undefined) return '';
  // Booleans print as "true"/"false" so `aria-pressed="${on}"` stays valid.
  // Conditional markup uses `cond ? html`…` : ''`, never `cond && …`.
  return escapeHtml(value);
};

export const html = (strings: TemplateStringsArray, ...values: unknown[]): Raw =>
  new Raw(strings.reduce((out, chunk, i) => out + chunk + (i < values.length ? render(values[i]) : ''), ''));

export const raw = (value: string): Raw => new Raw(value);

export const qs = <T extends Element = HTMLElement>(root: ParentNode, selector: string): T =>
  root.querySelector(selector) as T;

/** Delegated event handler keyed by `data-act` on the clicked element or an ancestor. */
export const onAction = (
  root: HTMLElement,
  type: string,
  handlers: Record<string, (el: HTMLElement, event: Event) => void>,
): (() => void) => {
  const listener = (event: Event) => {
    const el = (event.target as HTMLElement).closest<HTMLElement>('[data-act]');
    if (!el || !root.contains(el)) return;
    handlers[el.dataset.act!]?.(el, event);
  };
  root.addEventListener(type, listener);
  return () => root.removeEventListener(type, listener);
};

export const avatarUrl = (url: string | null | undefined, size = 80): string | null =>
  url ? `${url}${url.includes('?') ? '&' : '?'}s=${size}` : null;

export const formatCount = (value: number): string => value.toLocaleString();
