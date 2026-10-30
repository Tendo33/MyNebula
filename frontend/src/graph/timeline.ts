import { html, type Raw } from '../html';
import { getLang, t } from '../i18n';
import type { TimelinePoint } from '../types';

const BAR_MIN = 4;
const BAR_MAX = 48;

const monthLabel = (date: string): string =>
  new Intl.DateTimeFormat(getLang(), { month: 'short', year: 'numeric', timeZone: 'UTC' }).format(
    new Date(`${date}-01T00:00:00Z`),
  );

export const renderTimeline = (points: TimelinePoint[], range: [number, number] | null): Raw => {
  if (!points.length) return html``;
  const max = Math.max(...points.map((p) => p.count), 1);
  const inRange = (i: number) => !range || (i >= range[0] && i <= range[1]);
  return html`<section class="rail-section timeline">
    <div class="rail-section-head">
      <h2>${t('graph.timeline')}</h2>
      ${range ? html`<button class="link-btn" data-act="timeline-clear">${t('common.clear')}</button>` : ''}
    </div>
    <div class="timeline-bars" data-timeline>
      ${points.map(
        (point, i) => html`<button type="button" class="timeline-bar" data-index="${i}"
          aria-pressed="${Boolean(range && inRange(i))}"
          aria-label="${monthLabel(point.date)}: ${t('search.repo_count', { count: point.count })}">
          <span class="timeline-fill ${inRange(i) ? 'on' : ''} ${range && (i === range[0] || i === range[1]) ? 'edge' : ''}"
            style="height:${BAR_MIN + (point.count / max) * (BAR_MAX - BAR_MIN)}px"></span>
          <span class="tooltip">
            <strong>${t('search.repo_count', { count: point.count })}</strong>
            <span>${monthLabel(point.date)}</span>
            ${point.top_languages.length ? html`<small>${point.top_languages.slice(0, 3).join(', ')}</small>` : ''}
          </span>
        </button>`,
      )}
    </div>
    <div class="timeline-axis mono"><span>${monthLabel(points[0].date)}</span><span>${monthLabel(points[points.length - 1].date)}</span></div>
    ${range
      ? html`<div class="timeline-summary small">
          <span class="muted">${range[0] === range[1] ? monthLabel(points[range[0]].date) : `${monthLabel(points[range[0]].date)} — ${monthLabel(points[range[1]].date)}`}</span>
          <strong>${t('search.repo_count', {
            count: points.slice(range[0], range[1] + 1).reduce((sum, p) => sum + p.count, 0),
          })}</strong>
        </div>`
      : ''}
  </section>`;
};

/**
 * Drag-to-select brush over the timeline bars. Dragging an edge resizes,
 * dragging inside moves the window, and anywhere else starts a new range.
 */
export const attachTimelineBrush = (
  root: HTMLElement,
  getPoints: () => TimelinePoint[],
  getRange: () => [number, number] | null,
  preview: (range: [number, number]) => void,
  commit: (range: [number, number] | null) => void,
): (() => void) => {
  let drag: { handle: 'start' | 'end' | 'range'; offset: number; range: [number, number]; container: HTMLElement } | null = null;

  const indexAt = (container: HTMLElement, clientX: number) => {
    const rect = container.getBoundingClientRect();
    const count = getPoints().length;
    return Math.max(0, Math.min(count - 1, Math.floor(((clientX - rect.left) / rect.width) * count)));
  };

  const onDown = (event: PointerEvent) => {
    const container = (event.target as HTMLElement).closest<HTMLElement>('[data-timeline]');
    if (!container || !root.contains(container)) return;
    event.preventDefault();
    const index = indexAt(container, event.clientX);
    const current = getRange();
    if (current && index === current[0]) drag = { handle: 'start', offset: 0, range: current, container };
    else if (current && index === current[1]) drag = { handle: 'end', offset: 0, range: current, container };
    else if (current && index > current[0] && index < current[1]) {
      drag = { handle: 'range', offset: index - current[0], range: current, container };
    } else drag = { handle: 'end', offset: 0, range: [index, index], container };
    preview(drag.range);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp, { once: true });
  };

  const onMove = (event: PointerEvent) => {
    if (!drag) return;
    const index = indexAt(drag.container, event.clientX);
    const [start, end] = drag.range;
    const last = getPoints().length - 1;
    if (drag.handle === 'start') drag.range = [Math.min(index, end), end];
    else if (drag.handle === 'end') drag.range = [start, Math.max(index, start)];
    else {
      const width = end - start;
      const nextStart = Math.max(0, Math.min(index - drag.offset, last - width));
      drag.range = [nextStart, nextStart + width];
    }
    preview(drag.range);
  };

  const onUp = () => {
    window.removeEventListener('pointermove', onMove);
    if (!drag) return;
    const [start, end] = drag.range;
    // The full span is the same as no filter.
    commit(start === 0 && end === getPoints().length - 1 ? null : drag.range);
    drag = null;
  };

  const onKey = (event: KeyboardEvent) => {
    const bar = (event.target as HTMLElement).closest<HTMLElement>('.timeline-bar');
    if (!bar) return;
    const index = Number(bar.dataset.index);
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      commit([index, index]);
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      const next = bar.parentElement?.children[index + (event.key === 'ArrowRight' ? 1 : -1)] as HTMLElement | undefined;
      next?.focus();
    } else if (event.key === 'Escape') {
      commit(null);
    }
  };

  root.addEventListener('pointerdown', onDown);
  root.addEventListener('keydown', onKey);
  return () => {
    root.removeEventListener('pointerdown', onDown);
    root.removeEventListener('keydown', onKey);
    window.removeEventListener('pointermove', onMove);
  };
};
