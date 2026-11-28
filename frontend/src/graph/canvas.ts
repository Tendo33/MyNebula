import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from 'd3-force';

import type { ClusterInfo, GraphEdge, GraphNode } from '../types';
import { avatarUrl } from '../html';
import {
  COLORS,
  LINK_STYLE,
  UNFOCUSED_NODE_ALPHA,
  buildOverviewLinks,
  classifyLink,
  convexHull,
  focusClusterId,
  nodeColor,
  nodeDimmed,
  nodeRadius,
  showClusterHue,
  showLabel,
  type FocusContext,
  type LinkContext,
} from './style';

const sameIds = (left: ReadonlySet<number>, right: ReadonlySet<number>): boolean => {
  if (left.size !== right.size) return false;
  for (const id of left) if (!right.has(id)) return false;
  return true;
};

const FONT = 'Geist, system-ui, sans-serif';
const FIT_PADDING = 80;
const FOCUS_ZOOM = 1.6;
/** Below this zoom, neighbour labels would collide; only the focused node is labelled. */
const NEIGHBOR_LABEL_ZOOM = 2.2;
const MIN_ZOOM = 0.05;
const MAX_ZOOM = 12;
/** Let avatars start loading before the burst so it is not drawn as grey dots. */
const AVATAR_PRELOAD_MS = 400;
/** The viewport follows the unfolding layout for this long, unless the user takes over. */
const FOLLOW_MS = 4500;
/** Public d3 velocityDecay. The simulation keeps 60% of each tick's velocity. */
const VELOCITY_DECAY = 0.4;
/** Drawn sway after the layout settles. Small enough that labels and hits stay with the node. */
const DRIFT_PX = 2;
/** Once settled, redraw at ~30fps: the drift is slow and this halves the paint cost. */
const IDLE_FRAME_MS = 33;
/** GitHub avatars are requested at 80px. The sprite stays at that size. */
const AVATAR_SPRITE = 80;
/** Per-frame easing for dim/highlight fades. */
const FADE_RATE = 0.18;

export interface SimNode extends SimulationNodeDatum {
  id: number;
  name: string;
  cluster_id: number | null;
  color: string;
  radius: number;
  avatar: string | null;
  /** Drift phase and per-node frequency jitter. */
  phase: number;
  freq: number;
  /** Current drawn opacity, eased toward the focus rules. */
  fade: number;
}

interface SimLink extends SimulationLinkDatum<SimNode> {
  source: SimNode;
  target: SimNode;
  sourceId: number;
  targetId: number;
  weight: number;
  /** Cached stroke. Recomputed when focus changes, reused while nodes only drift. */
  stroke?: { color: string; width: number };
}

export interface GraphView {
  visible: ReadonlySet<number>;
  selectedId: number | null;
  neighbors: ReadonlySet<number>;
  showTrajectories: boolean;
  hqRendering: boolean;
}

interface Transform {
  k: number;
  x: number;
  y: number;
}

interface WorldView {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/** Deterministic 0..1 noise from an id, so the burst looks the same on every load. */
const hash01 = (n: number, salt: number) => {
  const x = Math.sin(n * 12.9898 + salt * 78.233) * 43758.5453;
  return x - Math.floor(x);
};

/**
 * Pull members toward their cluster's live centroid. Groups emerge from the
 * edges; this only keeps a community from smearing across the canvas.
 */
const clusterCohesion = (strength: number) => {
  let nodes: SimNode[] = [];
  const force = (alpha: number) => {
    const sums = new Map<number, { x: number; y: number; n: number }>();
    for (const node of nodes) {
      if (node.cluster_id == null) continue;
      const sum = sums.get(node.cluster_id) ?? { x: 0, y: 0, n: 0 };
      sum.x += node.x!;
      sum.y += node.y!;
      sum.n += 1;
      sums.set(node.cluster_id, sum);
    }
    const k = alpha * strength;
    for (const node of nodes) {
      const sum = node.cluster_id == null ? undefined : sums.get(node.cluster_id);
      if (!sum) continue;
      node.vx! += (sum.x / sum.n - node.x!) * k;
      node.vy! += (sum.y / sum.n - node.y!) * k;
    }
  };
  force.initialize = (next: SimNode[]) => {
    nodes = next;
  };
  return force;
};

/**
 * Canvas force graph laid out purely by similarity edges.
 *
 * Nodes burst out of the centre, one wedge per cluster, and settle. The force
 * simulation then stops. Drift is a position offset applied while drawing and
 * hit-testing, so the stored layout stays where the forces left it. The
 * viewport follows the burst until the user pans or zooms. Drift pauses in
 * background tabs and is off under reduced motion, where the layout is
 * computed up front and drawn once.
 */
export class GraphCanvas {
  readonly canvas = document.createElement('canvas');
  private ctx = this.canvas.getContext('2d')!;
  private nodes: SimNode[] = [];
  private nodeById = new Map<number, SimNode>();
  private links: SimLink[] = [];
  private overview = new Set<SimLink>();
  private clusterOf = new Map<number, number | null>();
  private clusters = new Map<number, ClusterInfo>();
  private sim: Simulation<SimNode, SimLink> | null = null;
  private view: GraphView = {
    visible: new Set(),
    selectedId: null,
    neighbors: new Set(),
    showTrajectories: true,
    hqRendering: true,
  };
  private hoverId: number | null = null;
  /** Link strokes depend on focus, not on the drift, so they can be reused between frames. */
  private linkStylesReady = false;
  /** True once the force simulation has cooled. Drift is then a draw offset. */
  private settled = false;
  private coasting = false;
  /** Time origin of the draw offset, on the paused clock, so the sway starts at zero. */
  private coastT0 = 0;
  /** Seconds the tab has spent hidden. The sway clock does not advance while paused. */
  private paused = 0;
  /** performance.now()/1000 when the tab was hidden. Zero while visible. */
  private hiddenAt = 0;
  /** The next frame must paint, even inside the idle interval. */
  private paintNow = false;
  /** Node opacity is still easing, so the idle interval cannot skip the frame. */
  private nodesFading = false;
  private sprites = new Map<string, HTMLCanvasElement>();
  private transform: Transform = { k: 1, x: 0, y: 0 };
  private animation: { from: Transform; to: Transform; start: number; duration: number } | null = null;
  private width = 0;
  private height = 0;
  /** Host left edge and window width at the last resize; an unchanged window means a panel moved. */
  private left = 0;
  private viewportWidth = 0;
  private dpr = 1;
  private frame = 0;
  private lastDraw = 0;
  private startTimer = 0;
  private followUntil = 0;
  /** True from setData until the burst has settled and the camera stopped following. */
  private bursting = false;
  private pendingFocus: { id: number; duration: number } | null = null;
  private userInteracted = false;
  private images = new Map<string, HTMLImageElement | 'loading' | 'error'>();
  private pointers = new Map<number, { x: number; y: number }>();
  private gesture:
    | { kind: 'pan'; startX: number; startY: number; origin: Transform; moved: boolean }
    | { kind: 'drag'; node: SimNode; moved: boolean }
    | { kind: 'pinch'; distance: number; origin: Transform; center: { x: number; y: number } }
    | null = null;
  private resizeObserver: ResizeObserver;
  private reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  /** Theme-dependent ink for labels, the hover ring, and the avatar hairline (refreshed per frame). */
  private ink: { labelBg: string; labelText: string; hover: string; ring: string } = {
    labelBg: COLORS.LABEL_BG,
    labelText: COLORS.LABEL_TEXT,
    hover: COLORS.NODE_HOVER,
    ring: 'rgba(0, 0, 0, 0.08)',
  };

  private host: HTMLElement;
  private callbacks: { onSelect: (id: number | null) => void; onHover: (id: number | null) => void };

  constructor(host: HTMLElement, callbacks: GraphCanvas['callbacks']) {
    this.host = host;
    this.callbacks = callbacks;
    this.canvas.className = 'graph-canvas';
    host.appendChild(this.canvas);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.resize();

    this.canvas.addEventListener('pointerdown', this.onPointerDown);
    this.canvas.addEventListener('pointermove', this.onPointerMove);
    this.canvas.addEventListener('pointerup', this.onPointerUp);
    this.canvas.addEventListener('pointercancel', this.onPointerUp);
    this.canvas.addEventListener('pointerleave', this.onPointerLeave);
    this.canvas.addEventListener('wheel', this.onWheel, { passive: false });
    document.addEventListener('visibilitychange', this.onVisibility);
  }

  destroy(): void {
    this.coasting = false;
    this.sim?.stop();
    window.clearTimeout(this.startTimer);
    cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.resizeObserver.disconnect();
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.canvas.remove();
  }

  setData(nodes: GraphNode[], edges: GraphEdge[], clusters: ClusterInfo[]): void {
    this.sim?.stop();
    this.settled = false;
    this.coasting = false;
    this.nodesFading = false;
    cancelAnimationFrame(this.frame);
    this.frame = 0;
    window.clearTimeout(this.startTimer);
    this.clusters = new Map(clusters.map((c) => [c.id, c]));
    const colorOf = new Map(clusters.map((c) => [c.id, c.color]));
    // One wedge of the burst per cluster, largest clusters spread evenly.
    const wedge = new Map(clusters.map((c, i) => [c.id, (i / Math.max(clusters.length, 1)) * Math.PI * 2]));

    this.nodes = nodes.map((node) => {
      const base = node.cluster_id != null ? wedge.get(node.cluster_id) : undefined;
      const angle = (base ?? hash01(node.id, 1) * Math.PI * 2) + (hash01(node.id, 2) - 0.5) * 0.9;
      const r = 4 + hash01(node.id, 3) * 26;
      return {
        id: node.id,
        name: node.name,
        cluster_id: node.cluster_id,
        color: (node.cluster_id != null && colorOf.get(node.cluster_id)) || COLORS.NODE_DEFAULT,
        radius: nodeRadius(node.stargazers_count),
        avatar: avatarUrl(node.owner_avatar_url),
        x: Math.cos(angle) * r,
        y: Math.sin(angle) * r,
        phase: hash01(node.id, 4) * Math.PI * 2,
        freq: 0.35 + hash01(node.id, 5) * 0.35,
        fade: 1,
      };
    });
    this.nodeById = new Map(this.nodes.map((n) => [n.id, n]));
    this.clusterOf = new Map(this.nodes.map((n) => [n.id, n.cluster_id]));
    this.links = edges
      .filter(([s, t]) => this.nodeById.has(s) && this.nodeById.has(t))
      .map(([s, t, weight]) => ({
        source: this.nodeById.get(s)!,
        target: this.nodeById.get(t)!,
        sourceId: s,
        targetId: t,
        weight,
      }));
    this.overview = buildOverviewLinks(this.links);
    this.linkStylesReady = false;
    this.userInteracted = false;
    this.pendingFocus = null;
    this.bursting = !this.reduceMotion && this.nodes.length > 0;
    this.transform = { k: 1, x: this.width / 2, y: this.height / 2 };

    this.sim = this.createSimulation();
    if (this.reduceMotion || this.nodes.length === 0) {
      // Compute the resting layout up front and draw it once.
      this.sim.stop();
      for (let i = 0; i < 300; i += 1) this.sim.tick();
      this.fitTo(0);
      this.requestDraw();
      return;
    }
    this.sim.stop();
    this.preloadAvatars();
    this.fitTo(0, 0.6);
    this.requestDraw();
    this.startTimer = window.setTimeout(() => {
      this.followUntil = performance.now() + FOLLOW_MS;
      this.sim?.alpha(1).restart();
    }, AVATAR_PRELOAD_MS);
  }

  private createSimulation(): Simulation<SimNode, SimLink> {
    const degree = new Map<number, number>();
    this.links.forEach((l) => {
      degree.set(l.sourceId, (degree.get(l.sourceId) ?? 0) + 1);
      degree.set(l.targetId, (degree.get(l.targetId) ?? 0) + 1);
    });
    const sim = forceSimulation<SimNode, SimLink>(this.nodes)
      .alphaDecay(0.03)
      .alphaTarget(0)
      .velocityDecay(VELOCITY_DECAY)
      .force(
        'link',
        forceLink<SimNode, SimLink>(this.links)
          // Stronger similarity, shorter and stiffer spring.
          .distance((l) => 24 + (1 - l.weight) * 110 + (l.source.radius + l.target.radius) * 0.6)
          .strength(
            (l) =>
              (0.3 + l.weight * 0.7) /
              Math.min(degree.get(l.sourceId) ?? 1, degree.get(l.targetId) ?? 1),
          ),
      )
      .force('charge', forceManyBody<SimNode>().strength((n) => -40 - n.radius * 3).distanceMax(420))
      .force('collide', forceCollide<SimNode>((n) => n.radius + 2).strength(0.8).iterations(2))
      .force('cluster', clusterCohesion(0.12))
      .force('x', forceX<SimNode>(0).strength(0.035))
      .force('y', forceY<SimNode>(0).strength(0.035));
    sim.on('tick', () => {
      if (sim.alpha() > 0.05 || performance.now() - this.lastDraw >= IDLE_FRAME_MS) this.requestDraw();
    });
    sim.on('end', () => {
      for (const node of this.nodes) {
        node.vx = 0;
        node.vy = 0;
      }
      this.settled = true;
      this.coastT0 = this.motionTime();
      this.coasting = !this.reduceMotion && this.nodes.length > 0 && !document.hidden;
      // The camera stops following at FOLLOW_MS; settle it on the finished layout
      // so a wide screen is not left with a small, off-centre cloud.
      if (!this.userInteracted) {
        this.bursting = false;
        this.followUntil = 0;
        if (this.pendingFocus) {
          const { id, duration } = this.pendingFocus;
          this.pendingFocus = null;
          this.focusNode(id, duration || 800);
        } else {
          this.fitTo(this.reduceMotion ? 0 : 500);
        }
      }
      if (this.coasting) this.requestDraw();
    });
    return sim;
  }

  setView(view: GraphView): void {
    const paint =
      view.selectedId !== this.view.selectedId ||
      view.showTrajectories !== this.view.showTrajectories ||
      !sameIds(view.visible, this.view.visible) ||
      !sameIds(view.neighbors, this.view.neighbors);
    const hqChanged = view.hqRendering !== this.view.hqRendering;
    if (!paint && !hqChanged) return;
    const enableHq = hqChanged && view.hqRendering;
    this.view = view;
    if (paint) this.linkStylesReady = false;
    if (enableHq) this.preloadAvatars();
    this.requestDraw();
  }

  /** Centre on a node at a fixed, readable zoom. */
  focusNode(id: number, duration = 800, { deferDuringBurst = false } = {}): void {
    if (deferDuringBurst && this.bursting && !this.userInteracted) {
      // Positions are still unfolding; focus where the node actually lands.
      this.pendingFocus = { id, duration };
      return;
    }
    const node = this.nodeById.get(id);
    // Measure now: a panel opening alongside the selection may have just resized us.
    this.measure();
    if (!node || !this.width) return;
    // An explicit focus ends the camera's burst-follow.
    this.followUntil = 0;
    this.bursting = false;
    this.pendingFocus = null;
    const placed = this.place(node);
    this.animateTo(
      { k: FOCUS_ZOOM, x: this.width / 2 - placed.x * FOCUS_ZOOM, y: this.height / 2 - placed.y * FOCUS_ZOOM },
      duration,
    );
  }

  private fitTransform(): Transform | null {
    if (!this.nodes.length || !this.width || !this.height) return null;
    // A fixed 80px pad would leave half a phone's canvas empty; scale it with the
    // short side so the graph fills the frame at every width.
    const pad = Math.max(24, Math.min(FIT_PADDING, Math.min(this.width, this.height) * 0.09));
    let [minX, maxX, minY, maxY] = [Infinity, -Infinity, Infinity, -Infinity];
    for (const n of this.nodes) {
      const p = this.place(n);
      minX = Math.min(minX, p.x - n.radius);
      maxX = Math.max(maxX, p.x + n.radius);
      minY = Math.min(minY, p.y - n.radius);
      maxY = Math.max(maxY, p.y + n.radius);
    }
    const k = Math.min(
      MAX_ZOOM,
      Math.max(
        MIN_ZOOM,
        Math.min((this.width - pad * 2) / Math.max(maxX - minX, 1), (this.height - pad * 2) / Math.max(maxY - minY, 1)),
      ),
    );
    return { k, x: this.width / 2 - ((minX + maxX) / 2) * k, y: this.height / 2 - ((minY + maxY) / 2) * k };
  }

  /** Fit all nodes; `maxZoom` keeps the compact pre-burst cluster from filling the screen. */
  private fitTo(duration: number, maxZoom = MAX_ZOOM): void {
    const target = this.fitTransform();
    if (!target) return;
    if (target.k > maxZoom) {
      const ratio = maxZoom / target.k;
      target.x = this.width / 2 - (this.width / 2 - target.x) * ratio;
      target.y = this.height / 2 - (this.height / 2 - target.y) * ratio;
      target.k = maxZoom;
    }
    this.animateTo(target, duration);
  }

  private animateTo(to: Transform, duration: number): void {
    if (duration <= 0) {
      this.animation = null;
      this.transform = to;
      this.requestDraw();
      return;
    }
    this.animation = { from: { ...this.transform }, to, start: performance.now(), duration };
    this.requestDraw();
  }

  private measure(): void {
    const rect = this.host.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height;
  }

  private resize(): void {
    const previous = { width: this.width, height: this.height, left: this.left, viewportWidth: this.viewportWidth };
    this.measure();
    this.left = this.host.getBoundingClientRect().left;
    this.viewportWidth = window.innerWidth;
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(this.width * this.dpr));
    this.canvas.height = Math.max(1, Math.round(this.height * this.dpr));
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    // Keep the same world point centred when panels open or the window resizes.
    const centred = {
      ...this.transform,
      x: this.transform.x + (this.width - previous.width) / 2,
      y: this.transform.y + (this.height - previous.height) / 2,
    };
    if (previous.width && previous.viewportWidth === this.viewportWidth && !this.animation) {
      // A panel opened or closed: hold every node where it is on screen, then glide
      // to the centred view instead of jumping by half the panel width.
      this.transform = { ...this.transform, x: this.transform.x + previous.left - this.left };
      this.animateTo(centred, this.reduceMotion ? 0 : 320);
    } else {
      this.transform = centred;
      this.requestDraw();
    }
  }

  private onVisibility = () => {
    if (!this.sim || this.reduceMotion) return;
    if (document.hidden) {
      this.hiddenAt = performance.now() / 1000;
      this.sim.stop();
      this.coasting = false;
      cancelAnimationFrame(this.frame);
      this.frame = 0;
      return;
    }
    if (this.hiddenAt) {
      this.paused += performance.now() / 1000 - this.hiddenAt;
      this.hiddenAt = 0;
    }
    if (this.settled) {
      this.coasting = this.nodes.length > 0;
      this.requestDraw();
    } else this.sim.restart();
  };

  private preloadAvatars(): void {
    if (!this.view.hqRendering) return;
    this.nodes.forEach((n) => n.avatar && this.image(n.avatar));
  }

  private image(url: string): HTMLImageElement | null {
    const cached = this.images.get(url);
    if (cached instanceof HTMLImageElement) return cached;
    if (cached) return null;
    this.images.set(url, 'loading');
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.decoding = 'async';
    img.onload = () => {
      this.images.set(url, img);
      this.requestDraw();
    };
    img.onerror = () => this.images.set(url, 'error');
    img.src = url;
    return null;
  }

  /** Clip once. Later frames scale this sprite instead of clipping the image again. */
  private avatarSprite(url: string): HTMLCanvasElement | null {
    const cached = this.sprites.get(url);
    if (cached) return cached;
    const img = this.image(url);
    if (!img) return null;
    const sprite = document.createElement('canvas');
    const size = Math.max(1, Math.min(img.naturalWidth || AVATAR_SPRITE, AVATAR_SPRITE));
    sprite.width = size;
    sprite.height = size;
    const g = sprite.getContext('2d');
    if (!g) return null;
    g.beginPath();
    g.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2);
    g.clip();
    g.drawImage(img, 0, 0, size, size);
    this.sprites.set(url, sprite);
    return sprite;
  }

  private worldView(): WorldView {
    const { k, x, y } = this.transform;
    return { x0: -x / k, y0: -y / k, x1: (this.width - x) / k, y1: (this.height - y) / k };
  }

  private onScreen(x: number, y: number, pad: number, view: WorldView): boolean {
    if (this.width <= 0 || this.height <= 0) return true;
    return x + pad >= view.x0 && x - pad <= view.x1 && y + pad >= view.y0 && y - pad <= view.y1;
  }

  // ---- input -------------------------------------------------------------

  private toWorld(clientX: number, clientY: number) {
    const rect = this.canvas.getBoundingClientRect();
    const sx = clientX - rect.left;
    const sy = clientY - rect.top;
    return { sx, sy, x: (sx - this.transform.x) / this.transform.k, y: (sy - this.transform.y) / this.transform.k };
  }

  private hitTest(x: number, y: number): SimNode | null {
    for (let i = this.nodes.length - 1; i >= 0; i -= 1) {
      const n = this.nodes[i];
      // Filtered nodes are drawn as ghosts and must not steal the click.
      if (!this.view.visible.has(n.id) && n.id !== this.view.selectedId) continue;
      const r = n.radius + 5;
      const p = this.place(n);
      if ((p.x - x) ** 2 + (p.y - y) ** 2 <= r * r) return n;
    }
    return null;
  }

  private takeOverViewport(): void {
    this.userInteracted = true;
    this.bursting = false;
    this.pendingFocus = null;
    this.followUntil = 0;
    this.animation = null;
  }

  private onPointerDown = (event: PointerEvent) => {
    this.canvas.setPointerCapture(event.pointerId);
    this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    this.takeOverViewport();

    if (this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      const rect = this.canvas.getBoundingClientRect();
      this.gesture = {
        kind: 'pinch',
        distance: Math.hypot(a.x - b.x, a.y - b.y),
        origin: { ...this.transform },
        center: { x: (a.x + b.x) / 2 - rect.left, y: (a.y + b.y) / 2 - rect.top },
      };
      return;
    }
    const { sx, sy, x, y } = this.toWorld(event.clientX, event.clientY);
    const node = this.hitTest(x, y);
    this.gesture = node
      ? { kind: 'drag', node, moved: false }
      : { kind: 'pan', startX: sx, startY: sy, origin: { ...this.transform }, moved: false };
  };

  private onPointerMove = (event: PointerEvent) => {
    if (this.pointers.has(event.pointerId)) this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const { sx, sy, x, y } = this.toWorld(event.clientX, event.clientY);
    const gesture = this.gesture;

    if (gesture?.kind === 'pinch' && this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      const k = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, gesture.origin.k * (Math.hypot(a.x - b.x, a.y - b.y) / gesture.distance)));
      const { center, origin } = gesture;
      this.transform = {
        k,
        x: center.x - ((center.x - origin.x) / origin.k) * k,
        y: center.y - ((center.y - origin.y) / origin.k) * k,
      };
      this.requestDraw();
    } else if (gesture?.kind === 'drag') {
      if (!gesture.moved) {
        gesture.moved = true;
        // The pointer grabbed the drawn position. Put that back into the layout
        // before the forces run, or the node jumps by the current sway.
        this.bakeDrift();
        this.settled = false;
        if (this.reduceMotion) this.sim?.stop();
        // A settled sim sits under alphaMin. Restarting alone climbs toward 0.3
        // from nearly zero, so the neighbours barely follow the pointer.
        else this.sim?.alphaTarget(0.3)?.alpha(0.3)?.restart();
      }
      gesture.node.fx = x;
      gesture.node.fy = y;
      if (this.reduceMotion) Object.assign(gesture.node, { x, y });
      this.requestDraw();
    } else if (gesture?.kind === 'pan') {
      if (Math.hypot(sx - gesture.startX, sy - gesture.startY) > 3) gesture.moved = true;
      this.transform = { ...gesture.origin, x: gesture.origin.x + sx - gesture.startX, y: gesture.origin.y + sy - gesture.startY };
      this.requestDraw();
    } else if (event.pointerType === 'mouse') {
      this.setHover(this.hitTest(x, y)?.id ?? null);
    }
  };

  private onPointerUp = (event: PointerEvent) => {
    this.pointers.delete(event.pointerId);
    const gesture = this.gesture;
    this.gesture = null;
    if (gesture?.kind === 'drag') {
      if (gesture.moved && this.reduceMotion) {
        // One hot burst while the dropped node stays pinned, then stop.
        // Cooling alpha during these ticks leaves the neighbours short of the drop.
        this.sim?.stop();
        this.sim?.alphaTarget(0.3)?.alpha(0.3);
        for (let i = 0; i < 60; i += 1) this.sim?.tick();
        this.sim?.alphaTarget(0);
        this.settled = true;
        this.coasting = false;
        this.requestDraw();
      }
      gesture.node.fx = undefined;
      gesture.node.fy = undefined;
      if (gesture.moved && !this.reduceMotion) this.sim?.alphaTarget(0);
      else if (!gesture.moved) {
        this.callbacks.onSelect(gesture.node.id);
        this.focusNode(gesture.node.id, 1000);
      }
    } else if (gesture?.kind === 'pan' && !gesture.moved) {
      this.callbacks.onSelect(null);
    }
  };

  private onPointerLeave = () => {
    if (!this.gesture) this.setHover(null);
  };

  private onWheel = (event: WheelEvent) => {
    event.preventDefault();
    this.takeOverViewport();
    const { sx, sy } = this.toWorld(event.clientX, event.clientY);
    const k = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, this.transform.k * Math.exp(-event.deltaY * 0.002)));
    const ratio = k / this.transform.k;
    this.transform = { k, x: sx - (sx - this.transform.x) * ratio, y: sy - (sy - this.transform.y) * ratio };
    this.requestDraw();
  };

  private setHover(id: number | null): void {
    if (id === this.hoverId) return;
    this.hoverId = id;
    this.linkStylesReady = false;
    this.canvas.style.cursor = id == null ? '' : 'pointer';
    this.callbacks.onHover(id);
    this.requestDraw();
  }

  // ---- drawing -----------------------------------------------------------

  private requestDraw(immediate = true): void {
    if (immediate) this.paintNow = true;
    if (!this.frame) this.frame = requestAnimationFrame(this.draw);
  }

  /** Seconds since load, not counting time the tab was hidden. */
  private motionTime(now = performance.now()): number {
    return now / 1000 - this.paused;
  }

  /** Sway at time t, in world pixels. Independent of the stored layout. */
  private sway(node: SimNode, t: number): { x: number; y: number } {
    return {
      x: Math.sin(t * node.freq + node.phase) * DRIFT_PX,
      y: Math.cos(t * node.freq * 0.83 + node.phase * 1.7) * DRIFT_PX,
    };
  }

  /**
   * Where a node is drawn. After the simulation ends this is the settled
   * position plus a sway that is zero at that moment, so the first coasting
   * frame does not pop.
   */
  private place(node: SimNode, t = this.motionTime()): { x: number; y: number } {
    const x = node.x ?? 0;
    const y = node.y ?? 0;
    if (!this.coasting) return { x, y };
    const now = this.sway(node, t);
    const origin = this.sway(node, this.coastT0);
    return { x: x + now.x - origin.x, y: y + now.y - origin.y };
  }

  /** Write the drawn position into the layout before a drag reheats the forces. */
  private bakeDrift(): void {
    if (!this.coasting) return;
    const t = this.motionTime();
    for (const node of this.nodes) {
      const p = this.place(node, t);
      node.x = p.x;
      node.y = p.y;
      node.vx = 0;
      node.vy = 0;
    }
    this.coasting = false;
  }

  private draw = (now: number) => {
    this.frame = 0;
    const painting =
      this.paintNow ||
      !this.coasting ||
      this.animation != null ||
      this.nodesFading ||
      (!this.userInteracted && now < this.followUntil) ||
      now - this.lastDraw >= IDLE_FRAME_MS;
    this.paintNow = false;
    if (!painting) {
      this.requestDraw(false);
      return;
    }
    this.lastDraw = now;
    if (this.animation) {
      const { from, to, start, duration } = this.animation;
      const t = Math.min(1, (now - start) / duration);
      const e = easeInOut(t);
      this.transform = { k: from.k + (to.k - from.k) * e, x: from.x + (to.x - from.x) * e, y: from.y + (to.y - from.y) * e };
      if (t >= 1) this.animation = null;
      else this.requestDraw();
    } else if (!this.userInteracted && now < this.followUntil) {
      // Glide the camera after the unfolding layout instead of snapping to it.
      const target = this.fitTransform();
      if (target) {
        const ease = 0.08;
        this.transform = {
          k: this.transform.k + (target.k - this.transform.k) * ease,
          x: this.transform.x + (target.x - this.transform.x) * ease,
          y: this.transform.y + (target.y - this.transform.y) * ease,
        };
      }
      this.requestDraw();
    } else if (this.bursting && this.followUntil && now >= this.followUntil) {
      // Keep following until the layout actually rests; the follow window is a cap,
      // not the moment to freeze a half-grown graph.
      if (!this.settled) {
        this.followUntil = now + 1500;
        this.requestDraw();
      } else {
        this.bursting = false;
        this.followUntil = 0;
        if (this.pendingFocus) {
          const { id, duration } = this.pendingFocus;
          this.pendingFocus = null;
          this.focusNode(id, duration || 800);
        } else {
          this.fitTo(this.reduceMotion ? 0 : 500);
        }
      }
    }

    const dark = document.documentElement.classList.contains('dark');
    this.ink = dark
      ? { labelBg: 'rgba(17, 17, 17, 0.92)', labelText: '#ededed', hover: '#ededed', ring: 'rgba(255, 255, 255, 0.1)' }
      : { labelBg: COLORS.LABEL_BG, labelText: COLORS.LABEL_TEXT, hover: COLORS.NODE_HOVER, ring: 'rgba(0, 0, 0, 0.08)' };

    const { ctx, transform: tf } = this;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.setTransform(this.dpr * tf.k, 0, 0, this.dpr * tf.k, this.dpr * tf.x, this.dpr * tf.y);

    const hover = this.hoverId == null ? null : (this.nodeById.get(this.hoverId) ?? null);
    const focus: FocusContext = {
      selectedId: this.view.selectedId ?? undefined,
      hover,
      neighbors: this.view.neighbors,
      visible: this.view.visible,
    };

    // Ease every node's opacity toward what the focus rules ask for.
    let fading = false;
    for (const node of this.nodes) {
      const target = this.targetAlpha(node, focus);
      if (this.reduceMotion || Math.abs(node.fade - target) < 0.01) node.fade = target;
      else {
        node.fade += (target - node.fade) * FADE_RATE;
        fading = true;
      }
    }
    this.nodesFading = fading;
    if (fading) this.requestDraw();

    const view = this.worldView();
    const t = this.motionTime(now);
    this.drawHulls(focus, tf.k, t);
    this.drawLinks({ ...focus, showTrajectories: this.view.showTrajectories, clusterOf: this.clusterOf }, view, t);
    const emphasized: SimNode[] = [];
    for (const node of this.nodes) {
      if (node.id === focus.selectedId || node.id === hover?.id) emphasized.push(node);
      else this.drawNode(node, focus, tf.k, view, t);
    }
    emphasized.forEach((node) => this.drawNode(node, focus, tf.k, view, t));
    if (this.coasting) this.requestDraw(false);
  };

  private targetAlpha(node: SimNode, focus: FocusContext): number {
    const emphasized = node.id === focus.selectedId || node.id === focus.hover?.id;
    if (!focus.visible.has(node.id) && !emphasized) return 0.25;
    if (nodeDimmed(node, focus) && !emphasized) return UNFOCUSED_NODE_ALPHA;
    return 1;
  }

  private drawHulls(focus: FocusContext, scale: number, t: number): void {
    const clusterId = focusClusterId(focus);
    const cluster = clusterId == null ? undefined : this.clusters.get(clusterId);
    if (!cluster) return;
    const members = this.nodes.filter((n) => n.cluster_id === clusterId && focus.visible.has(n.id));
    if (members.length < 3) return;
    const placed = members.map((n) => this.place(n, t));
    const hull = convexHull(placed);
    if (hull.length < 3) return;
    const { ctx } = this;
    ctx.beginPath();
    hull.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
    ctx.closePath();
    ctx.fillStyle = `${cluster.color}10`;
    ctx.fill();
    ctx.strokeStyle = `${cluster.color}30`;
    ctx.lineWidth = 1 / scale;
    ctx.stroke();
    if (scale > 0.5 && cluster.name) {
      const cx = placed.reduce((s, p) => s + p.x, 0) / placed.length;
      const cy = placed.reduce((s, p) => s + p.y, 0) / placed.length;
      ctx.font = `600 ${Math.max(14 / scale, 10)}px ${FONT}`;
      ctx.fillStyle = `${cluster.color}60`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(cluster.name, cx, cy);
    }
  }

  private drawLinks(context: LinkContext, view: WorldView, t: number): void {
    const { ctx } = this;
    if (!this.linkStylesReady) {
      for (const link of this.links) {
        const kind = classifyLink(link, context, this.overview.has(link));
        link.stroke = LINK_STYLE[kind];
      }
      this.linkStylesReady = true;
    }
    // Link colours already carry alpha. One stroke() of many curves would
    // composite them as a single layer and lighten every crossing.
    for (const link of this.links) {
      const style = link.stroke;
      if (!style?.width) continue;
      const a = this.place(link.source, t);
      const b = this.place(link.target, t);
      if (!this.linkOnScreen(a.x, a.y, b.x, b.y, view)) continue;
      const alpha = Math.min(link.source.fade, link.target.fade, 1) ** 0.5;
      if (alpha <= 0) continue;
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = style.color;
      ctx.lineWidth = style.width;
      ctx.beginPath();
      this.curve(a.x, a.y, b.x, b.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  /** Same gentle curve react-force-graph drew with linkCurvature 0.1. */
  private curve(x1: number, y1: number, x2: number, y2: number): void {
    const { ctx } = this;
    const offset = Math.hypot(x2 - x1, y2 - y1) * 0.1;
    const angle = Math.atan2(y2 - y1, x2 - x1) - Math.PI / 2;
    ctx.moveTo(x1, y1);
    ctx.quadraticCurveTo((x1 + x2) / 2 + offset * Math.cos(angle), (y1 + y2) / 2 + offset * Math.sin(angle), x2, y2);
  }

  private linkOnScreen(x1: number, y1: number, x2: number, y2: number, view: WorldView): boolean {
    if (this.width <= 0 || this.height <= 0) return true;
    const bulge = Math.hypot(x2 - x1, y2 - y1) * 0.1;
    return (
      Math.max(x1, x2) + bulge >= view.x0 &&
      Math.min(x1, x2) - bulge <= view.x1 &&
      Math.max(y1, y2) + bulge >= view.y0 &&
      Math.min(y1, y2) - bulge <= view.y1
    );
  }

  private drawNode(node: SimNode, focus: FocusContext, scale: number, view: WorldView, t: number): void {
    const { ctx } = this;
    const emphasized = node.id === focus.selectedId || node.id === focus.hover?.id;
    // The label font floors at 8 world pixels, so a pad that shrinks with zoom
    // hides names whose centres have just left the viewport.
    const fontSize = Math.max(10 / scale, 8);
    const labeled = showLabel(node.id, focus) && (emphasized || scale >= NEIGHBOR_LABEL_ZOOM);
    const pad = labeled ? node.radius + fontSize * (node.name.length + 1) + 4 / scale : node.radius + 4 / scale;
    const { x, y } = this.place(node, t);
    if (!this.onScreen(x, y, pad, view)) return;
    const visible = focus.visible.has(node.id);
    const isSelected = focus.selectedId === node.id;
    const isHovered = focus.hover?.id === node.id;
    const dimmed = nodeDimmed(node, focus);
    // Colour without the baked-in dim alpha: opacity comes from the eased fade.
    const color = dimmed ? COLORS.NODE_DEFAULT : nodeColor(node, focus);

    ctx.globalAlpha = node.fade;

    // Filtered-out nodes are uniform ghosts; an avatar would make them read as visible.
    const ghost = !visible && !isSelected && !isHovered;
    const sprite = this.view.hqRendering && node.avatar && !ghost ? this.avatarSprite(node.avatar) : null;
    if (sprite) {
      ctx.drawImage(sprite, x - node.radius, y - node.radius, node.radius * 2, node.radius * 2);
      // One hairline keeps hundreds of unrelated logos reading as one object.
      if (!isSelected && !isHovered) {
        ctx.beginPath();
        ctx.arc(x, y, node.radius, 0, Math.PI * 2);
        ctx.strokeStyle = this.ink.ring;
        ctx.lineWidth = 1 / scale;
        ctx.stroke();
      }
    } else {
      ctx.beginPath();
      ctx.arc(x, y, !visible && !isSelected && !isHovered ? node.radius * 0.8 : node.radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }

    if ((showClusterHue(node, focus) && !dimmed && !isSelected && !isHovered) || isSelected || isHovered) {
      ctx.beginPath();
      ctx.arc(x, y, node.radius + 1.5 / scale, 0, Math.PI * 2);
      ctx.strokeStyle = isSelected ? COLORS.NODE_SELECTED : isHovered ? this.ink.hover : color;
      ctx.lineWidth = (isSelected || isHovered ? 1.5 : 1.25) / scale;
      ctx.stroke();
    }

    if (labeled) {
      const padding = 3 / scale;
      const labelY = y + node.radius + fontSize + 2 / scale;
      ctx.font = `500 ${fontSize}px ${FONT}`;
      const textWidth = ctx.measureText(node.name).width;
      ctx.fillStyle = this.ink.labelBg;
      ctx.fillRect(x - textWidth / 2 - padding, labelY - fontSize + 2 / scale, textWidth + padding * 2, fontSize + padding);
      ctx.fillStyle = this.ink.labelText;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.name, x, labelY - fontSize / 2 + padding);
    }
    ctx.globalAlpha = 1;
  }
}
