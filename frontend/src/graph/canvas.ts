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
  nodeColor,
  nodeDimmed,
  nodeRadius,
  showClusterHue,
  showLabel,
  type FocusContext,
  type LinkContext,
} from './style';

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
/** Simulation heat kept after settling; enough for the drift to read as motion. */
const DRIFT_ALPHA = 0.015;
const DRIFT_STRENGTH = 0.02;
/** Once settled, redraw at ~30fps: the drift is slow and this halves the cost. */
const IDLE_FRAME_MS = 33;
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

/** Slow, independent sway per node. Ignores alpha so it survives settling. */
const drift = (amplitude: number) => {
  let nodes: SimNode[] = [];
  const force = () => {
    const t = performance.now() / 1000;
    for (const node of nodes) {
      if (node.fx != null) continue;
      node.vx! += Math.sin(t * node.freq + node.phase) * amplitude;
      node.vy! += Math.cos(t * node.freq * 0.83 + node.phase * 1.7) * amplitude;
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
 * Nodes burst out of the centre, one wedge per cluster, settle into
 * communities, then keep drifting gently. The viewport follows the burst until
 * the user pans or zooms. Drift pauses in background tabs and is off under
 * reduced motion, where the layout is computed up front and drawn once.
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
  private transform: Transform = { k: 1, x: 0, y: 0 };
  private animation: { from: Transform; to: Transform; start: number; duration: number } | null = null;
  private width = 0;
  private height = 0;
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
  /** Theme-dependent ink for labels and the hover ring, refreshed every frame (cheap class check). */
  private ink: { labelBg: string; labelText: string; hover: string } = { labelBg: COLORS.LABEL_BG, labelText: COLORS.LABEL_TEXT, hover: COLORS.NODE_HOVER };

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
    this.sim?.stop();
    window.clearTimeout(this.startTimer);
    cancelAnimationFrame(this.frame);
    this.resizeObserver.disconnect();
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.canvas.remove();
  }

  setData(nodes: GraphNode[], edges: GraphEdge[], clusters: ClusterInfo[]): void {
    this.sim?.stop();
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
      .alphaTarget(this.reduceMotion ? 0 : DRIFT_ALPHA)
      .velocityDecay(0.4)
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
    if (!this.reduceMotion) sim.force('drift', drift(DRIFT_STRENGTH));
    sim.on('tick', () => {
      const now = performance.now();
      if (sim.alpha() > 0.05 || now - this.lastDraw >= IDLE_FRAME_MS) this.requestDraw();
    });
    return sim;
  }

  setView(view: GraphView): void {
    this.view = view;
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
    this.animateTo(
      { k: FOCUS_ZOOM, x: this.width / 2 - node.x! * FOCUS_ZOOM, y: this.height / 2 - node.y! * FOCUS_ZOOM },
      duration,
    );
  }

  private fitTransform(): Transform | null {
    if (!this.nodes.length || !this.width || !this.height) return null;
    let [minX, maxX, minY, maxY] = [Infinity, -Infinity, Infinity, -Infinity];
    for (const n of this.nodes) {
      minX = Math.min(minX, n.x! - n.radius);
      maxX = Math.max(maxX, n.x! + n.radius);
      minY = Math.min(minY, n.y! - n.radius);
      maxY = Math.max(maxY, n.y! + n.radius);
    }
    const k = Math.min(
      MAX_ZOOM,
      Math.max(
        MIN_ZOOM,
        Math.min((this.width - FIT_PADDING * 2) / Math.max(maxX - minX, 1), (this.height - FIT_PADDING * 2) / Math.max(maxY - minY, 1)),
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
    const previous = { width: this.width, height: this.height };
    this.measure();
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(this.width * this.dpr));
    this.canvas.height = Math.max(1, Math.round(this.height * this.dpr));
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    // Keep the same world point centred when panels open or the window resizes.
    this.transform = {
      ...this.transform,
      x: this.transform.x + (this.width - previous.width) / 2,
      y: this.transform.y + (this.height - previous.height) / 2,
    };
    this.requestDraw();
  }

  private onVisibility = () => {
    if (!this.sim || this.reduceMotion) return;
    if (document.hidden) this.sim.stop();
    else this.sim.restart();
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
      const r = n.radius + 5;
      if ((n.x! - x) ** 2 + (n.y! - y) ** 2 <= r * r) return n;
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
        this.sim?.alphaTarget(0.3).restart();
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
      gesture.node.fx = undefined;
      gesture.node.fy = undefined;
      if (gesture.moved) {
        if (this.reduceMotion) {
          // No live simulation: let the neighbours catch up once, then stop.
          this.sim?.alpha(0.3);
          for (let i = 0; i < 60; i += 1) this.sim?.tick();
          this.requestDraw();
        } else {
          this.sim?.alphaTarget(DRIFT_ALPHA);
        }
      } else {
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
    this.canvas.style.cursor = id == null ? '' : 'pointer';
    this.callbacks.onHover(id);
    this.requestDraw();
  }

  // ---- drawing -----------------------------------------------------------

  private requestDraw(): void {
    if (!this.frame) this.frame = requestAnimationFrame(this.draw);
  }

  private draw = (now: number) => {
    this.frame = 0;
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
      this.bursting = false;
      if (this.pendingFocus) this.focusNode(this.pendingFocus.id, this.pendingFocus.duration || 800);
      this.pendingFocus = null;
    }

    const dark = document.documentElement.classList.contains('dark');
    this.ink = dark
      ? { labelBg: 'rgba(17, 17, 17, 0.92)', labelText: '#ededed', hover: '#ededed' }
      : { labelBg: COLORS.LABEL_BG, labelText: COLORS.LABEL_TEXT, hover: COLORS.NODE_HOVER };

    const { ctx, transform: tf } = this;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.setTransform(this.dpr * tf.k, 0, 0, this.dpr * tf.k, this.dpr * tf.x, this.dpr * tf.y);

    const hover = this.hoverId == null ? null : (this.nodeById.get(this.hoverId) ?? null);
    const selected = this.view.selectedId == null ? undefined : this.nodeById.get(this.view.selectedId);
    const focus: FocusContext = {
      selectedId: this.view.selectedId ?? undefined,
      selectedClusterId: selected?.cluster_id,
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
    if (fading) this.requestDraw();

    this.drawHulls(focus, tf.k);
    this.drawLinks({ ...focus, showTrajectories: this.view.showTrajectories, clusterOf: this.clusterOf });
    const emphasized: SimNode[] = [];
    for (const node of this.nodes) {
      if (node.id === focus.selectedId || node.id === hover?.id) emphasized.push(node);
      else this.drawNode(node, focus, tf.k);
    }
    emphasized.forEach((node) => this.drawNode(node, focus, tf.k));
  };

  private targetAlpha(node: SimNode, focus: FocusContext): number {
    const emphasized = node.id === focus.selectedId || node.id === focus.hover?.id;
    if (!focus.visible.has(node.id) && !emphasized) return 0.25;
    if (nodeDimmed(node, focus) && !emphasized) return UNFOCUSED_NODE_ALPHA;
    return 1;
  }

  private drawHulls(focus: FocusContext, scale: number): void {
    const focused = new Set([focus.selectedClusterId, focus.hover?.cluster_id].filter((id) => id != null));
    for (const clusterId of focused) {
      const cluster = this.clusters.get(clusterId!);
      const members = this.nodes.filter((n) => n.cluster_id === clusterId && focus.visible.has(n.id));
      if (!cluster || members.length < 3) continue;
      const hull = convexHull(members.map((n) => ({ x: n.x!, y: n.y! })));
      if (hull.length < 3) continue;
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
        const cx = members.reduce((s, n) => s + n.x!, 0) / members.length;
        const cy = members.reduce((s, n) => s + n.y!, 0) / members.length;
        ctx.font = `600 ${Math.max(14 / scale, 10)}px ${FONT}`;
        ctx.fillStyle = `${cluster.color}60`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(cluster.name, cx, cy);
      }
    }
  }

  private drawLinks(context: LinkContext): void {
    const { ctx } = this;
    for (const link of this.links) {
      const style = LINK_STYLE[classifyLink(link, context, this.overview.has(link))];
      if (!style.width) continue;
      const { x: x1, y: y1 } = link.source as { x: number; y: number };
      const { x: x2, y: y2 } = link.target as { x: number; y: number };
      // Same gentle curve react-force-graph drew with linkCurvature 0.1.
      const offset = Math.hypot(x2 - x1, y2 - y1) * 0.1;
      const angle = Math.atan2(y2 - y1, x2 - x1) - Math.PI / 2;
      ctx.globalAlpha = Math.min(link.source.fade, link.target.fade, 1) ** 0.5;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.quadraticCurveTo((x1 + x2) / 2 + offset * Math.cos(angle), (y1 + y2) / 2 + offset * Math.sin(angle), x2, y2);
      ctx.strokeStyle = style.color;
      ctx.lineWidth = style.width;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  private drawNode(node: SimNode, focus: FocusContext, scale: number): void {
    const { ctx } = this;
    const x = node.x!;
    const y = node.y!;
    const visible = focus.visible.has(node.id);
    const isSelected = focus.selectedId === node.id;
    const isHovered = focus.hover?.id === node.id;
    const dimmed = nodeDimmed(node, focus);
    // Colour without the baked-in dim alpha: opacity comes from the eased fade.
    const color = dimmed ? COLORS.NODE_DEFAULT : nodeColor(node, focus);

    ctx.save();
    ctx.globalAlpha = node.fade;

    // Filtered-out nodes are uniform ghosts; an avatar would make them read as visible.
    const ghost = !visible && !isSelected && !isHovered;
    const img = this.view.hqRendering && node.avatar && !ghost ? this.image(node.avatar) : null;
    if (img) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, node.radius, 0, Math.PI * 2);
      ctx.clip();
      ctx.drawImage(img, x - node.radius, y - node.radius, node.radius * 2, node.radius * 2);
      ctx.restore();
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

    const emphasized = node.id === focus.selectedId || node.id === focus.hover?.id;
    if (showLabel(node.id, focus) && (emphasized || scale >= NEIGHBOR_LABEL_ZOOM)) {
      const fontSize = Math.max(10 / scale, 8);
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
    ctx.restore();
  }
}
