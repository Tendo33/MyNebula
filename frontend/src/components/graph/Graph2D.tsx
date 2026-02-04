import React, { useMemo, useRef, useCallback, useEffect, useState } from 'react';
import ForceGraph2D, { ForceGraphMethods, NodeObject, LinkObject } from 'react-force-graph-2d';
import { useResizeObserver } from '../../hooks/useResizeObserver';
import { useTranslation } from 'react-i18next';
import { useGraph, useNodeNeighbors } from '../../contexts/GraphContext';
import { ClusterInfo } from '../../types';

// ============================================================================
// Types
// ============================================================================

interface ProcessedNode extends NodeObject {
  id: number;
  name: string;
  full_name: string;
  description?: string;
  language?: string;
  cluster_id?: number;
  color: string;
  size: number;
  stargazers_count: number;
  // Force-graph will add x, y, vx, vy
  x?: number;
  y?: number;
  fx?: number; // Fixed position
  fy?: number;
}

interface ProcessedLink extends LinkObject {
  source: number | ProcessedNode;
  target: number | ProcessedNode;
  weight: number;
}

interface ProcessedData {
  nodes: ProcessedNode[];
  links: ProcessedLink[];
}

// ============================================================================
// Constants
// ============================================================================

const COLORS = {
  NODE_DEFAULT: '#6B7280',      // Gray
  NODE_HOVER: '#8B5CF6',        // Purple
  NODE_SELECTED: '#3B82F6',     // Blue
  NODE_NEIGHBOR: '#60A5FA',     // Light Blue
  NODE_DIM: 'rgba(107, 114, 128, 0.3)',
  LINK_DEFAULT: 'rgba(156, 163, 175, 0.4)',
  LINK_ACTIVE: 'rgba(139, 92, 246, 0.6)',
  LINK_DIM: 'rgba(156, 163, 175, 0.1)',
  CLUSTER_BG: 'rgba(0, 0, 0, 0.03)',
  LABEL_BG: 'rgba(255, 255, 255, 0.9)',
  LABEL_TEXT: '#1F2937',
};

const NODE_BASE_SIZE = 5;
const NODE_MAX_SIZE = 30;
const ZOOM_TO_FIT_PADDING = 50;

// ============================================================================
// Utility Functions
// ============================================================================

/** Get node ID from either number or node object (force-graph mutates links) */
const getNodeId = (node: number | ProcessedNode): number => {
  return typeof node === 'object' ? node.id : node;
};

/** Calculate node radius based on stars (logarithmic scale) */
const calculateNodeRadius = (stars: number): number => {
  const base = Math.log10(Math.max(stars, 1) + 1) * NODE_BASE_SIZE;
  return Math.min(Math.max(base, NODE_BASE_SIZE), NODE_MAX_SIZE);
};

/** Compute convex hull of points using Graham scan */
const computeConvexHull = (points: { x: number; y: number }[]): { x: number; y: number }[] => {
  if (points.length < 3) return points;

  // Find the bottom-most point (or left most point in case of tie)
  let start = 0;
  for (let i = 1; i < points.length; i++) {
    if (points[i].y < points[start].y ||
        (points[i].y === points[start].y && points[i].x < points[start].x)) {
      start = i;
    }
  }

  // Swap start to first position
  [points[0], points[start]] = [points[start], points[0]];
  const pivot = points[0];

  // Sort by polar angle
  points.sort((a, b) => {
    if (a === pivot) return -1;
    if (b === pivot) return 1;

    const angleA = Math.atan2(a.y - pivot.y, a.x - pivot.x);
    const angleB = Math.atan2(b.y - pivot.y, b.x - pivot.x);

    if (angleA !== angleB) return angleA - angleB;

    // If angles are same, sort by distance
    const distA = (a.x - pivot.x) ** 2 + (a.y - pivot.y) ** 2;
    const distB = (b.x - pivot.x) ** 2 + (b.y - pivot.y) ** 2;
    return distA - distB;
  });

  // Build hull
  const hull: { x: number; y: number }[] = [];
  for (const p of points) {
    while (hull.length >= 2) {
      const a = hull[hull.length - 2];
      const b = hull[hull.length - 1];
      const cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
      if (cross <= 0) hull.pop();
      else break;
    }
    hull.push(p);
  }

  return hull;
};

// ============================================================================
// Component
// ============================================================================

const Graph2D: React.FC = () => {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<ForceGraphMethods | undefined>(undefined);
  const { width, height } = useResizeObserver(containerRef);

  // Global state
  const {
    filteredData,
    rawData,
    selectedNode,
    hoveredNode,
    setSelectedNode,
    setHoveredNode,
    loading,
  } = useGraph();

  // Local state for graph-specific interactions
  const [localHoverNode, setLocalHoverNode] = useState<ProcessedNode | null>(null);
  const activeHoverNode = localHoverNode || (hoveredNode as ProcessedNode | null);

  // Get neighbors of hovered node for highlighting
  const hoverNeighbors = useNodeNeighbors(activeHoverNode?.id);

  // Process data for force-graph
  const processedData = useMemo((): ProcessedData => {
    if (!filteredData || filteredData.nodes.length === 0) {
      return { nodes: [], links: [] };
    }

    return {
      nodes: filteredData.nodes.map(n => ({
        id: n.id,
        name: n.name,
        full_name: n.full_name,
        description: n.description,
        language: n.language,
        cluster_id: n.cluster_id,
        color: n.color || COLORS.NODE_DEFAULT,
        size: n.size,
        stargazers_count: n.stargazers_count,
        // Use pre-computed positions if available (from clustering)
        x: n.x * 50, // Scale up for better spread
        y: n.y * 50,
      })),
      links: filteredData.edges.map(e => ({
        source: typeof e.source === 'object' ? e.source.id : e.source,
        target: typeof e.target === 'object' ? e.target.id : e.target,
        weight: e.weight,
      })),
    };
  }, [filteredData]);

  // Group nodes by cluster for hull drawing
  const clusterGroups = useMemo(() => {
    if (!rawData) return new Map<number, ClusterInfo>();

    const groups = new Map<number, ClusterInfo>();
    rawData.clusters.forEach(cluster => {
      groups.set(cluster.id, cluster);
    });
    return groups;
  }, [rawData]);

  // Configure forces after graph is created
  useEffect(() => {
    if (!graphRef.current) return;

    const fg = graphRef.current;

    // Configure link force
    fg.d3Force('link')
      ?.distance((link: any) => {
        // Shorter distance for same cluster, longer for different
        const sourceCluster = typeof link.source === 'object' ? link.source.cluster_id : undefined;
        const targetCluster = typeof link.target === 'object' ? link.target.cluster_id : undefined;
        return sourceCluster === targetCluster ? 50 : 100;
      })
      .strength((link: any) => {
        // Stronger links within same cluster
        const sourceCluster = typeof link.source === 'object' ? link.source.cluster_id : undefined;
        const targetCluster = typeof link.target === 'object' ? link.target.cluster_id : undefined;
        return sourceCluster === targetCluster ? 0.5 : 0.1;
      });

    // Configure charge force (repulsion)
    fg.d3Force('charge')
      ?.strength(-150)
      .distanceMax(300);

    // Add cluster force to group nodes by cluster
    // Using a custom force that pulls nodes toward their cluster center
    const clusterForce = (alpha: number) => {
      const clusterCenters = new Map<number, { x: number; y: number; count: number }>();

      // Calculate cluster centers
      processedData.nodes.forEach(node => {
        if (node.cluster_id !== undefined && node.x !== undefined && node.y !== undefined) {
          const current = clusterCenters.get(node.cluster_id) || { x: 0, y: 0, count: 0 };
          current.x += node.x;
          current.y += node.y;
          current.count += 1;
          clusterCenters.set(node.cluster_id, current);
        }
      });

      // Normalize centers
      clusterCenters.forEach(center => {
        center.x /= center.count;
        center.y /= center.count;
      });

      // Apply force toward cluster center
      processedData.nodes.forEach((node: any) => {
        if (node.cluster_id !== undefined) {
          const center = clusterCenters.get(node.cluster_id);
          if (center && node.x !== undefined && node.y !== undefined) {
            const k = alpha * 0.1; // Strength of clustering
            node.vx = (node.vx || 0) + (center.x - node.x) * k;
            node.vy = (node.vy || 0) + (center.y - node.y) * k;
          }
        }
      });
    };

    // Register custom force
    fg.d3Force('cluster', clusterForce);

  }, [processedData.nodes]);

  // Zoom to fit after initial render
  useEffect(() => {
    if (graphRef.current && processedData.nodes.length > 0) {
      // Wait for simulation to stabilize
      const timer = setTimeout(() => {
        graphRef.current?.zoomToFit(400, ZOOM_TO_FIT_PADDING);
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [processedData.nodes.length]);

  // Get node color based on state
  const getNodeColor = useCallback((node: ProcessedNode): string => {
    // Selected node
    if (selectedNode && selectedNode.id === node.id) {
      return COLORS.NODE_SELECTED;
    }

    // No hover - use cluster color
    if (!activeHoverNode) {
      return node.color || COLORS.NODE_DEFAULT;
    }

    // Hovered node
    if (node.id === activeHoverNode.id) {
      return COLORS.NODE_HOVER;
    }

    // Neighbor of hovered node
    if (hoverNeighbors.has(node.id)) {
      return COLORS.NODE_NEIGHBOR;
    }

    // Same cluster as hovered
    if (activeHoverNode.cluster_id !== undefined &&
        node.cluster_id === activeHoverNode.cluster_id) {
      return node.color || COLORS.NODE_DEFAULT;
    }

    // Dim other nodes
    return COLORS.NODE_DIM;
  }, [selectedNode, activeHoverNode, hoverNeighbors]);

  // Get link color based on state
  const getLinkColor = useCallback((link: ProcessedLink): string => {
    if (!activeHoverNode) return COLORS.LINK_DEFAULT;

    const sourceId = getNodeId(link.source);
    const targetId = getNodeId(link.target);

    // Highlight links connected to hovered node
    if (sourceId === activeHoverNode.id || targetId === activeHoverNode.id) {
      return COLORS.LINK_ACTIVE;
    }

    return COLORS.LINK_DIM;
  }, [activeHoverNode]);

  // Get link width based on state
  const getLinkWidth = useCallback((link: ProcessedLink): number => {
    if (!activeHoverNode) return 1;

    const sourceId = getNodeId(link.source);
    const targetId = getNodeId(link.target);

    if (sourceId === activeHoverNode.id || targetId === activeHoverNode.id) {
      return 2;
    }

    return 0.5;
  }, [activeHoverNode]);

  // Custom node painting
  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const { x, y, name, stargazers_count } = node as ProcessedNode;
    if (x === undefined || y === undefined) return;

    const radius = calculateNodeRadius(stargazers_count);
    const color = getNodeColor(node);
    const isHovered = activeHoverNode?.id === node.id;
    const isSelected = selectedNode?.id === node.id;

    // Draw node circle
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    // Draw border for selected/hovered nodes
    if (isHovered || isSelected) {
      ctx.strokeStyle = isSelected ? COLORS.NODE_SELECTED : COLORS.NODE_HOVER;
      ctx.lineWidth = 2 / globalScale;
      ctx.stroke();

      // Draw outer glow
      ctx.beginPath();
      ctx.arc(x, y, radius + 4 / globalScale, 0, 2 * Math.PI);
      ctx.strokeStyle = isSelected
        ? 'rgba(59, 130, 246, 0.3)'
        : 'rgba(139, 92, 246, 0.3)';
      ctx.lineWidth = 3 / globalScale;
      ctx.stroke();
    }

    // Draw label
    const fontSize = Math.max(10 / globalScale, 8);
    const showLabel = isHovered || isSelected || globalScale > 2 || radius > 15;

    if (showLabel) {
      const label = name;
      ctx.font = `${fontSize}px Inter, system-ui, sans-serif`;
      const textMetrics = ctx.measureText(label);
      const textWidth = textMetrics.width;
      const textHeight = fontSize;
      const padding = 3 / globalScale;
      const labelY = y + radius + fontSize + 2 / globalScale;

      // Background
      ctx.fillStyle = COLORS.LABEL_BG;
      ctx.fillRect(
        x - textWidth / 2 - padding,
        labelY - textHeight + 2 / globalScale,
        textWidth + padding * 2,
        textHeight + padding
      );

      // Text
      ctx.fillStyle = COLORS.LABEL_TEXT;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, x, labelY - textHeight / 2 + padding);
    }
  }, [getNodeColor, activeHoverNode, selectedNode]);

  // Node pointer area for click detection
  const paintNodeArea = useCallback((node: any, color: string, ctx: CanvasRenderingContext2D) => {
    const { x, y, stargazers_count } = node as ProcessedNode;
    if (x === undefined || y === undefined) return;

    const radius = calculateNodeRadius(stargazers_count);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius + 5, 0, 2 * Math.PI); // Slightly larger for easier clicking
    ctx.fill();
  }, []);

  // Draw cluster hulls in the background
  const drawClusterHulls = useCallback((ctx: CanvasRenderingContext2D, globalScale: number) => {
    if (!graphRef.current) return;

    // Get nodes from processed data instead of graphRef.graphData()
    const nodes = processedData.nodes;

    // Group nodes by cluster
    const nodesByCluster = new Map<number, ProcessedNode[]>();
    nodes.forEach(node => {
      if (node.cluster_id !== undefined && node.x !== undefined && node.y !== undefined) {
        const group = nodesByCluster.get(node.cluster_id) || [];
        group.push(node);
        nodesByCluster.set(node.cluster_id, group);
      }
    });

    // Draw hull for each cluster with enough nodes
    nodesByCluster.forEach((clusterNodes, clusterId) => {
      if (clusterNodes.length < 3) return;

      const cluster = clusterGroups.get(clusterId);
      if (!cluster) return;

      // Get node positions with padding
      const points = clusterNodes.map(n => ({
        x: n.x!,
        y: n.y!,
      }));

      // Compute convex hull
      const hull = computeConvexHull([...points]);
      if (hull.length < 3) return;

      // Draw filled hull with cluster color
      ctx.beginPath();
      ctx.moveTo(hull[0].x, hull[0].y);
      for (let i = 1; i < hull.length; i++) {
        ctx.lineTo(hull[i].x, hull[i].y);
      }
      ctx.closePath();

      // Parse cluster color and add transparency
      const baseColor = cluster.color || '#808080';
      ctx.fillStyle = baseColor + '10'; // Very transparent
      ctx.fill();

      ctx.strokeStyle = baseColor + '30'; // Slightly more visible border
      ctx.lineWidth = 1 / globalScale;
      ctx.stroke();

      // Draw cluster label at center
      const centerX = clusterNodes.reduce((sum, n) => sum + n.x!, 0) / clusterNodes.length;
      const centerY = clusterNodes.reduce((sum, n) => sum + n.y!, 0) / clusterNodes.length;

      if (globalScale > 0.5 && cluster.name) {
        const fontSize = Math.max(14 / globalScale, 10);
        ctx.font = `bold ${fontSize}px Inter, system-ui, sans-serif`;
        ctx.fillStyle = baseColor + '60';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(cluster.name, centerX, centerY);
      }
    });
  }, [clusterGroups, processedData.nodes]);

  // Handle node click
  const handleNodeClick = useCallback((node: any) => {
    const processedNode = node as ProcessedNode;

    // Find the full node data from filtered data
    const fullNode = filteredData?.nodes.find(n => n.id === processedNode.id);
    if (fullNode) {
      setSelectedNode(fullNode);
    }

    // Zoom to node
    if (graphRef.current && processedNode.x !== undefined && processedNode.y !== undefined) {
      graphRef.current.centerAt(processedNode.x, processedNode.y, 1000);
      graphRef.current.zoom(3, 1000);
    }
  }, [filteredData, setSelectedNode]);

  // Handle node hover
  const handleNodeHover = useCallback((node: any) => {
    setLocalHoverNode(node as ProcessedNode | null);
    setHoveredNode(node ? filteredData?.nodes.find(n => n.id === node.id) || null : null);
    document.body.style.cursor = node ? 'pointer' : 'default';
  }, [filteredData, setHoveredNode]);

  // Handle background click (deselect)
  const handleBackgroundClick = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  // Empty state
  if (!filteredData || filteredData.nodes.length === 0) {
    return (
      <div ref={containerRef} className="w-full h-full relative flex items-center justify-center bg-gray-50/50">
        <div className="text-center p-8 opacity-50">
          <div className="text-6xl mb-4 grayscale">🕸️</div>
          <h3 className="font-semibold text-lg text-text-main">
            {loading ? t('common.loading') : t('dashboard.subtitle_infinite')}
          </h3>
          {!loading && (
            <p className="text-sm text-text-muted mt-2">
              {t('graph.empty_hint')}
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full h-full relative bg-white">
      <ForceGraph2D
        ref={graphRef}
        width={width}
        height={height}
        graphData={processedData}

        // Interaction
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
        onBackgroundClick={handleBackgroundClick}
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}

        // Node rendering
        nodeCanvasObject={paintNode}
        nodePointerAreaPaint={paintNodeArea}
        nodeCanvasObjectMode={() => 'replace'}

        // Link rendering
        linkColor={getLinkColor}
        linkWidth={getLinkWidth}
        linkCurvature={0.1}
        linkDirectionalParticles={0}

        // Pre-render callback for cluster hulls
        onRenderFramePre={(ctx, globalScale) => {
          drawClusterHulls(ctx, globalScale);
        }}

        // Physics
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
        cooldownTicks={200}
        warmupTicks={100}

        // After engine stops
        onEngineStop={() => {
          if (graphRef.current) {
            graphRef.current.zoomToFit(400, ZOOM_TO_FIT_PADDING);
          }
        }}
      />

      {/* Hover info overlay */}
      {activeHoverNode && (
        <div className="absolute top-4 right-4 pointer-events-none bg-white/95 backdrop-blur-sm px-4 py-3 rounded-lg border border-border-light shadow-md z-10 max-w-xs">
          <div className="font-semibold text-sm text-text-main truncate">
            {activeHoverNode.name}
          </div>
          <div className="text-xs text-text-muted mt-1 flex items-center gap-2">
            {activeHoverNode.language && (
              <span className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-700">
                {activeHoverNode.language}
              </span>
            )}
            <span>⭐ {activeHoverNode.stargazers_count.toLocaleString()}</span>
          </div>
          {activeHoverNode.description && (
            <p className="text-xs text-text-muted mt-2 line-clamp-2">
              {activeHoverNode.description}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default Graph2D;
