import React, { useMemo, useRef, useCallback, useEffect, useState } from 'react';
import ForceGraph3D, { ForceGraphMethods } from 'react-force-graph-3d';
import * as THREE from 'three';
import { useResizeObserver } from '../../hooks/useResizeObserver';
import { useTranslation } from 'react-i18next';
import { useGraph, useNodeNeighbors } from '../../contexts/GraphContext';
import { ClusterInfo } from '../../types';
import { GraphSkeleton } from '../ui/Skeleton';

// ============================================================================
// Types
// ============================================================================

interface ProcessedNode {
  id: number;
  name: string;
  full_name: string;
  description?: string;
  language?: string;
  cluster_id?: number;
  color: string;
  size: number;
  stargazers_count: number;
  x: number;
  y: number;
  z: number;
  fx?: number;
  fy?: number;
  fz?: number;
}

interface ProcessedLink {
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
  NODE_DEFAULT: '#6B7280',
  NODE_HOVER: '#8B5CF6',
  NODE_SELECTED: '#3B82F6',
  NODE_NEIGHBOR: '#60A5FA',
  LINK_DEFAULT: 'rgba(156, 163, 175, 0.3)',
  LINK_ACTIVE: 'rgba(139, 92, 246, 0.5)',
  BACKGROUND: '#FAFAFA',
};

const NODE_BASE_SIZE = 2;
const NODE_MAX_SIZE = 12;
const POSITION_SCALE = 80; // Scale factor for pre-computed positions

// ============================================================================
// Utility Functions
// ============================================================================

/** Get node ID from either number or node object */
const getNodeId = (node: number | ProcessedNode): number => {
  return typeof node === 'object' ? node.id : node;
};

/** Calculate node size based on stars */
const calculateNodeSize = (stars: number): number => {
  const base = Math.log10(Math.max(stars, 1) + 1) * NODE_BASE_SIZE;
  return Math.min(Math.max(base, NODE_BASE_SIZE), NODE_MAX_SIZE);
};

/** Parse hex color to THREE.Color */
const parseColor = (hex: string): THREE.Color => {
  return new THREE.Color(hex);
};

// ============================================================================
// Component
// ============================================================================

const Graph3D: React.FC = () => {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<ForceGraphMethods | undefined>(undefined);
  const { width, height } = useResizeObserver(containerRef);

  // Cluster spheres refs for updating
  const clusterSpheresRef = useRef<Map<number, THREE.Mesh>>(new Map());

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

  // Local hover state
  const [localHoverNode, setLocalHoverNode] = useState<ProcessedNode | null>(null);
  const activeHoverNode = localHoverNode || (hoveredNode as ProcessedNode | null);

  // Get neighbors for highlighting
  const hoverNeighbors = useNodeNeighbors(activeHoverNode?.id);

  // Process data for force-graph-3d
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
        // Use pre-computed 3D positions scaled up
        x: n.x * POSITION_SCALE,
        y: n.y * POSITION_SCALE,
        z: n.z * POSITION_SCALE,
        // Fix positions to use pre-computed layout
        fx: n.x * POSITION_SCALE,
        fy: n.y * POSITION_SCALE,
        fz: n.z * POSITION_SCALE,
      })),
      links: filteredData.edges.map(e => ({
        source: typeof e.source === 'object' ? e.source.id : e.source,
        target: typeof e.target === 'object' ? e.target.id : e.target,
        weight: e.weight,
      })),
    };
  }, [filteredData]);

  // Cluster info map
  const clusterGroups = useMemo(() => {
    if (!rawData) return new Map<number, ClusterInfo>();

    const groups = new Map<number, ClusterInfo>();
    rawData.clusters.forEach(cluster => {
      groups.set(cluster.id, cluster);
    });
    return groups;
  }, [rawData]);

  // Calculate cluster bounding spheres
  const clusterBounds = useMemo(() => {
    const bounds = new Map<number, { center: THREE.Vector3; radius: number; color: string }>();

    if (!processedData.nodes.length) return bounds;

    // Group nodes by cluster
    const nodesByCluster = new Map<number, ProcessedNode[]>();
    processedData.nodes.forEach(node => {
      if (node.cluster_id !== undefined) {
        const group = nodesByCluster.get(node.cluster_id) || [];
        group.push(node);
        nodesByCluster.set(node.cluster_id, group);
      }
    });

    // Calculate bounding sphere for each cluster
    nodesByCluster.forEach((nodes, clusterId) => {
      if (nodes.length < 2) return;

      const cluster = clusterGroups.get(clusterId);
      if (!cluster) return;

      // Calculate center
      const center = new THREE.Vector3();
      nodes.forEach(n => {
        center.x += n.x;
        center.y += n.y;
        center.z += n.z;
      });
      center.divideScalar(nodes.length);

      // Calculate radius (max distance from center + padding)
      let maxDist = 0;
      nodes.forEach(n => {
        const dist = center.distanceTo(new THREE.Vector3(n.x, n.y, n.z));
        maxDist = Math.max(maxDist, dist);
      });

      bounds.set(clusterId, {
        center,
        radius: maxDist + 20, // Add padding
        color: cluster.color || '#808080',
      });
    });

    return bounds;
  }, [processedData.nodes, clusterGroups]);

  // Create cluster sphere meshes
  useEffect(() => {
    if (!graphRef.current) return;

    const scene = graphRef.current.scene();

    // Remove old spheres
    clusterSpheresRef.current.forEach(sphere => {
      scene.remove(sphere);
      sphere.geometry.dispose();
      (sphere.material as THREE.Material).dispose();
    });
    clusterSpheresRef.current.clear();

    // Create new spheres
    clusterBounds.forEach((bound, clusterId) => {
      const geometry = new THREE.SphereGeometry(bound.radius, 32, 32);
      const material = new THREE.MeshBasicMaterial({
        color: parseColor(bound.color),
        transparent: true,
        opacity: 0.05,
        side: THREE.BackSide,
        depthWrite: false,
      });

      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.copy(bound.center);
      sphere.renderOrder = -1; // Render behind nodes

      scene.add(sphere);
      clusterSpheresRef.current.set(clusterId, sphere);
    });

    return () => {
      clusterSpheresRef.current.forEach(sphere => {
        scene.remove(sphere);
        sphere.geometry.dispose();
        (sphere.material as THREE.Material).dispose();
      });
      clusterSpheresRef.current.clear();
    };
  }, [clusterBounds]);

  // Get node color based on state
  const getNodeColor = useCallback((node: ProcessedNode): string => {
    if (selectedNode && selectedNode.id === node.id) {
      return COLORS.NODE_SELECTED;
    }

    if (!activeHoverNode) {
      return node.color || COLORS.NODE_DEFAULT;
    }

    if (node.id === activeHoverNode.id) {
      return COLORS.NODE_HOVER;
    }

    if (hoverNeighbors.has(node.id)) {
      return COLORS.NODE_NEIGHBOR;
    }

    // Dim nodes not in same cluster
    if (activeHoverNode.cluster_id !== undefined &&
        node.cluster_id === activeHoverNode.cluster_id) {
      return node.color || COLORS.NODE_DEFAULT;
    }

    return 'rgba(107, 114, 128, 0.3)';
  }, [selectedNode, activeHoverNode, hoverNeighbors]);

  // Create node 3D object
  const createNodeObject = useCallback((node: any) => {
    const processedNode = node as ProcessedNode;
    const size = calculateNodeSize(processedNode.stargazers_count);
    const color = getNodeColor(processedNode);

    // Create sphere geometry
    const geometry = new THREE.SphereGeometry(size, 16, 16);

    // Create material with proper color handling
    const isTransparent = color.includes('rgba');
    const material = new THREE.MeshLambertMaterial({
      color: isTransparent ? '#6B7280' : color,
      transparent: isTransparent,
      opacity: isTransparent ? 0.3 : 1,
    });

    const mesh = new THREE.Mesh(geometry, material);

    // Add sprite label for important nodes
    if (processedNode.stargazers_count > 1000 ||
        activeHoverNode?.id === processedNode.id ||
        selectedNode?.id === processedNode.id) {
      const sprite = createTextSprite(processedNode.name);
      sprite.position.set(0, size + 5, 0);
      mesh.add(sprite);
    }

    return mesh;
  }, [getNodeColor, activeHoverNode, selectedNode]);

  // Create text sprite for labels
  const createTextSprite = (text: string): THREE.Sprite => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;

    const fontSize = 48;
    ctx.font = `${fontSize}px Inter, system-ui, sans-serif`;
    const textWidth = ctx.measureText(text).width;

    canvas.width = textWidth + 20;
    canvas.height = fontSize + 10;

    // Redraw with correct canvas size
    ctx.font = `${fontSize}px Inter, system-ui, sans-serif`;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#1F2937';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
    });

    const sprite = new THREE.Sprite(material);
    sprite.scale.set(canvas.width / 10, canvas.height / 10, 1);

    return sprite;
  };

  // Handle node click
  const handleNodeClick = useCallback((node: any) => {
    const processedNode = node as ProcessedNode;

    // Find full node data
    const fullNode = filteredData?.nodes.find(n => n.id === processedNode.id);
    if (fullNode) {
      setSelectedNode(fullNode);
    }

    // Fly camera to node
    if (graphRef.current) {
      const distance = 80;
      const distRatio = 1 + distance / Math.hypot(processedNode.x, processedNode.y, processedNode.z);

      graphRef.current.cameraPosition(
        {
          x: processedNode.x * distRatio,
          y: processedNode.y * distRatio,
          z: processedNode.z * distRatio,
        },
        { x: processedNode.x, y: processedNode.y, z: processedNode.z },
        2000
      );
    }
  }, [filteredData, setSelectedNode]);

  // Handle node hover
  const handleNodeHover = useCallback((node: any) => {
    setLocalHoverNode(node as ProcessedNode | null);
    setHoveredNode(node ? filteredData?.nodes.find(n => n.id === node.id) || null : null);
    document.body.style.cursor = node ? 'pointer' : 'default';
  }, [filteredData, setHoveredNode]);

  // Handle background click
  const handleBackgroundClick = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  // Get link color
  const getLinkColor = useCallback((link: ProcessedLink): string => {
    if (!activeHoverNode) return COLORS.LINK_DEFAULT;

    const sourceId = getNodeId(link.source);
    const targetId = getNodeId(link.target);

    if (sourceId === activeHoverNode.id || targetId === activeHoverNode.id) {
      return COLORS.LINK_ACTIVE;
    }

    return 'rgba(156, 163, 175, 0.1)';
  }, [activeHoverNode]);

  // Get link width
  const getLinkWidth = useCallback((link: ProcessedLink): number => {
    if (!activeHoverNode) return 0.5;

    const sourceId = getNodeId(link.source);
    const targetId = getNodeId(link.target);

    if (sourceId === activeHoverNode.id || targetId === activeHoverNode.id) {
      return 1.5;
    }

    return 0.3;
  }, [activeHoverNode]);

  // Loading state with skeleton
  if (loading) {
    return (
      <div ref={containerRef} className="w-full h-full relative overflow-hidden">
        <GraphSkeleton />
      </div>
    );
  }

  // Empty state
  if (!filteredData || filteredData.nodes.length === 0) {
    return (
      <div ref={containerRef} className="w-full h-full relative overflow-hidden rounded-sm border border-border-light bg-bg-sidebar/30 flex items-center justify-center">
        <div className="text-center p-8">
          <div className="text-6xl mb-4 grayscale opacity-20">🌌</div>
          <h3 className="text-text-main font-semibold text-lg mb-2">
            {t('graph.empty_title')}
          </h3>
          <p className="text-text-muted text-sm max-w-md">
            {t('graph.empty_hint')}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-hidden bg-[#FAFAFA]">
      <ForceGraph3D
        ref={graphRef}
        width={width}
        height={height}
        graphData={processedData as any}
        backgroundColor={COLORS.BACKGROUND}

        // Node rendering
        nodeThreeObject={createNodeObject}
        nodeThreeObjectExtend={false}
        nodeLabel={(node: any) => {
          const n = node as ProcessedNode;
          return `<div style="background: white; padding: 8px 12px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); font-family: Inter, system-ui, sans-serif;">
            <div style="font-weight: 600; color: #1F2937;">${n.name}</div>
            <div style="font-size: 12px; color: #6B7280; margin-top: 4px;">
              ${n.language ? `<span style="background: #F3F4F6; padding: 2px 6px; border-radius: 4px; margin-right: 8px;">${n.language}</span>` : ''}
              ⭐ ${n.stargazers_count.toLocaleString()}
            </div>
          </div>`;
        }}

        // Link rendering
        linkColor={getLinkColor}
        linkWidth={getLinkWidth}
        linkOpacity={0.4}

        // Interaction
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
        onBackgroundClick={handleBackgroundClick}

        // Camera
        enableNavigationControls={true}

        // Physics - disabled since we use pre-computed positions
        cooldownTicks={0}
        warmupTicks={0}
      />

      {/* Hover info overlay - Enhanced with full info */}
      {activeHoverNode && (
        <div className="absolute top-4 right-4 z-10 bg-white/98 backdrop-blur-sm px-4 py-3 rounded-lg border border-border-light shadow-lg max-w-sm pointer-events-none">
          {/* Header with Avatar */}
          <div className="flex items-start gap-3">
            {/* Owner Avatar */}
            {(activeHoverNode as any).owner_avatar_url ? (
              <img
                src={(activeHoverNode as any).owner_avatar_url}
                alt={(activeHoverNode as any).owner || activeHoverNode.name}
                className="w-10 h-10 rounded-md border border-border-light flex-shrink-0"
              />
            ) : (
              <div className="w-10 h-10 rounded-md bg-gray-200 flex items-center justify-center flex-shrink-0">
                <span className="text-gray-500 text-sm font-medium">
                  {((activeHoverNode as any).owner || activeHoverNode.name)?.charAt(0).toUpperCase()}
                </span>
              </div>
            )}

            <div className="flex-1 min-w-0">
              <h3 className="text-text-main font-semibold text-sm truncate">
                {activeHoverNode.name}
              </h3>
              <div className="flex items-center gap-2 mt-0.5">
                {activeHoverNode.language && (
                  <span className="text-[10px] px-1.5 py-0.5 bg-blue-50 rounded text-blue-700">
                    {activeHoverNode.language}
                  </span>
                )}
                <span className="text-xs text-orange-500 font-medium">
                  ⭐ {activeHoverNode.stargazers_count.toLocaleString()}
                </span>
              </div>
            </div>
          </div>

          {/* Full Description - Not truncated */}
          {(activeHoverNode.description || (activeHoverNode as any).ai_summary) && (
            <p className="text-xs text-text-muted mt-2 leading-relaxed">
              {activeHoverNode.description || (activeHoverNode as any).ai_summary}
            </p>
          )}

          {/* AI Tags Preview */}
          {(activeHoverNode as any).ai_tags && (activeHoverNode as any).ai_tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {(activeHoverNode as any).ai_tags.slice(0, 4).map((tag: string, idx: number) => (
                <span
                  key={idx}
                  className="text-[10px] px-1.5 py-0.5 bg-purple-50 text-purple-600 rounded"
                >
                  {tag}
                </span>
              ))}
              {(activeHoverNode as any).ai_tags.length > 4 && (
                <span className="text-[10px] text-text-dim">+{(activeHoverNode as any).ai_tags.length - 4}</span>
              )}
            </div>
          )}

          {/* Hint to click */}
          <div className="text-[10px] text-text-dim mt-2 pt-2 border-t border-border-light/50">
            Click for details & related repos
          </div>
        </div>
      )}
    </div>
  );
};

export default Graph3D;
