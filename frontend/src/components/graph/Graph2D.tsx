import React, { useMemo, useRef, useCallback, useEffect, useState } from 'react';
import ForceGraph2D, { ForceGraphMethods, NodeObject } from 'react-force-graph-2d';
import { useTranslation } from 'react-i18next';

import { useResizeObserver } from '../../hooks/useResizeObserver';
import { useGraph, useNodeNeighbors } from '../../contexts/GraphContext';
import { GraphSkeleton } from '../ui/Skeleton';
import { EmptyState } from '../ui/EmptyState';
import { GraphHoverCard } from './GraphHoverCard';
import type {
  HullCache,
  ProcessedData,
  ProcessedLink,
  ProcessedNode,
} from './graph2dTypes';
import {
  buildClusterGroups,
  buildClusterLayoutData,
  toProcessedLinks,
  toProcessedNodes,
} from './graph2dLayout';
import { resolveLinkColor, resolveLinkWidth, resolveNodeColor } from './graph2dStyles';
import {
  drawClusterHulls,
  paintNodeOnCanvas,
  paintNodePointerArea,
} from './graph2dPainters';
import { useAvatarImageCache } from './hooks/useAvatarImageCache';
import { useGraphForces } from './hooks/useGraphForces';
import { useFocusSelectedNode, useGraphViewport } from './hooks/useGraphViewport';

/**
 * Force-graph canvas.
 *
 * This component owns wiring only. Pure layout derivation lives in
 * `graph2dLayout`, colour rules in `graph2dStyles`, canvas painting in
 * `graph2dPainters`, and stateful concerns in `./hooks/*`.
 */
const Graph2D: React.FC = () => {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<ForceGraphMethods | undefined>(undefined);
  const { width, height } = useResizeObserver(containerRef);

  const hullCacheRef = useRef<HullCache>(new Map());
  const { imageCacheRef, triggerAvatarRedraw } = useAvatarImageCache();

  // Global state
  const { filteredData, rawData, selectedNode, setSelectedNode, settings, loading, nodesLoading, error } = useGraph();

  // Local state for graph-specific interactions
  const [activeHoverNode, setActiveHoverNode] = useState<ProcessedNode | null>(null);
  const [reduceMotion, setReduceMotion] = useState(() =>
    typeof window !== 'undefined'
      ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
      : false
  );

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = () => setReduceMotion(media.matches);
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, []);

  // Get neighbors of hovered node for highlighting
  const hoverNeighbors = useNodeNeighbors(activeHoverNode?.id);

  // Visibility set
  const visibleNodeIds = useMemo(() => {
    if (!filteredData) return new Set<number>();
    return new Set(filteredData.nodes.map((n) => n.id));
  }, [filteredData]);

  // Process data for force-graph (from rawData to keep layout stable!)
  const rawNodes = rawData?.nodes;
  const rawEdges = rawData?.edges;
  const processedNodes = useMemo(() => toProcessedNodes(rawNodes), [rawNodes]);
  const processedLinks = useMemo(() => toProcessedLinks(rawEdges), [rawEdges]);

  const processedData = useMemo<ProcessedData>(
    () => ({ nodes: processedNodes, links: processedLinks }),
    [processedLinks, processedNodes]
  );

  const layoutKey = useMemo(
    () => rawData?.version ?? processedNodes.map((node) => node.id).join(':'),
    [processedNodes, rawData?.version]
  );

  const clusterLayoutData = useMemo(
    () => buildClusterLayoutData(processedData.nodes),
    [processedData.nodes]
  );

  // Group clusters by id for hull drawing
  const clusterGroups = useMemo(() => buildClusterGroups(rawData), [rawData]);

  useGraphForces({ graphRef, clusterLayoutData });

  useEffect(() => {
    if (reduceMotion) return;
    graphRef.current?.d3ReheatSimulation();
  }, [layoutKey, reduceMotion]);

  const { tryAutoFit, getLiveNodeById, focusNodeById, markUserInteracted, skipNextFocusRef } =
    useGraphViewport({
      graphRef,
      hullCacheRef,
      layoutKey,
      nodeCount: processedData.nodes.length,
      width,
      height,
    });

  const selectedNodeId = selectedNode?.id;
  useFocusSelectedNode({
    selectedNodeId,
    graphRef,
    skipNextFocusRef,
    getLiveNodeById,
    focusNodeById,
  });

  const getNodeColor = useCallback(
    (node: ProcessedNode): string =>
      resolveNodeColor(node, {
        selectedNodeId: selectedNode?.id,
        activeHoverNode,
        hoverNeighbors,
        visibleNodeIds,
      }),
    [selectedNode, activeHoverNode, hoverNeighbors, visibleNodeIds]
  );

  const getLinkColor = useCallback(
    (link: ProcessedLink): string =>
      resolveLinkColor(link, {
        showTrajectories: settings.showTrajectories,
        activeHoverNodeId: activeHoverNode?.id,
        visibleNodeIds,
        selectedNodeId: selectedNode?.id,
      }),
    [activeHoverNode, settings.showTrajectories, visibleNodeIds, selectedNode]
  );

  const getLinkWidth = useCallback(
    (link: ProcessedLink): number =>
      resolveLinkWidth(link, {
        showTrajectories: settings.showTrajectories,
        activeHoverNodeId: activeHoverNode?.id,
        visibleNodeIds,
        selectedNodeId: selectedNode?.id,
      }),
    [activeHoverNode, settings.showTrajectories, visibleNodeIds, selectedNode]
  );

  const paintNode = useCallback(
    (nodeObject: NodeObject<NodeObject>, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const node = nodeObject as ProcessedNode;
      paintNodeOnCanvas({
        node,
        ctx,
        globalScale,
        color: getNodeColor(node),
        isVisible: visibleNodeIds.has(node.id),
        isSelected: selectedNode?.id === node.id,
        isHovered: activeHoverNode?.id === node.id,
        hqRendering: settings.hqRendering,
        imageCache: imageCacheRef.current,
        onAvatarLoaded: triggerAvatarRedraw,
      });
    },
    [
      getNodeColor,
      activeHoverNode,
      selectedNode,
      settings.hqRendering,
      triggerAvatarRedraw,
      visibleNodeIds,
      imageCacheRef,
    ]
  );

  const paintNodeArea = useCallback(
    (nodeObject: NodeObject<NodeObject>, color: string, ctx: CanvasRenderingContext2D) => {
      paintNodePointerArea(nodeObject as ProcessedNode, color, ctx);
    },
    []
  );

  const paintClusterHulls = useCallback(
    (ctx: CanvasRenderingContext2D, globalScale: number) => {
      if (!graphRef.current) return;
      drawClusterHulls({
        nodes: processedData.nodes,
        ctx,
        globalScale,
        visibleNodeIds,
        clusterGroups,
        hullCache: hullCacheRef.current,
      });
    },
    [clusterGroups, processedData.nodes, visibleNodeIds]
  );

  // Handle node click
  const handleNodeClick = useCallback(
    (nodeObject: NodeObject<NodeObject>) => {
      const processedNode = nodeObject as ProcessedNode;

      // Find the full node data from raw data
      const fullNode = rawData?.nodes.find((n) => n.id === processedNode.id);
      if (fullNode) {
        if (selectedNode?.id !== fullNode.id) {
          skipNextFocusRef.current = true;
          setSelectedNode(fullNode);
        }
      }

      // Always focus the node when explicitly clicked on the graph canvas
      focusNodeById(processedNode.id, 1000);
    },
    [rawData, selectedNode, setSelectedNode, focusNodeById, skipNextFocusRef]
  );

  // Handle node hover
  const handleNodeHover = useCallback((nodeObject: NodeObject<NodeObject> | null) => {
    const processedNode = nodeObject as ProcessedNode | null;

    setActiveHoverNode(processedNode);
    document.body.style.cursor = processedNode ? 'pointer' : '';
  }, []);

  useEffect(() => {
    return () => {
      document.body.style.cursor = '';
    };
  }, []);

  // Handle background click (deselect)
  const handleBackgroundClick = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  const awaitingFirstNodes = Boolean(
    rawData && rawData.total_nodes > 0 && rawData.nodes.length === 0 && nodesLoading
  );

  if (loading || awaitingFirstNodes) {
    return (
      <div ref={containerRef} className="w-full h-full relative">
        <GraphSkeleton />
      </div>
    );
  }

  if (error && (!filteredData || filteredData.nodes.length === 0)) {
    return <div ref={containerRef} className="relative h-full w-full" />;
  }

  if (!filteredData || filteredData.nodes.length === 0) {
    return (
      <div
        ref={containerRef}
        className="relative flex h-full w-full items-center justify-center bg-bg-hover/50 dark:bg-dark-bg-sidebar/60"
      >
        <EmptyState
          title={t('graph.empty_title')}
          description={t('graph.empty_hint')}
          actionTo="/settings"
          actionLabel={t('common.sync_now')}
        />
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full h-full relative bg-bg-main dark:bg-dark-bg-main">
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
        onZoom={markUserInteracted}
        onNodeDragEnd={markUserInteracted}
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
        onRenderFramePre={paintClusterHulls}
        // Physics
        d3AlphaDecay={reduceMotion ? 0.05 : 0.008}
        d3VelocityDecay={0.28}
        cooldownTicks={reduceMotion ? 0 : 480}
        warmupTicks={reduceMotion ? 80 : 0}
        // After engine stops
        onEngineStop={tryAutoFit}
      />

      {activeHoverNode && <GraphHoverCard node={activeHoverNode} />}
    </div>
  );
};

export default Graph2D;
