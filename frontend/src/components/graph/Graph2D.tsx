import React, { useMemo, useRef, useCallback, useEffect, useState } from 'react';
import ForceGraph2D, { ForceGraphMethods, NodeObject } from 'react-force-graph-2d';
import { useTranslation } from 'react-i18next';

import { useResizeObserver } from '../../hooks/useResizeObserver';
import { useGraph, useNodeNeighbors } from '../../contexts/GraphContext';
import { Link } from 'react-router-dom';

import { GraphSkeleton } from '../ui/page-skeletons';
import { Button } from '../ui/button';
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '../ui/empty';
import { Network } from 'lucide-react';
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
import {
  collectFocusClusterIds,
  resolveLinkColor,
  resolveLinkWidth,
  resolveNodeColor,
  resolveNodeDimmed,
  resolveShowClusterHue,
  shouldShowNodeLabel,
  type NodeColorContext,
} from './graph2dStyles';
import { getGraphMotionProfile } from './graph2dMotion';
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

  const selectedNeighbors = useNodeNeighbors(selectedNode?.id);

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

  const hasStablePositions = useMemo(
    () => processedNodes.length > 0 && processedNodes.every(
      (node) => Number.isFinite(node.x) && Number.isFinite(node.y)
    ),
    [processedNodes]
  );

  const layoutKey = useMemo(
    () => rawData?.version ?? rawData?.generated_at ?? 'graph',
    [rawData?.generated_at, rawData?.version]
  );

  const motionProfile = useMemo(
    () => getGraphMotionProfile(hasStablePositions, reduceMotion),
    [hasStablePositions, reduceMotion]
  );

  const clusterLayoutData = useMemo(
    () => buildClusterLayoutData(processedData.nodes),
    [processedData.nodes]
  );

  // Group clusters by id for hull drawing
  const clusterGroups = useMemo(() => buildClusterGroups(rawData), [rawData]);

  useGraphForces({
    graphRef,
    clusterLayoutData,
    layoutKey,
    forceScale: motionProfile.forceScale,
  });

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

  const nodeStyleContext = useMemo<NodeColorContext>(
    () => ({
      selectedNodeId: selectedNode?.id,
      selectedClusterId: selectedNode?.cluster_id,
      activeHoverNode,
      selectedNeighbors,
      visibleNodeIds,
    }),
    [activeHoverNode, selectedNeighbors, selectedNode?.cluster_id, selectedNode?.id, visibleNodeIds]
  );

  const focusedClusterIds = useMemo(
    () => collectFocusClusterIds(selectedNode?.cluster_id, activeHoverNode?.cluster_id),
    [activeHoverNode?.cluster_id, selectedNode?.cluster_id]
  );

  const clusterByNodeId = useMemo(() => {
    const map = new Map<number, number | null>();
    for (const node of processedNodes) {
      map.set(node.id, node.cluster_id);
    }
    return map;
  }, [processedNodes]);

  const getNodeColor = useCallback(
    (node: ProcessedNode): string => resolveNodeColor(node, nodeStyleContext),
    [nodeStyleContext]
  );

  const getLinkColor = useCallback(
    (link: ProcessedLink): string =>
      resolveLinkColor(link, {
        showTrajectories: settings.showTrajectories,
        activeHoverNodeId: activeHoverNode?.id,
        visibleNodeIds,
        selectedNodeId: selectedNode?.id,
        selectedClusterId: selectedNode?.cluster_id,
        hoverClusterId: activeHoverNode?.cluster_id,
        selectedNeighbors,
        clusterByNodeId,
      }),
    [
      activeHoverNode?.cluster_id,
      activeHoverNode?.id,
      clusterByNodeId,
      selectedNeighbors,
      selectedNode?.cluster_id,
      selectedNode?.id,
      settings.showTrajectories,
      visibleNodeIds,
    ]
  );

  const getLinkWidth = useCallback(
    (link: ProcessedLink): number =>
      resolveLinkWidth(link, {
        showTrajectories: settings.showTrajectories,
        activeHoverNodeId: activeHoverNode?.id,
        visibleNodeIds,
        selectedNodeId: selectedNode?.id,
        selectedClusterId: selectedNode?.cluster_id,
        hoverClusterId: activeHoverNode?.cluster_id,
        selectedNeighbors,
        clusterByNodeId,
      }),
    [
      activeHoverNode?.cluster_id,
      activeHoverNode?.id,
      clusterByNodeId,
      selectedNeighbors,
      selectedNode?.cluster_id,
      selectedNode?.id,
      settings.showTrajectories,
      visibleNodeIds,
    ]
  );

  const paintNode = useCallback(
    (nodeObject: NodeObject<NodeObject>, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const node = nodeObject as ProcessedNode;
      const isVisible = visibleNodeIds.has(node.id);
      paintNodeOnCanvas({
        node,
        ctx,
        globalScale,
        color: getNodeColor(node),
        isVisible,
        isSelected: selectedNode?.id === node.id,
        isHovered: activeHoverNode?.id === node.id,
        isDimmed: resolveNodeDimmed(node, nodeStyleContext),
        showLabel: shouldShowNodeLabel(node.id, nodeStyleContext, isVisible),
        showClusterHue: resolveShowClusterHue(node, nodeStyleContext),
        hqRendering: settings.hqRendering,
        imageCache: imageCacheRef.current,
        onAvatarLoaded: triggerAvatarRedraw,
      });
    },
    [
      getNodeColor,
      activeHoverNode,
      nodeStyleContext,
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
        focusedClusterIds,
      });
    },
    [clusterGroups, focusedClusterIds, processedData.nodes, visibleNodeIds]
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
        <Empty className="border-0 bg-transparent">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <Network />
            </EmptyMedia>
            <EmptyTitle>{t('graph.empty_title')}</EmptyTitle>
            <EmptyDescription>{t('graph.empty_hint')}</EmptyDescription>
          </EmptyHeader>
          <EmptyContent>
            <Button nativeButton={false} render={<Link to="/settings" />}>
              {t('common.sync_now')}
            </Button>
          </EmptyContent>
        </Empty>
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
        d3AlphaDecay={motionProfile.alphaDecay}
        d3VelocityDecay={motionProfile.velocityDecay}
        cooldownTicks={motionProfile.cooldownTicks}
        cooldownTime={motionProfile.cooldownTime}
        warmupTicks={motionProfile.warmupTicks}
        // After engine stops
        onEngineTick={() => {
          if (import.meta.env.DEV) {
            const w = window as Window & { __nebulaGraphTicks?: number };
            w.__nebulaGraphTicks = (w.__nebulaGraphTicks ?? 0) + 1;
          }
        }}
        onEngineStop={tryAutoFit}
      />

      {activeHoverNode && <GraphHoverCard node={activeHoverNode} />}
    </div>
  );
};

export default Graph2D;
