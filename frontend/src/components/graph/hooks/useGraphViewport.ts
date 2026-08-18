import { useCallback, useEffect, useRef } from 'react';
import type { ForceGraphMethods, NodeObject } from 'react-force-graph-2d';

import { ZOOM_TO_FIT_PADDING, type HullCache, type ProcessedNode } from '../graph2dTypes';

/**
 * Viewport behaviour for the graph canvas: auto-fit, focus-on-select, and
 * user-interaction tracking.
 *
 * Auto-fit runs once per layout and stops as soon as the user pans or zooms, so
 * the view never yanks out from under them.
 */
export const useGraphViewport = ({
  graphRef,
  hullCacheRef,
  layoutKey,
  nodeCount,
  width,
  height,
}: {
  graphRef: React.MutableRefObject<ForceGraphMethods | undefined>;
  hullCacheRef: React.MutableRefObject<HullCache>;
  layoutKey: string;
  nodeCount: number;
  width: number;
  height: number;
}) => {
  const autoFitKeyRef = useRef<string | null>(null);
  const userInteractedRef = useRef<boolean>(false);
  const skipNextFocusRef = useRef<boolean>(false);

  const tryAutoFit = useCallback(() => {
    if (!graphRef.current || nodeCount === 0) return;
    if (autoFitKeyRef.current === layoutKey) return;
    if (userInteractedRef.current) return;

    graphRef.current.zoomToFit(400, ZOOM_TO_FIT_PADDING);
    autoFitKeyRef.current = layoutKey;
  }, [graphRef, layoutKey, nodeCount]);

  // Reset auto-fit state when layout shape changes
  useEffect(() => {
    autoFitKeyRef.current = null;
    hullCacheRef.current.clear();
  }, [hullCacheRef, layoutKey]);

  // Fallback auto-fit after initial render in case engine-stop callback is delayed
  useEffect(() => {
    if (nodeCount === 0) return;

    const timer = setTimeout(() => {
      tryAutoFit();
    }, 600);
    return () => clearTimeout(timer);
  }, [nodeCount, tryAutoFit]);

  const getLiveNodeById = useCallback(
    (nodeId: number): ProcessedNode | null => {
      if (!graphRef.current) return null;

      // `graphData()` exists at runtime, but some react-force-graph type versions
      // don't expose it on ForceGraphMethods.
      const maybeWithGraphData = graphRef.current as ForceGraphMethods & {
        graphData?: () => { nodes: NodeObject[] };
      };
      const graphData = maybeWithGraphData.graphData?.();
      if (!graphData) return null;

      return (graphData.nodes as ProcessedNode[]).find((n) => n.id === nodeId) ?? null;
    },
    [graphRef]
  );

  const focusNodeById = useCallback(
    (nodeId: number, duration = 1200) => {
      if (!graphRef.current || !width || !height) return;

      // Fixed zoom target to ensure an obvious, but reasonable magnification on select
      const targetZoom = 1.05;

      // zoomToFit's scale calculation for a single point is:
      // scale = Math.min(width, height) / (padding * 2)
      // Thus padding = Math.min(width, height) / (2 * targetZoom)
      const padding = Math.min(width, height) / (2 * targetZoom);

      // Use built-in zoomToFit to perfectly animate pan and zoom simultaneously
      // without D3 transitions cancelling each other out.
      graphRef.current.zoomToFit(duration, padding, (n) => n.id === nodeId);
    },
    [graphRef, width, height]
  );

  const markUserInteracted = useCallback(() => {
    userInteractedRef.current = true;
  }, []);

  return {
    tryAutoFit,
    getLiveNodeById,
    focusNodeById,
    markUserInteracted,
    skipNextFocusRef,
  };
};

/**
 * Centre the canvas on the selected node once the simulation has given it
 * coordinates. Waits two frames so force-graph has ingested a possibly-new
 * graphData, then retries once if positions are still missing.
 */
export const useFocusSelectedNode = ({
  selectedNodeId,
  graphRef,
  skipNextFocusRef,
  getLiveNodeById,
  focusNodeById,
}: {
  selectedNodeId: number | undefined;
  graphRef: React.MutableRefObject<ForceGraphMethods | undefined>;
  skipNextFocusRef: React.MutableRefObject<boolean>;
  getLiveNodeById: (nodeId: number) => ProcessedNode | null;
  focusNodeById: (nodeId: number, duration?: number) => void;
}) => {
  useEffect(() => {
    if (!selectedNodeId) return;

    if (skipNextFocusRef.current) {
      skipNextFocusRef.current = false;
      return;
    }

    let frame1 = 0;
    let frame2 = 0;
    let retryTimer = 0;

    const tryFocus = () => {
      if (!graphRef.current) return;
      const node = getLiveNodeById(selectedNodeId);
      if (node?.x !== undefined && node?.y !== undefined) {
        focusNodeById(selectedNodeId, 800);
      } else {
        // Node positions not ready yet — retry after the simulation warms up
        retryTimer = window.setTimeout(() => focusNodeById(selectedNodeId, 800), 600);
      }
    };

    frame1 = window.requestAnimationFrame(() => {
      frame2 = window.requestAnimationFrame(tryFocus);
    });

    return () => {
      window.cancelAnimationFrame(frame1);
      window.cancelAnimationFrame(frame2);
      window.clearTimeout(retryTimer);
    };
  }, [selectedNodeId, focusNodeById, getLiveNodeById, graphRef, skipNextFocusRef]);
};
