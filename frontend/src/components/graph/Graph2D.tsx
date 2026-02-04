import React, { useMemo, useRef, useState, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { GraphData, GraphNode } from '../../types';
import { useResizeObserver } from '../../hooks/useResizeObserver';
import { useTranslation } from 'react-i18next';

interface Graph2DProps {
  data: GraphData | null;
  onNodeClick?: (node: GraphNode) => void;
}

const Graph2D: React.FC<Graph2DProps> = ({ data, onNodeClick }) => {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);
  const { width, height } = useResizeObserver(containerRef);
  const [hoverNode, setHoverNode] = useState<GraphNode | null>(null);

  // Pruned graph data for performance if needed, but for now we just map
  const processedData = useMemo(() => {
    if (!data || !data.nodes || data.nodes.length === 0) {
      return { nodes: [], links: [] };
    }
    return {
      nodes: data.nodes.map(n => ({ ...n })), // Shallow copy to avoid mutating original
      links: data.edges.map(e => ({ source: e.source, target: e.target }))
    };
  }, [data]);

  // Apply custom forces
  React.useEffect(() => {
    if (graphRef.current) {
        // Tighter links
        graphRef.current.d3Force('link').distance(30);
        // Slightly stronger charge to avoid total overlap but keep close
        graphRef.current.d3Force('charge').strength(-30);
    }
  }, [graphRef]);

  // Obsidian-like colors
  const NODE_REL_SIZE = 7;
  const NODE_MAX_SIZE = 25; // Cap max size
  const HOVER_COLOR = '#9980FA'; // A nice purple/blue for hover
  const DEFAULT_NODE_COLOR = '#535c68'; // Muted dark gray for unselected
  const ACTIVE_NODE_COLOR = '#dcdde1'; // Brighter for active/connected
  const LINK_COLOR = 'rgba(180, 180, 180, 0.5)'; // Much more visible
  const LINK_ACTIVE_COLOR = 'rgba(153, 128, 250, 0.6)';

  // Helper to determine node color
  const getNodeColor = useCallback((node: any) => {
    if (hoverNode) {
      // If hovering, highlight node and neighbors
      const isHovered = node.id === hoverNode.id;
      const isNeighbor = graphRef.current?.graphData().links.some((link: any) => {
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
        const targetId = typeof link.target === 'object' ? link.target.id : link.target;
        return (sourceId === hoverNode.id && targetId === node.id) ||
               (targetId === hoverNode.id && sourceId === node.id);
      });

      if (isHovered) return HOVER_COLOR;
      if (isNeighbor) return ACTIVE_NODE_COLOR;
      return 'rgba(83, 92, 104, 0.2)'; // Dim non-connected nodes
    }
    return node.color || DEFAULT_NODE_COLOR;
  }, [hoverNode]);

  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const label = node.name;
    const fontSize = 12 / globalScale;
    const val = node.val || node.size || 1;
    // Logarithmic size scaling, clamped
    const radius = Math.min(Math.max(Math.log(val + 1) * NODE_REL_SIZE, 4), NODE_MAX_SIZE);

    const color = getNodeColor(node);

    // Draw Circle
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
    ctx.fill();

    // Hover/Selection Halo
    if (hoverNode && (node.id === hoverNode.id)) {
        ctx.beginPath();
        ctx.lineWidth = 2 / globalScale;
        ctx.strokeStyle = HOVER_COLOR;
        ctx.arc(node.x, node.y, radius * 1.5, 0, 2 * Math.PI, false);
        ctx.stroke();
    }

    // Text Label - Show if hovered or sufficiently zoomed in or very large node
    const showLabel = (hoverNode && (node.id === hoverNode.id)) || globalScale > 1.5 || radius > 8;

    if (showLabel) {
      ctx.font = `${fontSize}px Sans-Serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = hoverNode ? (getNodeColor(node)) : '#2f3542'; // Dark text
      if (hoverNode && node.id !== hoverNode.id && getNodeColor(node).includes('0.2')) {
         ctx.fillStyle = 'rgba(47, 53, 66, 0.2)'; // Dim label too
      }

      // Background for text readability (optional, maybe just shadow)
      ctx.shadowColor = "rgba(255, 255, 255, 0.8)";
      ctx.shadowBlur = 4;

      ctx.fillText(label, node.x, node.y + radius + fontSize);

      ctx.shadowBlur = 0; // Reset
    }
  }, [hoverNode, getNodeColor]);

  // Handle click to fly/zoom or select
  const handleNodeClick = useCallback((node: any) => {
    if (onNodeClick) onNodeClick(node);

    // Zoom to node
    graphRef.current?.centerAt(node.x, node.y, 1000);
    graphRef.current?.zoom(4, 2000);
  }, [onNodeClick]);

  if (!data || !data.nodes || data.nodes.length === 0) {
    return (
        <div ref={containerRef} className="w-full h-full relative flex items-center justify-center bg-gray-50/50">
            <div className="text-center p-8 opacity-50">
                <div className="text-6xl mb-4 grayscale">🕸️</div>
                <h3 className="font-semibold text-lg">{t('dashboard.subtitle_infinite')}</h3>
            </div>
        </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full h-full relative bg-[#fdfdfd]">
        <ForceGraph2D
            ref={graphRef}
            width={width}
            height={height}
            graphData={processedData}

            // Interaction
            onNodeClick={handleNodeClick}
            onNodeHover={setHoverNode}

            // Rendering
            nodeCanvasObject={paintNode}
            nodePointerAreaPaint={(node: any, color, ctx) => {
                const val = node.val || node.size || 1;
                const radius = Math.min(Math.max(Math.log(val + 1) * NODE_REL_SIZE, 4), NODE_MAX_SIZE);
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(node.x, node.y, radius + 2, 0, 2 * Math.PI, false);
                ctx.fill();
            }}

            // Links
            linkColor={(link: any) => {
                 if (hoverNode) {
                    const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                    const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                    if (sourceId === hoverNode.id || targetId === hoverNode.id) {
                        return LINK_ACTIVE_COLOR;
                    }
                    return 'rgba(200, 200, 200, 0.1)'; // Slightly dim when not active but still visible
                 }
                 return LINK_COLOR;
            }}
            linkWidth={(link: any) => {
                if (hoverNode) {
                     const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                     const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                     if (sourceId === hoverNode.id || targetId === hoverNode.id) {
                         return 1.5;
                     }
                  }
                  return 0.5;
            }}

            // Forces - Tighter cluster
            d3AlphaDecay={0.02}
            d3VelocityDecay={0.3}
            cooldownTicks={100}
            onEngineStop={() => graphRef.current?.zoomToFit(400)}
        />

        {/* Overlay for hover info if needed (optional, since we have labels) */}
        {hoverNode && (
            <div className="absolute top-4 right-4 pointer-events-none bg-white/90 backdrop-blur px-3 py-2 rounded border border-gray-200 shadow-sm z-10">
                <div className="font-bold text-sm text-gray-800">{hoverNode.name}</div>
                <div className="text-xs text-gray-500">{hoverNode.language || 'Unknown'}</div>
            </div>
        )}
    </div>
  );
};

export default Graph2D;
