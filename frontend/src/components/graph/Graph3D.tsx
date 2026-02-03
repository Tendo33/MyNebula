import React, { useMemo, useRef, useState } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import { GraphData, GraphNode } from '../../types';
import { useResizeObserver } from '../../hooks/useResizeObserver';

interface Graph3DProps {
  data: GraphData | null;
  onNodeClick?: (node: GraphNode) => void;
}

const Graph3D: React.FC<Graph3DProps> = ({ data, onNodeClick }) => {

  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);
  const { width, height } = useResizeObserver(containerRef);
  const [hoverNode, setHoverNode] = useState<GraphNode | null>(null);

  // Transform data for the graph library
  const processedData = useMemo(() => {
    if (!data || !data.nodes || data.nodes.length === 0) {
      return { nodes: [], links: [] };
    }
    return {
      nodes: data.nodes,
      links: data.edges.map(e => ({ source: e.source, target: e.target }))
    };
  }, [data]);

  // Show empty state if no data
  if (!data || !data.nodes || data.nodes.length === 0) {
    return (
      <div ref={containerRef} className="w-full h-full relative overflow-hidden rounded-sm border border-border-light bg-bg-sidebar/30 flex items-center justify-center">
        <div className="text-center p-8">
          <div className="text-6xl mb-4 grayscale opacity-20">🌌</div>
          <h3 className="text-text-main font-semibold text-lg mb-2">星空尚未点亮</h3>
          <p className="text-text-muted text-sm max-w-md">
            点击右上角的"同步"按钮，同步你的 GitHub Star 项目，开始探索你的代码宇宙！
          </p>
        </div>
      </div>
    );
  }


  return (
    <div ref={containerRef} className="w-full h-full relative overflow-hidden bg-white">
      {/* Overlay UI - Minimal Hover Card */}
      {hoverNode && (
        <div className="absolute top-4 right-4 z-10 bg-white/90 backdrop-blur-sm px-3 py-2 rounded-sm border border-border-light shadow-sm max-w-xs transition-opacity duration-200">
            <h3 className="text-text-main font-semibold text-sm mb-0.5">
            {hoverNode.name}
            </h3>
            <p className="text-text-muted text-xs truncate">
            {hoverNode.description || 'No description'}
            </p>
        </div>
      )}

      <ForceGraph3D
        ref={graphRef}
        width={width}
        height={height}
        graphData={processedData as any}
        backgroundColor="#FFFFFF"
        nodeLabel="name"

        nodeColor={(node: any) => node.color || '#37352F'} // Dark gray default
        nodeVal={(node: any) => node.size || node.val || 1}
        nodeResolution={16}
        linkOpacity={0.3}
        linkWidth={1}
        linkColor={() => '#E9E9E7'} // Light gray links

        // Custom Node Layout
        nodeThreeObject={(node: any) => {
          const size = node.size || node.val || 1;
          const color = node.color ? node.color : '#37352F'; // Use node color if present, else dark gray

          // Use solid materials instead of emissive for flat look
          const geometry = new THREE.SphereGeometry(Math.max(1, Math.log(size) * 2));
           const material = new THREE.MeshLambertMaterial({
            color: color,
            transparent: false
          });

          return new THREE.Mesh(geometry, material);
        }}

        onNodeHover={(node: any) => {
            document.body.style.cursor = node ? 'pointer' : 'default';
            setHoverNode(node);
        }}
        onNodeClick={(node: any) => {
            if (graphRef.current) {
                // Fly to node
                const distance = 40;
                const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);

                graphRef.current.cameraPosition(
                    { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }, // new position
                    node, // lookAt ({ x, y, z })
                    3000  // ms transition duration
                );
            }
            if (onNodeClick) onNodeClick(node);
        }}
      />
    </div>
  );
};

export default Graph3D;
