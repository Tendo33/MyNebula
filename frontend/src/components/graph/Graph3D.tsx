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
      <div ref={containerRef} className="w-full h-full relative overflow-hidden rounded-2xl border border-nebula-border bg-nebula-bg shadow-inner flex items-center justify-center">
        <div className="text-center p-8">
          <div className="text-6xl mb-4">🌌</div>
          <h3 className="text-nebula-text-main font-bold text-xl mb-2">星空尚未点亮</h3>
          <p className="text-nebula-text-muted text-sm max-w-md">
            点击右上角的"开始同步"按钮，同步你的 GitHub Star 项目，开始探索你的代码宇宙！
          </p>
        </div>
      </div>
    );
  }


  return (
    <div ref={containerRef} className="w-full h-full relative overflow-hidden rounded-2xl border border-nebula-border bg-nebula-bg shadow-inner">
      {/* Overlay UI */}
      <div className="absolute top-4 right-4 z-10 bg-nebula-surface/90 backdrop-blur-md p-4 rounded-xl border border-nebula-border/50 max-w-xs transition-opacity duration-300">
        <h3 className="text-nebula-text-main font-bold mb-1">
          {hoverNode ? hoverNode.name : 'Galactic View'}
        </h3>
        <p className="text-nebula-text-muted text-sm">
          {hoverNode ? hoverNode.description || 'No description' : 'Hover over a star to reveal details'}
        </p>
      </div>

      <ForceGraph3D
        ref={graphRef}
        width={width}
        height={height}
        graphData={processedData as any}
        backgroundColor="#0B0B10"
        nodeLabel="name"

        nodeColor={(node: any) => node.color || '#E0E0FF'}
        nodeVal={(node: any) => node.size || node.val || 1}
        nodeResolution={16}
        linkOpacity={0.2}
        linkWidth={0.5}
        linkColor={() => '#2D2D3A'}

        // Custom Node Layout
        nodeThreeObject={(node: any) => {
          const size = node.size || node.val || 1;
          const color = node.color || '#E0E0FF';

          const group = new THREE.Group();

          // Core Sphere
          const geometry = new THREE.SphereGeometry(Math.max(1, Math.log(size) * 2));
          const material = new THREE.MeshStandardMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: 0.8,
            roughness: 0.1,
            metalness: 0.5
          });
          const mesh = new THREE.Mesh(geometry, material);
          group.add(mesh);

          // Glow Halo (transparent slightly larger sphere)
          if (hoverNode && hoverNode.id === node.id) {
             const glowGeo = new THREE.SphereGeometry(Math.max(1, Math.log(size) * 3));
             const glowMat = new THREE.MeshBasicMaterial({
                color: color,
                transparent: true,
                opacity: 0.3,
             });
             const glowMesh = new THREE.Mesh(glowGeo, glowMat);
             group.add(glowMesh);
          }

          return group;
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
