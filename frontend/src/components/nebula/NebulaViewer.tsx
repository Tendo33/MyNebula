import React, { useCallback, useEffect, useRef, useState } from 'react';
import ForceGraph3D, { ForceGraphMethods } from 'react-force-graph-3d';
import { useResizeObserver } from '../../hooks/useResizeObserver';

// Mock data for initial view
const MOCK_DATA = {
  nodes: [
    { id: 1, name: 'fastapi', val: 20, color: '#06b6d4', group: 1 },
    { id: 2, name: 'flask', val: 15, color: '#06b6d4', group: 1 },
    { id: 3, name: 'django', val: 25, color: '#06b6d4', group: 1 },
    { id: 4, name: 'react', val: 30, color: '#8B5CF6', group: 2 },
    { id: 5, name: 'vue', val: 20, color: '#8B5CF6', group: 2 },
    { id: 6, name: 'pytorch', val: 25, color: '#EC4899', group: 3 },
    { id: 7, name: 'tensorflow', val: 25, color: '#EC4899', group: 3 },
  ],
  links: [
    { source: 1, target: 2 },
    { source: 2, target: 3 },
    { source: 4, target: 5 },
    { source: 6, target: 7 },
    { source: 1, target: 3 },
  ]
};

export const NebulaViewer: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<ForceGraphMethods>();
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight
        });
      }
    };

    window.addEventListener('resize', updateDimensions);
    updateDimensions();

    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  const handleNodeClick = useCallback((node: any) => {
    if (fgRef.current) {
      // Aim at node from outside it
      const distance = 40;
      const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);

      fgRef.current.cameraPosition(
        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }, // new position
        node, // lookAt ({ x, y, z })
        3000  // ms transition duration
      );
    }
  }, []);

  return (
    <div ref={containerRef} className="w-full h-full absolute inset-0 bg-space-950">
      {/* Background Starfield Effect (CSS) */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-space-800/50 via-space-950 to-space-950 pointer-events-none" />

      <ForceGraph3D
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={MOCK_DATA}
        nodeLabel="name"
        nodeColor="color"
        nodeVal="val"
        enableNodeDrag={false}
        backgroundColor="#00000000" // Transparent
        showNavInfo={false}
        onNodeClick={handleNodeClick}
        nodeResolution={16}
        linkOpacity={0.2}
        linkWidth={1}
      />
    </div>
  );
};
