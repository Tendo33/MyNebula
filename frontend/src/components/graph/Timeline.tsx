import React from 'react';
import { clsx } from 'clsx';
import { TimelineData } from '../../types';

interface TimelineProps {
  data?: TimelineData;
  className?: string;
  onRangeChange?: (range: [number, number]) => void;
}

const Timeline: React.FC<TimelineProps> = ({ className, data, onRangeChange }) => {
  // Mock data if missing
  const points = data?.points || Array.from({ length: 12 }).map((_, i) => ({
      date: `2024-${i+1}`,
      count: Math.floor(Math.random() * 50) + 10,
      year: 2024,
      month: i + 1
  }));


  const maxCount = Math.max(...points.map(p => p.count));

  return (
    <div className={clsx("bg-nebula-surface/90 backdrop-blur-md border border-nebula-border rounded-xl p-4 flex flex-col gap-2 shadow-xl", className)}>
        <div className="flex justify-between items-end h-16 gap-1 px-2">
            {points.map((point, idx) => {
                const heightPercent = (point.count / maxCount) * 100;
                return (
                    <div key={idx} className="flex-1 flex flex-col justify-end items-center group cursor-pointer">
                        <div
                            className="w-full bg-nebula-primary/20 hover:bg-nebula-primary/50 transition-all rounded-t-sm relative"
                            style={{ height: `${heightPercent}%` }}
                        >
                            {/* Tooltip */}
                            <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-nebula-surface px-2 py-1 rounded text-xs border border-nebula-border opacity-0 group-hover:opacity-100 whitespace-nowrap z-50 pointer-events-none transition-opacity">
                                <span className="text-nebula-primary font-bold">{point.count} stars</span>
                                <span className="text-nebula-text-muted ml-2">{point.date}</span>
                            </div>
                        </div>
                    </div>
                );
            })}
        </div>
        <div className="flex justify-between text-[10px] text-nebula-text-dim px-1 font-mono uppercase tracking-wider">
            <span>{points[0]?.date}</span>
            <span>{points[points.length - 1]?.date}</span>
        </div>

        {/* Range Slider Overlay (Conceptual) */}
        <div className="absolute inset-x-4 bottom-2 h-1 bg-gray-700/50 rounded-full overflow-hidden">
             {/* This would be a real dual-thumb slider in production */}
             <div className="h-full bg-nebula-primary/50 w-full" />
        </div>
        <input
            type="range"
            min="0"
            max="11"
            defaultValue="0"
            className="absolute inset-x-4 bottom-1 w-[calc(100%-2rem)] h-2 opacity-0 cursor-pointer"
            onChange={(e) => {
                 const start = parseInt(e.target.value);
                 if (onRangeChange) onRangeChange([start, 11]);
            }}
        />
    </div>
  );
};

export default Timeline;
