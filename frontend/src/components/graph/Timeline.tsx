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
    <div className={clsx("bg-white border border-border-light rounded-md p-3 flex flex-col gap-2 shadow-sm", className)}>
        <div className="flex justify-between items-end h-12 gap-1 px-1">
            {points.map((point, idx) => {
                const heightPercent = (point.count / maxCount) * 100;
                return (
                    <div key={idx} className="flex-1 flex flex-col justify-end items-center group cursor-pointer">
                        <div
                            className="w-full bg-bg-hover hover:bg-action-primary/60 transition-all rounded-t-[2px] relative"
                            style={{ height: `${heightPercent}%` }}
                        >
                            {/* Tooltip */}
                            <div className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 bg-text-main text-white px-2 py-0.5 rounded-sm text-[10px] opacity-0 group-hover:opacity-100 whitespace-nowrap z-50 pointer-events-none transition-opacity">
                                <span className="font-semibold">{point.count} stars</span>
                                <span className="opacity-80 ml-1">{point.date}</span>
                            </div>
                        </div>
                    </div>
                );
            })}
        </div>
        <div className="flex justify-between text-[10px] text-text-muted px-1 font-mono uppercase tracking-wider">
            <span>{points[0]?.date}</span>
            <span>{points[points.length - 1]?.date}</span>
        </div>

        {/* Range Slider Overlay (Conceptual) */}
        <div className="absolute inset-x-3 bottom-2 h-0.5 bg-border-light rounded-full overflow-hidden">
             {/* This would be a real dual-thumb slider in production */}
             <div className="h-full bg-action-primary/30 w-full" />
        </div>
        <input
            type="range"
            min="0"
            max="11"
            defaultValue="0"
            className="absolute inset-x-3 bottom-1 w-[calc(100%-1.5rem)] h-2 opacity-0 cursor-pointer"
            onChange={(e) => {
                 const start = parseInt(e.target.value);
                 if (onRangeChange) onRangeChange([start, 11]);
            }}
        />
    </div>
  );
};

export default Timeline;
