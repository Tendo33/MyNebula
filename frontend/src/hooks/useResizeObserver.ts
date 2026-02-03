import { useEffect, useState, RefObject } from 'react';

export const useResizeObserver = (ref: RefObject<HTMLElement>) => {
  const [dimensions, setDimensions] = useState({ width: 1, height: 1 });

  useEffect(() => {
    if (!ref.current) return;

    const observer = new ResizeObserver((entries) => {
      if (!entries || entries.length === 0) return;
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });

    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [ref]);

  return dimensions;
};
