import { useEffect, useState, RefObject } from 'react';

export const useResizeObserver = (ref: RefObject<HTMLElement>) => {
  const [dimensions, setDimensions] = useState({ width: 1, height: 1 });

  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;

    const apply = (width: number, height: number) => {
      const nextWidth = Math.max(1, Math.round(width));
      const nextHeight = Math.max(1, Math.round(height));
      setDimensions((prev) =>
        prev.width === nextWidth && prev.height === nextHeight
          ? prev
          : { width: nextWidth, height: nextHeight }
      );
    };

    apply(el.clientWidth, el.clientHeight);

    const observer = new ResizeObserver((entries) => {
      if (!entries || entries.length === 0) return;
      const { width, height } = entries[0].contentRect;
      apply(width, height);
    });

    observer.observe(el);
    return () => observer.disconnect();
  }, [ref]);

  return dimensions;
};
