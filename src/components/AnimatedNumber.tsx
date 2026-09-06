import React, { useEffect, useState } from 'react';

interface AnimatedNumberProps {
  value: number; // e.g. 0.648 for 64.8%
  decimals?: number;
  suffix?: string;
  durationMs?: number;
  className?: string;
}

export const AnimatedNumber: React.FC<AnimatedNumberProps> = ({
  value,
  decimals = 1,
  suffix = '%',
  durationMs = 400,
  className = '',
}) => {
  const [displayValue, setDisplayValue] = useState<number>(value * 100);

  useEffect(() => {
    const target = value * 100;
    const start = displayValue;
    const startTime = performance.now();

    let animationFrameId: number;

    const tick = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / durationMs);
      // Ease out cubic
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = start + (target - start) * ease;
      setDisplayValue(current);

      if (progress < 1) {
        animationFrameId = requestAnimationFrame(tick);
      } else {
        setDisplayValue(target);
      }
    };

    animationFrameId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [value, durationMs]);

  return (
    <span className={className}>
      {displayValue.toFixed(decimals)}
      {suffix}
    </span>
  );
};
