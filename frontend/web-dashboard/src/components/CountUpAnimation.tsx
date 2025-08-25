import { useEffect, useState, useRef } from 'react';
import { motion, useInView } from 'framer-motion';

interface CountUpAnimationProps {
  end: number;
  duration?: number;
  suffix?: string;
  decimals?: number;
}

const CountUpAnimation = ({ 
  end, 
  duration = 2, 
  suffix = '',
  decimals = 0 
}: CountUpAnimationProps) => {
  const [count, setCount] = useState('0');
  const ref = useRef(null);
  const isInView = useInView(ref, { once: false, margin: "-100px" }); // once: false로 변경
  const [hasStarted, setHasStarted] = useState(false);

  useEffect(() => {
    if (isInView) {
      // 화면에 들어올 때마다 리셋하고 다시 시작
      setCount('0');
      const startTime = Date.now();
      const endTime = startTime + duration * 1000;
      let animationId: number;

      const updateCount = () => {
        const now = Date.now();
        const progress = Math.min((now - startTime) / (duration * 1000), 1);
        
        // easeOutCubic 이징 함수
        const eased = 1 - Math.pow(1 - progress, 3);
        const currentValue = eased * end;
        
        // 숫자 포맷팅
        if (decimals > 0) {
          setCount(currentValue.toFixed(decimals));
        } else {
          setCount(Math.floor(currentValue).toString());
        }

        if (now < endTime) {
          animationId = requestAnimationFrame(updateCount);
        } else {
          // 최종값 확실히 설정
          if (decimals > 0) {
            setCount(end.toFixed(decimals));
          } else {
            setCount(Math.floor(end).toString());
          }
        }
      };

      animationId = requestAnimationFrame(updateCount);

      return () => {
        if (animationId) {
          cancelAnimationFrame(animationId);
        }
      };
    } else {
      // 화면을 벗어나면 0으로 리셋
      setCount('0');
    }
  }, [end, duration, isInView, decimals]);

  return (
    <motion.span
      ref={ref}
      initial={{ opacity: 0, scale: 0.5 }}
      animate={isInView ? { opacity: 1, scale: 1 } : {}}
      transition={{ duration: 0.5 }}
    >
      {count}{suffix}
    </motion.span>
  );
};

export default CountUpAnimation;