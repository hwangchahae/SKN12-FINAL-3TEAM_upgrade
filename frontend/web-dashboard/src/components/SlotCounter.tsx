import { useEffect, useState, useRef } from 'react';
import { motion, useInView } from 'framer-motion';

interface SlotCounterProps {
  value: string;
  duration?: number;
}

const SlotCounter = ({ value, duration = 2 }: SlotCounterProps) => {
  const [displayChars, setDisplayChars] = useState<string[]>(value.split(''));
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });
  const [hasStarted, setHasStarted] = useState(false);

  useEffect(() => {
    if (isInView && !hasStarted) {
      setHasStarted(true);
      const chars = value.split('');
      const intervals: NodeJS.Timeout[] = [];
      
      chars.forEach((targetChar, index) => {
        // 각 자리수마다 다른 타이밍
        const delay = index * 200; // 각 자리 200ms 차이
        const spinDuration = duration * 1000 + (index * 300); // 뒤로 갈수록 더 오래
        
        setTimeout(() => {
          let startTime = Date.now();
          
          const interval = setInterval(() => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / spinDuration, 1);
            
            // 속도 감소 곡선 (처음엔 빠르게, 나중엔 천천히)
            const speed = Math.pow(1 - progress, 3);
            
            if (progress < 0.9 || Math.random() < speed) {
              // 숫자만 처리, 특수문자는 그대로
              if (targetChar >= '0' && targetChar <= '9') {
                const randomDigit = Math.floor(Math.random() * 10).toString();
                setDisplayChars(prev => {
                  const newChars = [...prev];
                  newChars[index] = randomDigit;
                  return newChars;
                });
              }
            } else {
              // 최종값으로 고정
              setDisplayChars(prev => {
                const newChars = [...prev];
                newChars[index] = targetChar;
                return newChars;
              });
              clearInterval(interval);
            }
          }, 50); // 50ms마다 업데이트
          
          intervals.push(interval);
        }, delay);
      });
      
      return () => {
        intervals.forEach(interval => clearInterval(interval));
      };
    }
  }, [value, duration, isInView, hasStarted]);

  return (
    <span ref={ref} className="inline-flex">
      {displayChars.map((char, index) => (
        <motion.span
          key={index}
          className="inline-block"
          initial={{ opacity: 0.5, y: 10 }}
          animate={{ 
            opacity: 1, 
            y: 0,
          }}
          transition={{
            duration: 0.3,
            delay: index * 0.05
          }}
        >
          {char}
        </motion.span>
      ))}
    </span>
  );
};

export default SlotCounter;