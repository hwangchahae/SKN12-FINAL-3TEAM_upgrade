import { motion, useSpring, useTransform, useInView } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';

interface AnimatedCounterProps {
  from?: number;
  to: number;
  duration?: number;
  delay?: number;
  suffix?: string;
  prefix?: string;
  decimals?: number;
  className?: string;
  onComplete?: () => void;
  startOnView?: boolean;
}

const AnimatedCounter = ({
  from = 0,
  to,
  duration = 2,
  delay = 0,
  suffix = '',
  prefix = '',
  decimals = 0,
  className = '',
  onComplete,
  startOnView = true
}: AnimatedCounterProps) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });
  const [hasStarted, setHasStarted] = useState(false);
  
  const spring = useSpring(from, {
    stiffness: 70,
    damping: 30,
    restDelta: 0.001
  });
  
  const display = useTransform(spring, (value) => {
    return `${prefix}${value.toFixed(decimals)}${suffix}`;
  });

  useEffect(() => {
    if (startOnView && isInView && !hasStarted) {
      setHasStarted(true);
      setTimeout(() => {
        spring.set(to);
        setTimeout(() => {
          if (onComplete) onComplete();
        }, duration * 1000);
      }, delay * 1000);
    } else if (!startOnView && !hasStarted) {
      setHasStarted(true);
      setTimeout(() => {
        spring.set(to);
        setTimeout(() => {
          if (onComplete) onComplete();
        }, duration * 1000);
      }, delay * 1000);
    }
  }, [isInView, spring, to, duration, delay, onComplete, hasStarted, startOnView]);

  return (
    <motion.span ref={ref} className={className}>
      {display}
    </motion.span>
  );
};

export default AnimatedCounter;

export const RandomNumberAnimation = ({
  finalValue,
  duration = 3,
  className = '',
  prefix = '',
  suffix = '',
  decimals = 0,
  onComplete
}: {
  finalValue: number;
  duration?: number;
  className?: string;
  prefix?: string;
  suffix?: string;
  decimals?: number;
  onComplete?: () => void;
}) => {
  const [displayValue, setDisplayValue] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });
  const intervalRef = useRef<NodeJS.Timeout>();
  
  useEffect(() => {
    if (isInView && !isAnimating) {
      setIsAnimating(true);
      const startTime = Date.now();
      
      // 정수부와 소수부 분리
      const integerPart = Math.floor(finalValue);
      const decimalPart = finalValue - integerPart;
      
      // 최대값 계산 (정수부 기준)
      const digits = String(integerPart).length;
      const maxValue = Math.pow(10, digits) - 1;
      
      intervalRef.current = setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000;
        const progress = Math.min(elapsed / duration, 1);
        
        if (progress < 0.7) {
          // 70%까지는 완전 랜덤
          const randomNum = Math.random() * maxValue;
          setDisplayValue(randomNum);
        } else if (progress < 0.9) {
          // 70-90%는 목표값 근처에서 진동
          const speed = 1 - (progress - 0.7) / 0.2;
          const range = maxValue * speed * 0.5;
          const randomNum = finalValue + (Math.random() - 0.5) * range;
          setDisplayValue(Math.max(0, randomNum));
        } else if (progress < 1) {
          // 90-100%는 미세 조정
          const speed = 1 - (progress - 0.9) / 0.1;
          const range = maxValue * 0.1 * speed;
          const randomNum = finalValue + (Math.random() - 0.5) * range;
          setDisplayValue(Math.max(0, randomNum));
        } else {
          // 완료
          setDisplayValue(finalValue);
          clearInterval(intervalRef.current);
          if (onComplete) onComplete();
        }
      }, 30);
      
      return () => {
        if (intervalRef.current) clearInterval(intervalRef.current);
      };
    }
  }, [isInView, finalValue, duration, isAnimating, onComplete, decimals]);
  
  return (
    <motion.span
      ref={ref}
      className={className}
      initial={{ opacity: 0 }}
      animate={isInView ? { opacity: 1 } : {}}
      transition={{ duration: 0.3 }}
    >
      {prefix}{displayValue.toLocaleString('ko-KR', { 
        minimumFractionDigits: decimals, 
        maximumFractionDigits: decimals 
      })}{suffix}
    </motion.span>
  );
};

export const SlotMachineCounter = ({
  value,
  duration = 2,
  className = '',
  prefix = '',
  suffix = ''
}: {
  value: number;
  duration?: number;
  className?: string;
  prefix?: string;
  suffix?: string;
}) => {
  const [displayDigits, setDisplayDigits] = useState<string[]>(['0', '0', '0']);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });
  const valueString = String(value).padStart(3, '0');
  
  useEffect(() => {
    if (isInView) {
      const digits = valueString.split('');
      
      digits.forEach((targetDigit, index) => {
        const spinDuration = duration + (index * 0.3);
        const startTime = Date.now();
        
        const interval = setInterval(() => {
          const elapsed = (Date.now() - startTime) / 1000;
          
          if (elapsed < spinDuration) {
            setDisplayDigits(prev => {
              const newDigits = [...prev];
              newDigits[index] = String(Math.floor(Math.random() * 10));
              return newDigits;
            });
          } else {
            setDisplayDigits(prev => {
              const newDigits = [...prev];
              newDigits[index] = targetDigit;
              return newDigits;
            });
            clearInterval(interval);
          }
        }, 50);
      });
    }
  }, [isInView, valueString, duration]);
  
  return (
    <motion.div
      ref={ref}
      className={`${className} font-mono`}
      initial={{ opacity: 0, y: 20 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5 }}
    >
      {prefix}
      <span className="inline-flex">
        {displayDigits.map((digit, index) => (
          <motion.span
            key={index}
            className="inline-block mx-0.5 bg-gradient-to-b from-gray-100 to-gray-200 rounded px-2 py-1 shadow-inner"
            animate={{
              rotateX: isInView ? [0, 360, 720, 1080, 1440, 0] : 0
            }}
            transition={{
              duration: duration + (index * 0.3),
              ease: "easeOut"
            }}
          >
            {digit}
          </motion.span>
        ))}
      </span>
      {suffix}
    </motion.div>
  );
};