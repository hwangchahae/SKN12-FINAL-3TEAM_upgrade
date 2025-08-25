import { RandomNumberAnimation, SlotMachineCounter } from '../components/AnimatedCounter';
import { motion } from 'framer-motion';

const LandingExample = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 py-20">
      <div className="max-w-6xl mx-auto px-6">
        <h1 className="text-4xl font-bold text-center mb-12">숫자 애니메이션 예제</h1>
        
        <div className="grid md:grid-cols-3 gap-8">
          
          {/* 카드 1: 회의 시간 절약 */}
          <motion.div
            className="bg-white rounded-2xl p-8 shadow-xl"
            whileHover={{ y: -10 }}
            transition={{ duration: 0.3 }}
          >
            <h3 className="text-xl font-semibold mb-4 text-gray-800">회의 시간 절약</h3>
            <div className="text-5xl font-bold text-blue-600 mb-2">
              <RandomNumberAnimation
                finalValue={85}
                duration={2.5}
                suffix="%"
                className="tabular-nums"
              />
            </div>
            <p className="text-gray-600">평균 시간 단축률</p>
          </motion.div>

          {/* 카드 2: 처리된 회의 */}
          <motion.div
            className="bg-gradient-to-br from-purple-500 to-pink-500 text-white rounded-2xl p-8 shadow-xl"
            whileHover={{ y: -10 }}
            transition={{ duration: 0.3 }}
          >
            <h3 className="text-xl font-semibold mb-4">처리된 회의</h3>
            <div className="text-5xl font-bold mb-2">
              <RandomNumberAnimation
                finalValue={1247}
                duration={3}
                suffix="+"
                className="tabular-nums"
              />
            </div>
            <p className="text-purple-100">이번 달 처리 건수</p>
          </motion.div>

          {/* 카드 3: 생산성 향상 */}
          <motion.div
            className="bg-gradient-to-br from-green-500 to-teal-500 text-white rounded-2xl p-8 shadow-xl"
            whileHover={{ y: -10 }}
            transition={{ duration: 0.3 }}
          >
            <h3 className="text-xl font-semibold mb-4">생산성 향상</h3>
            <div className="text-5xl font-bold mb-2">
              <SlotMachineCounter
                value={234}
                duration={2}
                suffix="%"
                className="tabular-nums"
              />
            </div>
            <p className="text-green-100">팀 효율성 증가</p>
          </motion.div>
        </div>

        {/* 큰 숫자 카운터 */}
        <motion.div
          className="mt-16 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-3xl p-12 text-center shadow-2xl"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-3xl font-bold mb-6">누적 사용자 수</h2>
          <div className="text-7xl font-black">
            <RandomNumberAnimation
              finalValue={52847}
              duration={4}
              prefix=""
              suffix=" 명"
              className="tabular-nums"
              onComplete={() => console.log('Animation completed!')}
            />
          </div>
          <p className="mt-4 text-indigo-200 text-lg">매일 증가하는 사용자들이 TtalKkak을 선택했습니다</p>
        </motion.div>

        {/* 작은 통계들 */}
        <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-6">
          {[
            { label: '절약된 시간', value: 9876, suffix: '시간' },
            { label: '생성된 태스크', value: 45321, suffix: '개' },
            { label: '만족도', value: 98, suffix: '%' },
            { label: '연동 서비스', value: 127, suffix: '개' }
          ].map((stat, index) => (
            <motion.div
              key={stat.label}
              className="bg-white rounded-xl p-6 text-center shadow-lg"
              initial={{ opacity: 0, scale: 0.5 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <div className="text-3xl font-bold text-gray-800 mb-2">
                <RandomNumberAnimation
                  finalValue={stat.value}
                  duration={2 + index * 0.3}
                  suffix={stat.suffix}
                />
              </div>
              <p className="text-gray-600 text-sm">{stat.label}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default LandingExample;