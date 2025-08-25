import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import ttalkkakLogo from '../assets/logo.png';
import { useState } from 'react';
import SplitText from '../components/SplitText';



const slackUrl = 'https://slack.com/app_redirect?app=YOUR_APP_ID';

const Landing = () => {
  const navigate = useNavigate();
  const [currentReview, setCurrentReview] = useState(0);
  const [isTtalKkakHovered, setIsTtalKkakHovered] = useState(false);
  const [isDashboardHovered, setIsDashboardHovered] = useState(false);
  const [heroAnimationComplete, setHeroAnimationComplete] = useState(false);
  const [card1Clicked, setCard1Clicked] = useState(false);
  const [card2Clicked, setCard2Clicked] = useState(false);
  const [card3Clicked, setCard3Clicked] = useState(false);
  const [problemCardsClicked, setProblemCardsClicked] = useState<Record<number, boolean>>({
    0: false,
    1: false,
    2: false,
    3: false,
    4: false,
    5: false
  });
  const [successCardsClicked, setSuccessCardsClicked] = useState<Record<number, boolean>>({
    0: false,
    1: false,
    2: false
  });

  return (
    <div className="min-h-screen bg-white">
      {/* 👉 여기에 버튼 추가 */}
      <div className="fixed right-4 top-1/2 transform -translate-y-1/2 z-50 flex flex-col space-y-3">
        {/* 위로 이동 */}
        <button
          onClick={() => {
            const top = document.getElementById('hero');
            top?.scrollIntoView({ behavior: 'smooth' });
          }}
          className="bg-white border border-neutral-300 rounded-full p-2 shadow hover:shadow-md hover:bg-neutral-100 transition"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
          </svg>
        </button>

        {/* 아래로 이동 */}
        <button
          onClick={() => {
            const bottom = document.getElementById('cta');
            bottom?.scrollIntoView({ behavior: 'smooth' });
          }}
          className="bg-white border border-neutral-300 rounded-full p-2 shadow hover:shadow-md hover:bg-neutral-100 transition"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {/* 헤더 */}
      <motion.header 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="fixed top-0 left-0 right-0 z-50 bg-white/90 backdrop-blur-md border-b border-neutral-100"
      >
        <div className="px-6 py-4 flex items-center justify-between">
          <motion.div
            className="flex items-center space-x-4"
            whileHover={{ scale: 1.05, rotate: 2 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <motion.img 
              src={ttalkkakLogo} 
              alt="TtalKkak Logo" 
              className="w-24 h-16"
              animate={{ 
                y: [0, -5, 0],
                rotate: [0, 1, 0]
              }}
              transition={{ 
                duration: 4,
                repeat: 0,
                ease: "easeInOut"
              }}
            />
          </motion.div>
          
          <div className="flex items-center space-x-4 pr-6">
            <motion.button
              whileHover={{ 
                scale: 1.05, 
                y: -3,
                boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)"
              }}
              whileTap={{ scale: 0.95, y: 0 }}
              transition={{ type: "spring", stiffness: 400, damping: 10 }}
              onClick={() => navigate('/login')}
              className="group relative px-6 py-3 bg-gradient-to-r from-[#e7e972] to-[#e7e972] text-black rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all duration-300 border border-[#e7e972] hover:border-[#e7e972] overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-700"></div>
              <span className="relative flex items-center space-x-2">
                <motion.svg 
                  className="w-4 h-4" 
                  viewBox="0 0 24 24" 
                  fill="currentColor"
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 2, repeat: 0, ease: "linear" }}
                >
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                </motion.svg>
                <span>로그인</span>
              </span>
            </motion.button>
          </div>
        </div>
      </motion.header>


            {/* 메인 히어로 섹션 */}
            <section
  id="hero"
  className="relative min-h-screen bg-gradient-to-b from-[#0f172a] to-[#1e293b] flex items-center justify-center text-white overflow-hidden"
>

  {/* 기존 배경 효과 (영상 위) */}
  <div className="absolute inset-0 bg-[#1e293b] z-[-1]" />

  {/* 🧨  영상 (맨 뒤) */}
  <video
    src="/3267226-uhd_3840_2160_25fps.mp4"
    autoPlay
    loop
    muted
    playsInline
    className="absolute inset-0 w-full h-full object-cover z-0 pointer-events-none blur-sm"
  />
  
  {/* 어두운 오버레이 추가 */}
  <div className="absolute inset-0 bg-black/50 z-10"></div>

  {/* 글씨 & 버튼 (맨 위, 절대 사라지지 않음) */}
  <div className="max-w-4xl mx-auto px-8 text-center relative z-20">
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 1.2, ease: "easeOut" }}
    >
              <motion.h1
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, delay: 0.3 }}
        className="text-4xl lg:text-5xl font-black tracking-tight mb-6 leading-tight text-white"
      >
        <motion.div
          initial={{ opacity: 0, y: 60 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.3 }}
          className="block mb-3"
        >
          {!heroAnimationComplete ? (
            <SplitText
              text="회의에서"
              className="text-4xl lg:text-5xl font-bold text-white"
              delay={100}
              duration={0.6}
              ease="power3.out"
              splitType="chars"
              from={{ opacity: 0, y: 40 }}
              to={{ opacity: 1, y: 0 }}
              threshold={0.1}
              rootMargin="-100px"
              textAlign="center"
              onLetterAnimationComplete={() => {
                if (!heroAnimationComplete) {
                  setTimeout(() => setHeroAnimationComplete(true), 500);
                }
              }}
            />
          ) : (
            <span className="text-4xl lg:text-5xl font-bold text-white">회의에서</span>
          )}
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 60 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 1.5 }}
          className="block"
        >
          {!heroAnimationComplete ? (
            <SplitText
              text="자동으로 분석까지"
              className="text-4xl lg:text-5xl font-bold text-white"
              delay={100}
              duration={0.6}
              ease="power3.out"
              splitType="chars"
              from={{ opacity: 0, y: 40 }}
              to={{ opacity: 1, y: 0 }}
              threshold={0.1}
              rootMargin="-100px"
              textAlign="center"
              onLetterAnimationComplete={() => {
                if (!heroAnimationComplete) {
                  setTimeout(() => setHeroAnimationComplete(true), 500);
                }
              }}
            />
          ) : (
            <span className="text-4xl lg:text-5xl font-bold text-white">자동으로 분석까지</span>
          )}
        </motion.div>
      </motion.h1>
      <motion.p 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, delay: 0.6 }}
        className="text-lg lg:text-xl font-semibold text-white mb-8" 
        style={{ textShadow: '2px 2px 6px rgba(0,0,0,0.6)' }}
      >
        음성 업로드만 하면, AI가 요약하고 업무를 생성해줍니다.<br />
        누구보다 빠르게, 누구보다 정확하게.
      </motion.p>
      <motion.div 
        className="flex flex-col sm:flex-row gap-4 justify-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, delay: 0.9 }}
      >
        <motion.a
          href={slackUrl}
          whileHover={{ 
            scale: 1.02
          }}
          whileTap={{ scale: 0.98 }}
          onMouseEnter={() => setIsTtalKkakHovered(true)}
          onMouseLeave={() => setIsTtalKkakHovered(false)}
          className="group relative px-6 py-3 rounded-lg font-semibold bg-white text-slate-800 overflow-hidden transition-all duration-300"
        >
          {/* 왼쪽에서 오른쪽으로 채워지는 효과 */}
          <div className={`absolute inset-0 bg-gradient-to-r from-[#e7e972] to-[#e7e972] transform transition-transform duration-500 ease-out ${isTtalKkakHovered ? 'translate-x-0' : '-translate-x-full'}`}></div>
          
          <span className="relative flex items-center space-x-2">
            <svg 
              className="w-4 h-4" 
              viewBox="0 0 24 24" 
              fill="currentColor"
            >
              <path d="M6 15a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 2a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm6-8a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 2a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm6 8a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 2a4 4 0 1 0 0-8 4 4 0 0 0 0 8z"/>
            </svg>
            <span>Go to TtalKkak</span>
          </span>
        </motion.a>
        
        <motion.button
          onClick={() => navigate('/dashboard')}
          whileHover={{ 
            scale: 1.02
          }}
          whileTap={{ scale: 0.98 }}
          onMouseEnter={() => setIsDashboardHovered(true)}
          onMouseLeave={() => setIsDashboardHovered(false)}
          className="group relative px-6 py-3 rounded-lg font-semibold border-2 border-white text-white overflow-hidden transition-all duration-300"
        >
          {/* 왼쪽에서 오른쪽으로 채워지는 효과 */}
          <div className={`absolute inset-0 bg-[#1e3a8a] transform transition-transform duration-500 ease-out ${isDashboardHovered ? 'translate-x-0' : '-translate-x-full'}`}></div>
          
          <span className="relative flex items-center space-x-2">
            <svg 
              className="w-4 h-4" 
              viewBox="0 0 24 24" 
              fill="currentColor"
            >
              <path d="M3 7a1 1 0 011-1h16a1 1 0 011 1v10a1 1 0 01-1 1H4a1 1 0 01-1-1V7zM3 7l2.5-4.5a1 1 0 01.8-.4h11.4a1 1 0 01.8.4L21 7H3z"/>
            </svg>
            <span>Go to Dashboard</span>
          </span>
        </motion.button>
      </motion.div>
    </motion.div>
  </div>
</section>


            {/* 1. 문제 공감 섹션 */}
      <section className="py-20 bg-neutral-50">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-4xl font-bold text-neutral-900 mb-6">
              이런 문제 겪고 있지 않나요?
            </h2>
            <p className="text-xl lg:text-2xl text-neutral-600">
              회의 후 업무 정리에 시간을 너무 많이 쓰고 있나요?
            </p>
          </motion.div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                title: '회의록 작성에 시간 낭비',
                desc: '회의 후 수동으로 회의록을 정리하는데 평균 2시간이 소요됩니다.',
                impact: '매일 2시간 낭비!',
                color: 'from-slate-700 to-slate-800'
              },
              {
                title: '업무 배정의 어려움',
                desc: '회의 내용을 바탕으로 팀원들에게 업무를 배정하는 과정이 복잡합니다.',
                impact: '복잡한 업무 배정!',
                color: 'from-slate-700 to-slate-800'
              },
              {
                title: '중요 정보 놓치기',
                desc: '긴 회의 중 핵심 내용을 놓치거나 기록하지 못하는 경우가 많습니다.',
                impact: '핵심 정보 유실!',
                color: 'from-slate-700 to-slate-800'
              },
              {
                title: '업무 추적의 어려움',
                desc: '회의에서 결정된 사항들이 실제로 진행되는지 추적하기 어렵습니다.',
                impact: '업무 추적 불가!',
                color: 'from-slate-700 to-slate-800'
              },
              {
                title: '팀 협업의 비효율',
                desc: '각자 다른 방식으로 정보를 정리해서 팀 간 소통이 원활하지 않습니다.',
                impact: '팀 소통 차단!',
                color: 'from-slate-700 to-slate-800'
              },
              {
                title: '시간과 비용 낭비',
                desc: '수동 작업으로 인한 시간 낭비가 회사의 생산성을 저하시킵니다.',
                impact: '생산성 저하!',
                color: 'from-slate-700 to-slate-800'
              }
            ].map((problem, index) => (
              <motion.div
                key={problem.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                whileHover={{ 
                  scale: 1.05, 
                  y: -10,
                  boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.25)"
                }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white rounded-2xl p-8 shadow-lg transition-all duration-300 group cursor-pointer"
              >
                {/* 상단 임팩트 배지 */}
                <div className={`bg-gradient-to-r ${problem.color} text-white px-4 py-2 rounded-full text-sm font-bold mb-4 inline-block transition-transform duration-300`}>
                  {problem.impact}
                </div>
                
                {/* 제목 */}
                <h3 className="text-2xl font-bold text-neutral-900 mb-4 group-hover:text-slate-700 transition-colors duration-300">{problem.title}</h3>
                
                {/* 설명 */}
                <p className="text-neutral-600 leading-relaxed text-lg group-hover:text-neutral-800 transition-colors duration-300">{problem.desc}</p>
                
                {/* 하단 강조선 */}
                <div className={`h-1 bg-gradient-to-r ${problem.color} mt-4 transition-all duration-500 ease-out group-hover:w-full w-0`}></div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* 2. 솔루션 소개 섹션 */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl lg:text-4xl font-bold text-neutral-900 mb-4">
              TtalKkak으로 해결하세요
            </h2>
            <p className="text-xl text-neutral-600">
              AI가 회의를 분석하고 업무를 자동으로 생성해드립니다
            </p>
          </motion.div>
          
          <div className="grid md:grid-cols-3 gap-8">
            
            <motion.div
              initial={{ opacity: 0, y: 50, scale: 0.8 }}
              whileInView={{ opacity: 1, y: 0, scale: 1 }}
              whileTap={{ 
                scale: 0.95, 
                y: 5
              }}
              onClick={() => setCard1Clicked(!card1Clicked)}
              transition={{ 
                duration: 0.8, 
                type: "spring", 
                stiffness: 300 
              }}
              className="bg-gradient-to-br from-slate-800 to-slate-900 text-white rounded-2xl p-10 text-center shadow-xl transition-all duration-300 group relative overflow-hidden cursor-pointer"
            >
              {/* 배경 패턴 */}
              <div className="absolute inset-0 opacity-10">
                <div className="absolute inset-0" style={{
                  backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
                  backgroundSize: '60px 60px'
                }}></div>
              </div>
              

              
              <motion.div 
                className="text-5xl font-black mb-3 relative z-10"
                animate={card1Clicked ? { 
                  scale: [1, 1.1, 1],
                  color: ["#ffffff", "#fbbf24", "#ffffff"]
                } : {}}
                transition={{ duration: 0.5 }}
              >
                95%
              </motion.div>
              <div className="text-xl font-bold text-white/90 mb-2 relative z-10">음성 인식 정확도</div>
              <div className="text-sm text-white/70 relative z-10">WhisperX 기반 고정밀 분석</div>
              
              {/* 하단 강조선 */}
              <div className={`h-1 bg-gradient-to-r from-green-500 to-emerald-500 mt-4 transition-all duration-500 ease-out relative z-10 ${card1Clicked ? 'w-full' : 'w-0'}`}></div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 50, scale: 0.8 }}
              whileInView={{ opacity: 1, y: 0, scale: 1 }}
              whileTap={{ 
                scale: 0.95, 
                y: 5
              }}
              onClick={() => setCard2Clicked(!card2Clicked)}
              transition={{ 
                duration: 0.8, 
                delay: 0.2,
                type: "spring", 
                stiffness: 300 
              }}
              className="bg-gradient-to-br from-slate-800 to-slate-900 text-white rounded-2xl p-10 text-center shadow-xl transition-all duration-300 group relative overflow-hidden cursor-pointer"
            >
              {/* 배경 패턴 */}
              <div className="absolute inset-0 opacity-10">
                <div className="absolute inset-0" style={{
                  backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
                  backgroundSize: '60px 60px'
                }}></div>
              </div>
              

              
              <motion.div 
                className="text-5xl font-black mb-3 relative z-10"
                animate={card2Clicked ? { 
                  scale: [1, 1.1, 1],
                  color: ["#ffffff", "#fbbf24", "#ffffff"]
                } : {}}
                transition={{ duration: 0.5 }}
              >
                2.5배
              </motion.div>
              <div className="text-xl font-bold text-white/90 mb-2 relative z-10">업무 효율성 향상</div>
              <div className="text-sm text-white/70 relative z-10">AI 자동화로 업무 속도 대폭 증가</div>
              
              {/* 하단 강조선 */}
              <div className={`h-1 bg-gradient-to-r from-blue-500 to-indigo-500 mt-4 transition-all duration-500 ease-out relative z-10 ${card2Clicked ? 'w-full' : 'w-0'}`}></div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 50, scale: 0.8 }}
              whileInView={{ opacity: 1, y: 0, scale: 1 }}
              whileTap={{ 
                scale: 0.95, 
                y: 5
              }}
              onClick={() => setCard3Clicked(!card3Clicked)}
              transition={{ 
                duration: 0.8, 
                delay: 0.4,
                type: "spring", 
                stiffness: 300 
              }}
              className="bg-gradient-to-br from-slate-800 to-slate-900 text-white rounded-2xl p-10 text-center shadow-xl transition-all duration-300 group relative overflow-hidden cursor-pointer"
            >
              {/* 배경 패턴 */}
              <div className="absolute inset-0 opacity-10">
                <div className="absolute inset-0" style={{
                  backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
                  backgroundSize: '60px 60px'
                }}></div>
              </div>
              

              
              <motion.div 
                className="text-5xl font-black mb-3 relative z-10"
                animate={card3Clicked ? { 
                  scale: [1, 1.1, 1],
                  color: ["#ffffff", "#fbbf24", "#ffffff"]
                } : {}}
                transition={{ duration: 0.5 }}
              >
                80%
              </motion.div>
              <div className="text-xl font-bold text-white/90 mb-2 relative z-10">회의록 작성 시간 절약</div>
              <div className="text-sm text-white/70 relative z-10">5분 만에 완벽한 회의록 생성</div>
              
              {/* 하단 강조선 */}
              <div className={`h-1 bg-gradient-to-r from-purple-500 to-pink-500 mt-4 transition-all duration-500 ease-out relative z-10 ${card3Clicked ? 'w-full' : 'w-0'}`}></div>
            </motion.div>

          </div>
        </div>
      </section>

      {/* 3. 사용 통계 & 성과 섹션 */}
      <section className="py-20 bg-gradient-to-br from-slate-50 to-neutral-100">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-neutral-900 mb-6">
              실제 사용 현황
            </h2>
            <p className="text-xl text-neutral-600">
              TtalKkak을 사용하는 팀들의 놀라운 성과
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
            {[
              { number: '10,000+', label: '월간 처리 회의', icon: '📊' },
              { number: '40시간', label: '평균 시간 절약', icon: '⏰' },
              { number: '95%', label: '사용자 만족도', icon: '😊' },
              { number: '300+', label: '활성 기업', icon: '🏢' }
            ].map((stat, index) => (
              <div key={stat.label} className="text-center">
                <div className="text-4xl mb-2">{stat.icon}</div>
                <div className="text-3xl lg:text-4xl font-bold text-neutral-900 mb-2">{stat.number}</div>
                <div className="text-lg text-neutral-600">{stat.label}</div>
              </div>
            ))}
          </div>

          {/* 성공 사례 */}
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                company: '글로벌 제조업체 A사',
                result: '회의 효율성 340% 향상',
                desc: '월 250회의 처리, 회의록 작성 시간 92% 단축, 연간 1,200시간 절약',
                color: 'from-blue-500 to-purple-600'
              },
              {
                company: 'AI 스타트업 B사',
                result: '업무 생성 시간 87% 단축',
                desc: 'AI 자동 업무 배정으로 팀 생산성 3.2배 향상, 월 160시간 절약',
                color: 'from-green-500 to-blue-600'
              },
              {
                company: '중견 IT기업 C사',
                result: '팀 협업 효율성 156% 향상',
                desc: '통합 업무 관리로 팀 간 소통 시간 68% 단축, 프로젝트 완료율 45% 향상',
                color: 'from-orange-500 to-red-600'
              }
            ].map((story, index) => (
              <motion.div 
                key={story.company} 
                className="bg-white rounded-2xl p-8 shadow-lg transition-all duration-300 cursor-pointer"
                whileHover={{ 
                  scale: 1.05, 
                  y: -10,
                  boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.25)"
                }}
              >
                <motion.div 
                  className={`w-16 h-16 bg-gradient-to-r ${story.color} rounded-2xl flex items-center justify-center text-white text-2xl font-bold mb-6`}
                  whileHover={{ 
                    scale: [1, 1.2, 1],
                    rotate: [0, 10, -10, 0]
                  }}
                  transition={{ duration: 0.6 }}
                >
                  {index === 0 ? '🏭' : index === 1 ? '🤖' : '💻'}
                </motion.div>
                <h3 className="text-xl font-semibold text-neutral-900 mb-2">{story.company}</h3>
                <div className="text-lg font-bold text-neutral-800 mb-3">{story.result}</div>
                <p className="text-neutral-600">{story.desc}</p>
                
                {/* 하단 강조선 */}
                <div className={`h-1 bg-gradient-to-r ${story.color} mt-4 transition-all duration-500 ease-out group-hover:w-full w-0`}></div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* 4. 통합 & 연동 섹션 */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-neutral-900 mb-6">
              완벽한 생태계 연동
            </h2>
            <p className="text-xl text-neutral-600">
              이미 사용 중인 도구들과 원활하게 연동됩니다
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                category: '협업 도구',
                tools: ['Slack', 'Microsoft Teams', 'Zoom'],
                icon: '💬',
                desc: '실시간 메시징과 화상회의 플랫폼과 완벽 연동'
              },
              {
                category: '프로젝트 관리',
                tools: ['Jira', 'Notion', 'Asana'],
                icon: '📋',
                desc: '업무 생성부터 추적까지 원활한 워크플로우'
              },
              {
                category: '개발 도구',
                tools: ['GitHub', 'GitLab', 'Bitbucket'],
                icon: '💻',
                desc: '개발팀의 코드 관리와 이슈 추적 연동'
              },
              {
                category: '문서 관리',
                tools: ['Google Docs', 'Microsoft 365', 'Dropbox'],
                icon: '📄',
                desc: '회의록과 문서를 클라우드에 자동 저장'
              },
              {
                category: 'CRM & 마케팅',
                tools: ['Salesforce', 'HubSpot', 'Mailchimp'],
                icon: '📈',
                desc: '고객 관리와 마케팅 활동 연동'
              },
              {
                category: 'API 연동',
                tools: ['REST API', 'Webhook', 'Custom Integration'],
                icon: '🔗',
                desc: '자체 시스템과의 맞춤형 연동 지원'
              }
            ].map((integration, index) => (
              <div
                key={integration.category}
                className="bg-neutral-50 rounded-2xl p-8 hover:bg-white hover:shadow-lg transition-all duration-300"
              >
                <div className="text-3xl mb-4">{integration.icon}</div>
                <h3 className="text-xl font-semibold text-neutral-900 mb-3">{integration.category}</h3>
                <div className="flex flex-wrap gap-2 mb-4">
                  {integration.tools.map((tool) => (
                    <span key={tool} className="px-3 py-1 bg-white rounded-full text-sm font-medium text-neutral-700 border border-neutral-200">
                      {tool}
                    </span>
                  ))}
                </div>
                <p className="text-neutral-600 text-sm">{integration.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 5. ROI & 비용 절약 섹션 */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-neutral-900 mb-6 hover:scale-105 hover:text-[#e7e972] hover:drop-shadow-[0_0_20px_rgba(231,233,114,0.5)] transition-all duration-300">
              투자 대비 효과
            </h2>
            <p className="text-xl text-neutral-600">
              TtalKkak 도입으로 얻을 수 있는 구체적인 효과
            </p>
          </div>
          
          {/* 1행: 시간 절약 효과 시각화 */}
          <div className="mb-16">
            <h3 className="text-3xl font-bold text-neutral-900 mb-8 text-center">⏰ 시간 절약 효과</h3>
            <div className="grid md:grid-cols-4 gap-6">
              {[
                { item: '회의록 작성', savings: '87%', before: '2시간', after: '15분', color: 'from-emerald-500 to-green-600' },
                { item: '업무 배정', savings: '92%', before: '45분', after: '3분', color: 'from-blue-500 to-indigo-600' },
                { item: '정보 검색', savings: '78%', before: '25분', after: '5분', color: 'from-purple-500 to-pink-600' },
                { item: '팀 소통', savings: '71%', before: '1시간', after: '17분', color: 'from-orange-500 to-red-600' }
              ].map((effect, index) => (
                <div key={effect.item} className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-300">
                  <div className="text-center mb-4">
                    <span className="font-bold text-neutral-900 text-lg">{effect.item}</span>
                    <div className="text-sm text-neutral-600 mt-1">
                      {effect.before} → {effect.after}
                    </div>
                  </div>
                  
                                     {/* 전문적인 원형 도넛 차트 */}
                   <div className="mb-4 flex justify-center">
                     <div className="relative w-28 h-28">
                       <svg className="w-28 h-28 transform -rotate-90" viewBox="0 0 56 56">
                         {/* 배경 원 */}
                         <circle
                           cx="28"
                           cy="28"
                           r="24"
                           fill="none"
                           className="stroke-gray-200"
                           strokeWidth="3"
                         />
                         {/* 데이터 원 - 그라데이션 */}
                         <defs>
                           <linearGradient id={`timeGradient-${index}`} x1="0%" y1="0%" x2="100%" y2="100%">
                             <stop offset="0%" className={`${effect.color.split(' ')[0]}`} stopOpacity="1"/>
                             <stop offset="100%" className={`${effect.color.split(' ')[1]}`} stopOpacity="1"/>
                           </linearGradient>
                         </defs>
                         <circle
                           cx="28"
                           cy="28"
                           r="24"
                           fill="none"
                           stroke={`url(#timeGradient-${index})`}
                           strokeWidth="4"
                           strokeDasharray={`${parseInt(effect.savings) * 1.51} 100`}
                           strokeLinecap="round"
                           className="drop-shadow-lg"
                         />
                         {/* 중앙 작은 원 */}
                         <circle
                           cx="28"
                           cy="28"
                           r="16"
                           fill="white"
                           className="drop-shadow-md"
                         />
                       </svg>
                       {/* 중앙 텍스트 */}
                       <div className="absolute inset-0 flex items-center justify-center">
                         <div className="text-center">
                           <div className="text-2xl font-black text-neutral-800">{effect.savings}</div>
                           <div className="text-sm font-bold text-neutral-600">절약</div>
                         </div>
                       </div>
                     </div>
                   </div>
                  
                                     {/* 절약률 강조 */}
                   <div className="text-center bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl p-3">
                     <div className="text-2xl font-black text-neutral-800">{effect.savings}</div>
                     <div className="text-sm font-bold text-neutral-600">절약률</div>
                   </div>
                   
                   {/* 시간 비교 */}
                   <div className="text-center mt-3">
                     <div className="text-sm text-neutral-500 line-through">{effect.before}</div>
                     <div className="text-lg font-bold text-neutral-800">{effect.after}</div>
                   </div>
                   
                   {/* 시간 절약 트렌드 그래프 */}
                   <div className="mt-3 h-12">
                     <svg className="w-full h-full" viewBox="0 0 100 48" preserveAspectRatio="none">
                       <defs>
                         <linearGradient id={`trendGradient-${index}`} x1="0%" y1="0%" x2="100%" y2="100%">
                           <stop offset="0%" className={`${effect.color.split(' ')[0]}`} stopOpacity="0.8"/>
                           <stop offset="100%" className={`${effect.color.split(' ')[1]}`} stopOpacity="0.3"/>
                         </linearGradient>
                       </defs>
                       {/* 배경 영역 */}
                       <path
                         d={`M0,40 L25,${40 - parseInt(effect.savings) * 0.4} L50,${40 - parseInt(effect.savings) * 0.4} L75,${40 - parseInt(effect.savings) * 0.4} L100,${40 - parseInt(effect.savings) * 0.4} L100,40 Z`}
                         fill={`url(#trendGradient-${index})`}
                         opacity="0.2"
                       />
                       {/* 메인 선 */}
                       <path
                         d={`M0,40 L25,${40 - parseInt(effect.savings) * 0.4} L50,${40 - parseInt(effect.savings) * 0.4} L75,${40 - parseInt(effect.savings) * 0.4} L100,${40 - parseInt(effect.savings) * 0.4}`}
                         fill="none"
                         stroke={`url(#trendGradient-${index})`}
                         strokeWidth="2"
                         className="drop-shadow-sm"
                       />
                       {/* 데이터 포인트 */}
                       <circle cx="0" cy="40" r="2" fill={`url(#trendGradient-${index})`} />
                       <circle cx="25" cy={40 - parseInt(effect.savings) * 0.4} r="2" fill={`url(#trendGradient-${index})`} />
                       <circle cx="50" cy={40 - parseInt(effect.savings) * 0.4} r="2" fill={`url(#trendGradient-${index})`} />
                       <circle cx="75" cy={40 - parseInt(effect.savings) * 0.4} r="2" fill={`url(#trendGradient-${index})`} />
                       <circle cx="100" cy={40 - parseInt(effect.savings) * 0.4} r="2" fill={`url(#trendGradient-${index})`} />
                     </svg>
                   </div>
                </div>
              ))}
            </div>
          </div>

          {/* 2행: 비용 절약 효과 시각화 */}
          <div className="mb-16">
            <h3 className="text-3xl font-bold text-neutral-900 mb-8 text-center">💰 비용 절약 효과</h3>
            <div className="grid md:grid-cols-4 gap-6">
              {[
                { category: '인건비 절약', amount: '180만원', percentage: 23, desc: '회의록 작성 시간 단축', color: 'from-blue-500 to-indigo-600' },
                { category: '생산성 향상', amount: '220만원', percentage: 28, desc: '업무 자동화로 인한 효율성 증대', color: 'from-green-500 to-emerald-600' },
                { category: '의사결정 속도', amount: '120만원', percentage: 15, desc: '빠른 정보 공유로 인한 기회비용 절약', color: 'from-purple-500 to-pink-600' },
                { category: '팀 협업 개선', amount: '80만원', percentage: 10, desc: '소통 비용 절약 및 오류 감소', color: 'from-orange-500 to-red-600' }
              ].map((saving, index) => (
                <div key={saving.category} className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-300">
                  <div className="text-center mb-4">
                    <span className="font-bold text-neutral-900 text-lg">{saving.category}</span>
                  </div>
                  
                                     {/* 전문적인 수평 막대 차트 */}
                   <div className="mb-4">
                     <div className="space-y-4">
                       {/* 메인 막대 */}
                       <div className="relative">
                         <div className="w-full bg-gray-100 rounded-xl h-10 shadow-inner border border-gray-200">
                           <div 
                             className={`bg-gradient-to-r ${saving.color} h-10 rounded-xl transition-all duration-1000 ease-out shadow-lg relative overflow-hidden`}
                             style={{ width: `${saving.percentage}%` }}
                           >
                             {/* 그라데이션 오버레이 효과 */}
                             <div className="absolute inset-0 bg-gradient-to-r from-white/20 via-transparent to-white/20 animate-pulse"></div>
                           </div>
                         </div>
                         {/* 퍼센트 라벨 */}
                         <div className="absolute right-0 top-0 -mt-12">
                           <span className="text-sm font-bold text-neutral-700 bg-white px-3 py-1 rounded-lg shadow-lg border border-gray-200">
                             {saving.percentage}%
                           </span>
                         </div>
                         {/* 그리드 라인 */}
                         <div className="absolute inset-0 flex justify-between items-center pointer-events-none">
                           <div className="w-px h-full bg-gray-300 opacity-30"></div>
                           <div className="w-px h-full bg-gray-300 opacity-30"></div>
                           <div className="w-px h-full bg-gray-300 opacity-30"></div>
                           <div className="w-px h-full bg-gray-300 opacity-30"></div>
                         </div>
                       </div>
                       
                                              {/* 금액 표시 */}
                       <div className="text-center">
                         <div className="text-2xl font-black text-neutral-800">{saving.amount}</div>
                         <div className="text-sm font-bold text-neutral-600">월 절약</div>
                       </div>
                     </div>
                   </div>
                   
                   {/* 설명 텍스트 */}
                   <p className="text-xs text-neutral-600 text-center leading-relaxed mt-3">{saving.desc}</p>
                </div>
              ))}
            </div>
          </div>

          {/* 3행: ROI 시각화 */}
          <div>
            <h3 className="text-3xl font-bold text-neutral-900 mb-8 text-center">📈 투자 대비 수익률 (ROI)</h3>
            <div className="grid md:grid-cols-3 gap-8">
              {/* 월 절약 효과 */}
              <div className="bg-gradient-to-br from-emerald-500 to-green-600 rounded-2xl p-8 text-white text-center shadow-xl">
                <div className="text-4xl mb-2">💰</div>
                <div className="text-2xl font-bold mb-2">월 절약 효과</div>
                <div className="text-4xl font-black mb-2">600만원</div>
                <div className="text-sm opacity-90">연간 7,200만원 절약</div>
              </div>
              
              {/* ROI 달성 기간 */}
              <div className="bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl p-8 text-white text-center shadow-xl relative overflow-hidden">
                <div className="text-4xl mb-2">⚡</div>
                <div className="text-2xl font-bold mb-2">ROI 달성 기간</div>
                <div className="text-4xl font-black mb-2">3개월</div>
                <div className="text-sm opacity-90">초기 투자비 회수 완료</div>
                
                {/* 배경 선 그래프 */}
                <div className="absolute bottom-4 left-4 right-4 h-16 opacity-20">
                  <svg className="w-full h-full" viewBox="0 0 100 40" preserveAspectRatio="none">
                    <path
                      d="M0,30 Q25,20 50,25 T100,15"
                      fill="none"
                      stroke="white"
                      strokeWidth="2"
                      className="drop-shadow-sm"
                    />
                    <circle cx="0" cy="30" r="2" fill="white" />
                    <circle cx="25" cy="20" r="2" fill="white" />
                    <circle cx="50" cy="25" r="2" fill="white" />
                    <circle cx="75" cy="18" r="2" fill="white" />
                    <circle cx="100" cy="15" r="2" fill="white" />
                  </svg>
                </div>
              </div>
              
              {/* 연간 수익률 */}
              <div className="bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl p-8 text-white text-center shadow-xl relative overflow-hidden">
                <div className="text-4xl mb-2">📊</div>
                <div className="text-2xl font-bold mb-2">연간 수익률</div>
                <div className="text-4xl font-black mb-2">400%</div>
                <div className="text-sm opacity-90">투자 대비 4배 수익</div>
                
                {/* 배경 막대 그래프 */}
                <div className="absolute bottom-4 left-4 right-4 h-16 opacity-20">
                  <svg className="w-full h-full" viewBox="0 0 100 40" preserveAspectRatio="none">
                    <rect x="10" y="25" width="8" height="15" fill="white" rx="1" />
                    <rect x="25" y="20" width="8" height="20" fill="white" rx="1" />
                    <rect x="40" y="15" width="8" height="25" fill="white" rx="1" />
                    <rect x="55" y="10" width="8" height="30" fill="white" rx="1" />
                    <rect x="70" y="5" width="8" height="35" fill="white" rx="1" />
                    <rect x="85" y="0" width="8" height="40" fill="white" rx="1" />
                  </svg>
                </div>
              </div>
            </div>
          </div>
          

        </div>
      </section>


      {/* 기능 소개 섹션 */}
      <section className="py-16 bg-neutral-50">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-neutral-900 mb-4">
              어떻게 동작하는지 확인해보세요
            </h2>
            <p className="text-xl text-neutral-600">
              단순한 회의 녹음에서 시작해서 완성된 업무까지
            </p>
          </motion.div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300"
            >
                             <div className="w-12 h-12 bg-slate-100 rounded-xl flex items-center justify-center mb-6">
                 <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                 </svg>
               </div>
              <h3 className="text-xl font-semibold text-neutral-900 mb-4">회의 녹음 업로드</h3>
              <p className="text-neutral-600">Slack 채널에 회의 녹음 파일을 업로드하면 자동으로 분석이 시작됩니다.</p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300"
            >
                             <div className="w-12 h-12 bg-gray-100 rounded-xl flex items-center justify-center mb-6">
                 <svg className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                 </svg>
               </div>
              <h3 className="text-xl font-semibold text-neutral-900 mb-4">AI 분석 및 업무 생성</h3>
              <p className="text-neutral-600">WhisperX로 음성을 텍스트로 변환하고, AI가 중요한 업무를 자동으로 추출합니다.</p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300"
            >
                             <div className="w-12 h-12 bg-zinc-100 rounded-xl flex items-center justify-center mb-6">
                 <svg className="w-8 h-8 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                 </svg>
               </div>
              <h3 className="text-xl font-semibold text-neutral-900 mb-4">스마트 업무 배정</h3>
              <p className="text-neutral-600">팀원들의 스킬과 현재 업무량을 고려하여 최적의 담당자를 자동으로 배정합니다.</p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* 4. 데모 영상 섹션 */}
      <section className="py-24 bg-neutral-50">
          <div className="max-w-5xl mx-auto px-6 text-center">
            <h2 className="text-3xl lg:text-4xl font-bold text-neutral-900 mb-6 tracking-tight">
              실제 작동 화면을 확인해보세요
            </h2>
            <p className="text-xl lg:text-2xl text-neutral-700 mb-8 leading-relaxed">
              회의 녹음만 업로드하면<br />
              AI가 업무 생성까지 자동으로 처리합니다.
            </p>

            <div className="relative max-w-4xl mx-auto rounded-2xl overflow-hidden shadow-xl transition-transform hover:scale-105">
              <iframe
                src="https://www.youtube.com/embed/YOUR_DEMO_VIDEO_ID" // 실제 링크로 교체
                title="TtalKkak Demo"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;"
                allowFullScreen
                className="w-full aspect-video"
              ></iframe>
            </div>

            <p className="mt-6 text-neutral-500 text-sm">
              👆 클릭해서 실제 흐름을 영상으로 확인해보세요
            </p>
          </div>
        </section>


              {/* 고객 후기 섹션 */}
      <section className="py-24 bg-white">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl lg:text-4xl font-bold text-neutral-900 mb-4">
              실제 사용자들의 생생한 후기
            </h2>
            <p className="text-lg lg:text-xl text-neutral-600">
              TtalKkak을 실제로 사용하고 있는 팀들의 솔직한 경험담입니다
            </p>
          </motion.div>

          <div className="flex justify-center mb-8">
            <div className="flex space-x-2">
              <motion.button 
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setCurrentReview(Math.max(0, currentReview - 1))}
                className="w-10 h-10 rounded-full bg-neutral-200 hover:bg-neutral-300 transition-colors flex items-center justify-center"
              >
                <svg className="w-5 h-5 text-neutral-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </motion.button>
              <motion.button 
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setCurrentReview(Math.min(3, currentReview + 1))}
                className="w-10 h-10 rounded-full bg-neutral-200 hover:bg-neutral-300 transition-colors flex items-center justify-center"
              >
                <svg className="w-5 h-5 text-neutral-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </motion.button>
            </div>
          </div>

          {/* 인디케이터 */}
          <div className="flex justify-center mb-8">
            <div className="flex space-x-2">
              {[0, 1, 2, 3].map((index) => (
                <motion.button
                  key={index}
                  onClick={() => setCurrentReview(index)}
                  className={`w-3 h-3 rounded-full transition-colors ${
                    currentReview === index ? 'bg-neutral-600' : 'bg-neutral-300'
                  }`}
                  whileHover={{ scale: 1.2 }}
                />
              ))}
            </div>
          </div>

          <div className="relative overflow-hidden">
            <motion.div 
              className="flex transition-transform duration-500 ease-in-out"
              style={{ transform: `translateX(-${currentReview * 100}%)` }}
            >
              {/* 첫 번째 후기 */}
              <div className="w-full flex-shrink-0 px-4">
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6 }}
                  className="bg-neutral-50 rounded-2xl p-8 hover:shadow-lg transition-all duration-300"
                >
                  <div className="flex items-center mb-6">
                    <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center text-white font-semibold text-lg mr-4">
                      김개발
                    </div>
                    <div>
                      <h3 className="font-semibold text-neutral-900">김개발</h3>
                      <p className="text-sm text-neutral-600">개발팀 팀장</p>
                      <div className="flex mt-1">
                        {[...Array(5)].map((_, i) => (
                          <svg key={i} className="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                          </svg>
                        ))}
                      </div>
                    </div>
                  </div>
                  <p className="text-neutral-700 leading-relaxed">
                    회의가 끝나고 나면 항상 회의록 정리하는 데 시간이 오래 걸렸는데, TtalKkak 덕분에 그런 번거로움이 완전히 사라졌어요! 음성 파일만 업로드하면 AI가 알아서 요약하고 업무까지 생성해주니까 정말 편리합니다.
                  </p>
                </motion.div>
              </div>

              {/* 두 번째 후기 */}
              <div className="w-full flex-shrink-0 px-4">
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.1 }}
                  className="bg-neutral-50 rounded-2xl p-8 hover:shadow-lg transition-all duration-300"
                >
                  <div className="flex items-center mb-6">
                    <div className="w-16 h-16 bg-gradient-to-br from-pink-400 to-red-500 rounded-full flex items-center justify-center text-white font-semibold text-lg mr-4">
                      박마케팅
                    </div>
                    <div>
                      <h3 className="font-semibold text-neutral-900">박마케팅</h3>
                      <p className="text-sm text-neutral-600">마케팅팀 매니저</p>
                      <div className="flex mt-1">
                        {[...Array(5)].map((_, i) => (
                          <svg key={i} className="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                          </svg>
                        ))}
                      </div>
                    </div>
                  </div>
                  <p className="text-neutral-700 leading-relaxed">
                    평소에 회의 내용을 업무로 연결하는 게 어려웠는데, TtalKkak의 스마트 업무 배정 기능 덕분에 그 문제가 완전히 해결됐어요! 팀원들의 스킬과 업무량을 고려해서 자동으로 배정해주니까 정말 효율적입니다!!!
                  </p>
                </motion.div>
              </div>

              {/* 세 번째 후기 */}
              <div className="w-full flex-shrink-0 px-4">
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.2 }}
                  className="bg-neutral-50 rounded-2xl p-8 hover:shadow-lg transition-all duration-300"
                >
                  <div className="flex items-center mb-6">
                    <div className="w-16 h-16 bg-gradient-to-br from-green-400 to-blue-500 rounded-full flex items-center justify-center text-white font-semibold text-lg mr-4">
                      최기획
                    </div>
                    <div>
                      <h3 className="font-semibold text-neutral-900">최기획</h3>
                      <p className="text-sm text-neutral-600">기획팀 대리</p>
                      <div className="flex mt-1">
                        {[...Array(5)].map((_, i) => (
                          <svg key={i} className="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                          </svg>
                        ))}
                      </div>
                    </div>
                  </div>
                  <p className="text-neutral-700 leading-relaxed">
                    WhisperX 음성 인식 정확도가 정말 놀라워요. 한국어 회의 내용을 거의 완벽하게 텍스트로 변환해주고, AI가 핵심 업무만 선별해서 정리해주니까 회의 효율성이 2배 이상 향상됐습니다.
                  </p>
                </motion.div>
              </div>

              {/* 네 번째 후기 */}
              <div className="w-full flex-shrink-0 px-4">
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.3 }}
                  className="bg-neutral-50 rounded-2xl p-8 hover:shadow-lg transition-all duration-300"
                >
                  <div className="flex items-center mb-6">
                    <div className="w-16 h-16 bg-gradient-to-br from-orange-400 to-red-500 rounded-full flex items-center justify-center text-white font-semibold text-lg mr-4">
                      이디자인
                    </div>
                    <div>
                      <h3 className="font-semibold text-neutral-900">이디자인</h3>
                      <p className="text-sm text-neutral-600">디자인팀 팀장</p>
                      <div className="flex mt-1">
                        {[...Array(5)].map((_, i) => (
                          <svg key={i} className="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                          </svg>
                        ))}
                      </div>
                    </div>
                  </div>
                  <p className="text-neutral-700 leading-relaxed">
                    디자인 리뷰 회의가 끝나면 항상 피드백 정리가 어려웠는데, TtalKkak이 모든 피드백을 자동으로 정리해주니까 정말 편해졌어요! 팀원들이 각자 받은 피드백을 놓치지 않고 모두 반영할 수 있게 됐습니다.
                  </p>
                </motion.div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

              {/* 3. FAQ 섹션 */}
        <section className="py-20 bg-neutral-50">
          <div className="max-w-5xl mx-auto px-6">
            <h2 className="text-3xl lg:text-4xl font-bold text-center text-neutral-900 mb-12">
              자주 묻는 질문
            </h2>
            <div className="space-y-6">
              {[
                {
                  q: '슬랙 외에도 지원하나요?',
                  a: '현재는 슬랙만 지원하지만 추후 다른 협업 툴과의 연동도 예정되어 있습니다.',
                },
                {
                  q: '보안은 어떻게 처리되나요?',
                  a: '모든 데이터는 암호화되어 전송되며, 저장된 정보는 내부 정책에 따라 자동 삭제됩니다.',
                },
                {
                  q: '실시간으로 분석되나요?',
                  a: '네, 녹음 업로드 후 수초 내로 분석이 시작되어 업무 생성까지 자동으로 처리됩니다.',
                },
                {
                  q: '한국어 회의도 정확하게 인식되나요?',
                  a: 'WhisperX의 한국어 음성 인식 정확도는 95% 이상으로 매우 높습니다.',
                },
                {
                  q: '무료 체험 기간은 얼마나 되나요?',
                  a: '14일 무료 체험을 제공하며, 신용카드 정보 없이 바로 시작할 수 있습니다.',
                },
              ].map(({ q, a }, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: i * 0.1 }}
                  className="bg-white border border-neutral-200 rounded-2xl p-6 md:p-8 hover:shadow-lg transition-all"
                >
                  <h3 className="text-lg lg:text-xl font-semibold text-neutral-800 mb-3">
                    Q. {q}
                  </h3>
                  <p className="text-base lg:text-lg text-neutral-600 leading-relaxed">
                    A. {a}
                  </p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>



      {/* CTA 섹션 */}
      <section id="cta" className="py-20 bg-gradient-to-br from-[#e7e972] via-[#f0f085] via-[#d4d45a] to-[#c1c142]">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-3xl lg:text-4xl font-bold text-slate-800 mb-6">
              지금 바로 시작하세요
            </h2>
            <p className="text-xl text-slate-700 mb-8">
              무료 14일 체험 • 5분 설치 • 24/7 고객 지원
            </p>
            <div className="flex flex-col sm:flex-row gap-6 justify-center">
              <motion.a
                href={slackUrl}
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="group relative px-8 py-4 bg-gradient-to-r from-white to-slate-100 text-slate-800 rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-300 flex items-center justify-center space-x-3 overflow-hidden"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-700"></div>
                <svg className="w-5 h-5 relative z-10" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M6 15a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 2a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm6-8a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 2a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm6 8a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 2a4 4 0 1 0 0-8 4 4 0 0 0 0 8z"/>
                </svg>
                <span className="relative z-10">Slack에서 시작하기</span>
              </motion.a>
              <motion.button
                onClick={() => navigate('/dashboard')}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="group relative px-8 py-4 border-2 border-slate-700 text-slate-700 rounded-xl font-semibold text-lg hover:bg-slate-700 hover:text-white transition-all duration-300 overflow-hidden"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-slate-700/0 via-slate-700/10 to-slate-700/0 transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-700"></div>
                <span className="relative z-10">데모 체험하기</span>
              </motion.button>
            </div>
            <p className="mt-6 text-slate-600 text-sm">
              💡 5분만에 설정하고 바로 사용해보세요
            </p>
          </motion.div>
        </div>
      </section>

      {/* 푸터 */}
      <footer className="bg-neutral-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <motion.div className="flex items-center space-x-3 mb-4">
                <img src={ttalkkakLogo} alt="TtalKkak Logo" className="w-6 h-6" />
                <span className="text-xl font-bold">TtalKkak</span>
              </motion.div>
              <p className="text-neutral-400 mb-4">
                AI 기반 스마트 업무 관리 플랫폼으로<br />
                회의에서 업무까지 자동화하세요.
              </p>
              <div className="flex space-x-4">
                <a href="#" className="text-neutral-400 hover:text-white transition-colors">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/>
                  </svg>
                </a>
                <a href="#" className="text-neutral-400 hover:text-white transition-colors">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                  </svg>
                </a>
                <a href="#" className="text-neutral-400 hover:text-white transition-colors">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/>
                  </svg>
                </a>
              </div>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold mb-4">제품</h3>
              <ul className="space-y-2 text-neutral-400">
                <li><a href="#" className="hover:text-white transition-colors">기능</a></li>
                <li><a href="#" className="hover:text-white transition-colors">가격</a></li>
                <li><a href="#" className="hover:text-white transition-colors">통합</a></li>
                <li><a href="#" className="hover:text-white transition-colors">API</a></li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold mb-4">지원</h3>
              <ul className="space-y-2 text-neutral-400">
                <li><a href="#" className="hover:text-white transition-colors">도움말</a></li>
                <li><a href="#" className="hover:text-white transition-colors">문서</a></li>
                <li><a href="#" className="hover:text-white transition-colors">커뮤니티</a></li>
                <li><a href="#" className="hover:text-white transition-colors">연락처</a></li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold mb-4">회사</h3>
              <ul className="space-y-2 text-neutral-400">
                <li><a href="#" className="hover:text-white transition-colors">소개</a></li>
                <li><a href="#" className="hover:text-white transition-colors">블로그</a></li>
                <li><a href="#" className="hover:text-white transition-colors">채용</a></li>
                <li><a href="#" className="hover:text-white transition-colors">뉴스</a></li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-neutral-800 mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-neutral-400 text-sm">
              © 2024 TtalKkak. All rights reserved.
            </p>
            <div className="flex space-x-6 mt-4 md:mt-0">
              <a href="#" className="text-neutral-400 hover:text-white text-sm transition-colors">Privacy Policy</a>
              <a href="#" className="text-neutral-400 hover:text-white text-sm transition-colors">Terms of Service</a>
              <a href="#" className="text-neutral-400 hover:text-white text-sm transition-colors">Cookie Policy</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;