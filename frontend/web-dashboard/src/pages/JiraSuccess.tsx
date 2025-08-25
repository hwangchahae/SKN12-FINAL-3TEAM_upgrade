import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { CheckCircle, X, ArrowRight, Sparkles, Check, ExternalLink, Zap, Link, FileText, Database } from "lucide-react";
import ttalkkakLogo from '../assets/logo.png';

const IntegrationSuccess = () => {
  const [isButtonHovered, setIsButtonHovered] = useState(false);
  const [userName, setUserName] = useState("사용자");

  // 사용자 정보 가져오기
  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        const userData = JSON.parse(storedUser);
        setUserName(userData.name || userData.displayName || "사용자");
      } catch (error) {
        console.log('사용자 정보 파싱 오류:', error);
      }
    }

    const urlParams = new URLSearchParams(window.location.search);
    const userFromUrl = urlParams.get('user');
    if (userFromUrl) {
      setUserName(userFromUrl);
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-gray-50 to-slate-100 relative overflow-hidden">
      
      {/* 배경 효과 */}
      <div className="absolute inset-0">
        {/* 부드러운 그라데이션 오버레이 */}
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-blue-500/5 via-transparent to-indigo-500/5"></div>
        
        {/* 동적 원형 요소들 */}
        <div className="absolute top-20 left-20 w-96 h-96 bg-blue-400/8 rounded-full filter blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-80 h-80 bg-indigo-400/8 rounded-full filter blur-3xl animate-pulse delay-1000"></div>
      </div>

      {/* 헤더 */}
      <motion.header 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-xl border-b border-gray-200/50 shadow-sm"
      >
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <motion.div 
            className="flex items-center"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 400, damping: 10 }}
          >
            <img src={ttalkkakLogo} alt="TtalKkak Logo" className="w-20 h-15" />
          </motion.div>
          
          <motion.button
            whileHover={{ scale: 1.1, rotate: 90 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => window.close()}
            className="p-3 text-gray-500 hover:text-gray-700 bg-gray-100/50 hover:bg-gray-200/50 rounded-full transition-all duration-300"
          >
            <X className="w-5 h-5" />
          </motion.button>
        </div>
      </motion.header>

      {/* 메인 컨텐츠 */}
      <div className="pt-32 pb-12 flex items-center justify-center p-6">
        <div className="w-full max-w-4xl">
          
          {/* 성공 애니메이션 섹션 */}
          <motion.div
            className="text-center mb-12"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            {/* 성공 체크마크 GIF */}
            <motion.div
              className="w-48 h-48 mx-auto mb-8 relative"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.3, type: "spring", stiffness: 150 }}
            >
              <img
                src="/checkmark_new.gif"
                alt="Success Checkmark Animation"
                className="w-full h-full drop-shadow-xl"
                style={{
                  filter: 'hue-rotate(240deg) saturate(3) brightness(0.8) contrast(2) sepia(1) hue-rotate(200deg)'
                }}
              />
            </motion.div>

            {/* 제목 */}
            <motion.h1
              className="text-5xl font-bold text-gray-800 mb-4"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.8 }}
            >
              <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent drop-shadow-lg">
                연동 완료!
              </span>
            </motion.h1>

            {/* 부제목 */}
            <motion.p
              className="text-xl text-gray-600 mb-8"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1, duration: 0.8 }}
            >
              <span className="text-blue-600 font-semibold">Jira</span>와의 연동이 성공적으로 완료되었습니다
            </motion.p>
          </motion.div>

          {/* 카드 그리드 */}
          <div className="grid md:grid-cols-2 gap-8 mb-12">
            
            {/* 연동 정보 카드 */}
            <motion.div
              className="bg-white/90 backdrop-blur-xl rounded-3xl p-8 border border-gray-200/50 shadow-xl"
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 1.2, duration: 0.8 }}
              whileHover={{ 
                scale: 1.02, 
                boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.15)" 
              }}
            >
              <div className="flex items-center space-x-4 mb-6">
                <motion.div 
                  className="w-16 h-16 rounded-2xl bg-white border border-gray-200 flex items-center justify-center shadow-lg overflow-hidden"
                  whileHover={{ scale: 1.1, rotate: 5 }}
                >
                  <img 
                    src="https://cdn.worldvectorlogo.com/logos/jira-1.svg" 
                    alt="Jira Logo" 
                    className="w-10 h-10 object-contain"
                  />
                </motion.div>
                <div>
                  <h3 className="text-2xl font-bold text-gray-800">
                    {userName}의 Jira
                  </h3>
                  <p className="text-gray-500">프로젝트 연동됨</p>
                </div>
                <motion.div 
                  className="ml-auto w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center"
                  whileHover={{ scale: 1.2 }}
                >
                  <CheckCircle className="w-7 h-7 text-blue-600" />
                </motion.div>
              </div>
              
              <div className="space-y-3">
                <div className="flex items-center text-gray-700">
                  <Zap className="w-4 h-4 text-blue-600 mr-3" />
                  <span>실시간 이슈 동기화</span>
                </div>
                <div className="flex items-center text-gray-700">
                  <Check className="w-4 h-4 text-blue-600 mr-3" />
                  <span>자동 티켓 생성</span>
                </div>
                <div className="flex items-center text-gray-700">
                  <Check className="w-4 h-4 text-blue-600 mr-3" />
                  <span>업무 자동 할당</span>
                </div>
              </div>
            </motion.div>

            {/* 다음 단계 카드 */}
            <motion.div
              className="bg-gradient-to-br from-blue-50 to-indigo-50 backdrop-blur-xl rounded-3xl p-8 border border-blue-200/50 shadow-xl"
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 1.4, duration: 0.8 }}
              whileHover={{ 
                scale: 1.02,
                boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.15)" 
              }}
            >
              <div className="flex items-center mb-6">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mr-4">
                  <Sparkles className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-gray-800">다음 단계</h3>
              </div>
              
              <div className="space-y-4 text-gray-700">
                <div className="flex items-start">
                  <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center mr-3 mt-0.5 text-xs font-bold text-white">1</div>
                  <span>Slack 채널에서 회의 시작</span>
                </div>
                <div className="flex items-start">
                  <div className="w-6 h-6 bg-indigo-500 rounded-full flex items-center justify-center mr-3 mt-0.5 text-xs font-bold text-white">2</div>
                  <span>AI가 자동으로 회의 분석</span>
                </div>
                <div className="flex items-start">
                  <div className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center mr-3 mt-0.5 text-xs font-bold text-white">3</div>
                  <span>Jira에 이슈 자동 생성</span>
                </div>
              </div>
            </motion.div>
          </div>

          {/* 액션 버튼들 */}
          <motion.div
            className="flex flex-col sm:flex-row gap-4 justify-center"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.6, duration: 0.8 }}
          >
            {/* 메인 버튼 */}
            <motion.button
              onClick={() => window.close()}
              onMouseEnter={() => setIsButtonHovered(true)}
              onMouseLeave={() => setIsButtonHovered(false)}
              className="group relative px-8 py-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white font-bold rounded-2xl shadow-2xl shadow-blue-500/30 transition-all duration-300 overflow-hidden"
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
            >
              {/* 버튼 효과 */}
              <div className={`absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 transform -skew-x-12 transition-transform duration-700 ease-out ${isButtonHovered ? 'translate-x-full' : '-translate-x-full'}`}></div>
              
              <span className="relative flex items-center space-x-3">
                <span className="text-lg">Slack으로 돌아가기</span>
                <ArrowRight className="w-5 h-5" />
              </span>
            </motion.button>

            {/* 서브 버튼 */}
            <motion.button
              className="group px-8 py-4 bg-white border-2 border-gray-300 hover:border-blue-500 text-gray-700 hover:text-blue-600 font-semibold rounded-2xl transition-all duration-300"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="flex items-center space-x-3">
                <span>Jira에서 확인</span>
                <ExternalLink className="w-4 h-4" />
              </span>
            </motion.button>
          </motion.div>

          {/* 하단 안내 */}
          <motion.div
            className="text-center mt-12"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 2, duration: 1 }}
          >
            <motion.p
              className="text-gray-500 text-sm"
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 3, repeat: Infinity }}
            >
              🎉 이제 AI 기반 프로젝트 관리를 경험해보세요!
            </motion.p>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default IntegrationSuccess;