import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Eye, 
  EyeOff, 
  Mail, 
  Lock, 
  ArrowRight, 
  Brain,
  CheckCircle,
  AlertCircle,
  Play
} from 'lucide-react';
import ttalkkakLogo from '../assets/logo.png';

const Login = () => {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [errors, setErrors] = useState<{[key: string]: string}>({});

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // 에러 클리어
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateForm = () => {
    const newErrors: {[key: string]: string} = {};

    if (!formData.email) {
      newErrors.email = '이메일을 입력해주세요';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = '올바른 이메일 형식을 입력해주세요';
    }

    if (!formData.password) {
      newErrors.password = '비밀번호를 입력해주세요';
    } else if (formData.password.length < 6) {
      newErrors.password = '비밀번호는 6자 이상이어야 합니다';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    setIsLoading(true);
    
    try {
      // TODO: 실제 로그인 API 호출
      await new Promise(resolve => setTimeout(resolve, 2000)); // 시뮬레이션
      
      // 로그인 성공 시 대시보드로 이동
      navigate('/dashboard');
    } catch (error) {
      setErrors({ general: '로그인에 실패했습니다. 다시 시도해주세요.' });
    } finally {
      setIsLoading(false);
    }
  };

  const features = [
    {
      icon: <CheckCircle className="w-6 h-6" />,
      text: "AI 기반 회의 분석"
    },
    {
      icon: <CheckCircle className="w-6 h-6" />,
      text: "자동 업무 생성 및 배정"
    },
    {
      icon: <CheckCircle className="w-6 h-6" />,
      text: "실시간 협업 도구"
    },
    {
      icon: <CheckCircle className="w-6 h-6" />,
      text: "JIRA, Notion 자동 연동"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-50">
      {/* 헤더 */}
      <motion.header 
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-neutral-200"
      >
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <motion.div 
            className="flex items-center space-x-3"
            whileHover={{ scale: 1.05 }}
          >
          </motion.div>
          
          <div className="flex items-center space-x-4">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate('/')}
              className="px-6 py-2 text-neutral-700 font-medium hover:text-blue-600 transition-colors bg-[#e7e972] rounded-md"
            >
              홈으로
            </motion.button>
          </div>
        </div>
      </motion.header>

      <div className="pt-32 flex items-center justify-center p-4">
        <div className="w-full max-w-6xl grid lg:grid-cols-[3fr_2fr] gap-12 items-center">
          
          {/* 왼쪽: 로그인 폼 */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="bg-white rounded-3xl shadow-2xl p-8 lg:p-12"
          >
            {/* 헤더 */}
            <div className="text-center mb-8">
              <motion.div 
                className="flex items-center justify-center mb-6"
                whileHover={{ scale: 1.05 }}
              >
                <img src={ttalkkakLogo} alt="TtalKkak Logo" className="w-20 h-14" />
              </motion.div>
              
                          <h1 className="text-3xl font-bold text-neutral-900 mb-2">
              Slack으로 로그인
            </h1>
            <p className="text-neutral-600">
              Slack 계정으로 간편하게 로그인하고 TtalKkak을 시작하세요
            </p>
            </div>

            {/* Slack 로그인 섹션 */}
            <div className="space-y-6">
              {/* Slack 로그인 설명 */}
              <div className="text-center mb-8">
                <div className="w-16 h-16 bg-gradient-to-r from-indigo-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-white" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M6 15a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 2a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm6-8a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 2a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm6 8a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 2a4 4 0 1 0 0-8 4 4 0 0 0 0 8z"/>
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-neutral-900 mb-2">
                  Slack 워크스페이스로 로그인
                </h3>
                <p className="text-neutral-600 text-sm">
                  기존 Slack 계정으로 간편하게 로그인하고<br />
                  팀과 함께 TtalKkak을 사용하세요
                </p>
              </div>

              {/* Slack 로그인 버튼 */}
              <motion.button
                type="button"
                onClick={() => window.open('https://slack.com/app_redirect?app=YOUR_APP_ID', '_blank')}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98, y: 0 }}
                className="w-full py-4 px-6 bg-gradient-to-r from-indigo-500 to-blue-600 text-white rounded-xl font-bold text-lg hover:shadow-xl hover:shadow-indigo-500/25 transition-all duration-300 flex items-center justify-center space-x-3 relative overflow-hidden group"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/10 to-white/0 transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-700"></div>
                <svg className="w-6 h-6 relative" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M6 15a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 2a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm6-8a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 2a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm6 8a2 2 0 1 1 0-4 2 2 0 0 1 0 4zm0 2a4 4 0 1 0 0-8 4 4 0 0 0 0 8z"/>
                </svg>
                <span className="relative">Slack으로 로그인</span>
                <ArrowRight className="w-5 h-5 relative" />
              </motion.button>

              {/* 추가 정보 */}
              <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
                <div className="flex items-start space-x-3">
                  <div className="w-5 h-5 bg-blue-500 rounded-full flex items-center justify-center mt-0.5">
                    <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="font-semibold text-blue-900 mb-1">Slack 로그인의 장점</h4>
                    <ul className="text-sm text-blue-800 space-y-1">
                      <li>• 별도 계정 생성 없이 바로 시작</li>
                      <li>• 팀원들과 자동으로 연동</li>
                      <li>• 기존 Slack 채널에서 바로 사용</li>
                      <li>• 보안 인증 자동 처리</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* 구분선 */}
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-neutral-300" />
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-4 bg-white text-neutral-500">또는</span>
                </div>
              </div>

              {/* 데모 체험 버튼 */}
              <motion.button
                type="button"
                onClick={() => navigate('/dashboard')}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full py-3 px-6 border-2 border-neutral-300 rounded-xl font-medium text-neutral-700 hover:bg-neutral-50 hover:border-neutral-400 transition-all flex items-center justify-center space-x-2"
              >
                <Play className="w-5 h-5" />
                <span>데모 체험하기</span>
              </motion.button>

              {/* 도움말 */}
              <div className="text-center text-sm text-neutral-500">
                <p>Slack 워크스페이스가 없으신가요?</p>
                <a href="https://slack.com/create" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700 font-medium">
                  무료로 워크스페이스 만들기
                </a>
              </div>
            </div>
          </motion.div>

          {/* 오른쪽: 기능 소개 */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="hidden lg:block"
          >
            <div className="space-y-8">
              {/* 메인 메시지 */}
              <div>
                <h2 className="text-4xl font-bold text-neutral-900 mb-4">
                  AI와 함께하는
                  <br />
                  <span className="bg-gradient-to-r from-indigo-600 via-blue-600 to-cyan-600 bg-clip-text text-transparent">
                    업무 혁신
                  </span>
                </h2>
                <p className="text-xl text-neutral-600 leading-relaxed">
                  회의 음성을 AI가 실시간으로 분석하여 업무를 자동 생성하고, 
                  최적의 담당자에게 자동 배정하는 혁신적인 플랫폼입니다.
                </p>
              </div>

              {/* 기능 목록 */}
              <div className="space-y-4">
                {features.map((feature, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: 30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 0.4 + index * 0.1 }}
                    className="flex items-center space-x-3"
                  >
                    <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-blue-600 rounded-lg flex items-center justify-center text-white">
                      {feature.icon}
                    </div>
                    <span className="text-lg text-neutral-700">{feature.text}</span>
                  </motion.div>
                ))}
              </div>

              {/* 통계 */}
              <div className="grid grid-cols-2 gap-6 pt-8">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.8 }}
                  className="text-center"
                >
                  <div className="text-3xl font-bold text-blue-600 mb-1">95%+</div>
                  <div className="text-sm text-neutral-600">음성 인식 정확도</div>
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.9 }}
                  className="text-center"
                >
                  <div className="text-3xl font-bold text-purple-600 mb-1">2.5배</div>
                  <div className="text-sm text-neutral-600">업무 효율성 향상</div>
                </motion.div>
              </div>

              {/* CTA */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 1.0 }}
                className="pt-8"
              >
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => navigate('/')}
                  className="px-8 py-4 bg-gradient-to-r from-indigo-500 to-blue-600 text-white rounded-2xl font-bold text-lg shadow-xl hover:shadow-2xl transition-all flex items-center space-x-2"
                >
                  <span>더 알아보기</span>
                  <ArrowRight className="w-3 h-3" />
                </motion.button>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Login;



