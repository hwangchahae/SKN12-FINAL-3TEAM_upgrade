import { Link, useLocation } from 'react-router-dom';
import { Home, MessageSquare, FileText, Settings, ChevronRight, Link as LinkIcon } from 'lucide-react';
import ttalkkakLogo from '../assets/logo.png';
import { useQuery } from '@tanstack/react-query';
import { integrationAPI, taskAPI } from '../services/api';

interface SidebarProps {
  setActiveMenu: (menu: string) => void;
}

interface MenuItemType {
  id: string;
  icon: any;
  label: string;
  path: string;
  description: string;
  badge?: number;
  status?: 'connected' | 'disconnected' | 'loading';
}

const Sidebar = ({ setActiveMenu }: SidebarProps) => {
  const location = useLocation();

  // 실시간 데이터 가져오기
  const { data: integrationStatus } = useQuery({
    queryKey: ['integrationStatus'],
    queryFn: integrationAPI.getStatus,
    refetchInterval: 30000 // 30초마다 갱신
  });

  const { data: tasks = [] } = useQuery({
    queryKey: ['tasks'],
    queryFn: taskAPI.getTasks,
    refetchInterval: 30000 // 30초마다 갱신
  });

  // 진행 중인 작업 개수 계산
  const inProgressTasksCount = tasks.filter(task => task.status === 'IN_PROGRESS').length;
  
  const menuItems: MenuItemType[] = [
    { 
      id: 'home', 
      icon: Home, 
      label: '대시보드', 
      path: '/',
      description: '전체 현황 보기'
    },
    { 
      id: 'meeting', 
      icon: MessageSquare, 
      label: '회의 분석', 
      path: '/meeting',
      description: '회의록 및 인사이트'
    },
    { 
      id: 'task', 
      icon: FileText, 
      label: '업무 관리', 
      path: '/task',
      description: '프로젝트 및 태스크',
      badge: inProgressTasksCount > 0 ? inProgressTasksCount : undefined
    },
    // 연동 메뉴 추가
    {
      id: 'integration',
      icon: LinkIcon,
      label: '연동',
      path: '/integration',
      description: '외부 서비스 연동',
      status: integrationStatus ? 
        (integrationStatus.slack || integrationStatus.notion || integrationStatus.jira ? 'connected' : 'disconnected') 
        : 'loading'
    },
    { 
      id: 'settings', 
      icon: Settings, 
      label: '설정', 
      path: '/settings',
      description: '시스템 설정'
    },
  ];

  const getActiveMenu = () => {
    const currentPath = location.pathname;
    
    // 정확한 경로 매칭
    if (currentPath === '/') return 'home';
    if (currentPath === '/meeting') return 'meeting';
    if (currentPath === '/task') return 'task';
    if (currentPath === '/integration') return 'integration';
    if (currentPath === '/settings') return 'settings';
    
    // 기본값
    return 'home';
  };

  return (
    <div className="w-72 bg-gradient-to-br from-brand-50 to-brand-100 dark:from-gray-800 dark:to-gray-900 border-r border-brand-200 dark:border-gray-700 min-h-screen relative overflow-hidden transition-colors">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute -top-4 -left-4 w-24 h-24 bg-brand-300 rounded-full blur-xl"></div>
        <div className="absolute top-32 -right-8 w-32 h-32 bg-brand-400 rounded-full blur-2xl"></div>
        <div className="absolute bottom-24 -left-8 w-28 h-28 bg-brand-500 rounded-full blur-xl"></div>
      </div>

      {/* Header */}
      <div className="relative z-10 p-6 border-b border-brand-200/50 dark:border-gray-700/50">
        <div className="flex items-center justify-start">
          <img 
            src={ttalkkakLogo} 
            alt="TtalKkak Logo" 
            className="w-28 h-22"
          />
        </div>
      </div>
      
      {/* Navigation */}
      <nav className="relative z-10 p-4 space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = getActiveMenu() === item.id;
          
          return (
            <Link
              key={item.id}
              to={item.path}
              onClick={() => setActiveMenu(item.id)}
              className={`group relative w-full flex items-center p-4 rounded-2xl transition-all duration-200 ${
                isActive 
                  ? 'bg-white dark:bg-gray-700 shadow-medium text-brand-700 dark:text-blue-400 scale-[1.02]' 
                  : 'text-neutral-700 dark:text-gray-300 hover:bg-white/50 dark:hover:bg-gray-700/50 hover:shadow-soft hover:scale-[1.01]'
              }`}
            >
              {/* Active Indicator */}
              {isActive && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-gradient-to-b from-brand-400 to-brand-600 rounded-r-full"></div>
              )}
              
              {/* Icon Container */}
              <div className={`flex items-center justify-center w-10 h-10 rounded-xl transition-colors duration-200 ${
                isActive 
                  ? 'bg-brand-100 dark:bg-blue-900/30 text-brand-600 dark:text-blue-400' 
                  : 'bg-neutral-100 dark:bg-gray-600 text-neutral-500 dark:text-gray-300 group-hover:bg-brand-50 dark:group-hover:bg-blue-900/20 group-hover:text-brand-500 dark:group-hover:text-blue-400'
              }`}>
                <Icon className="w-5 h-5" />
              </div>
              
              {/* Content */}
              <div className="ml-4 flex-1">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className={`font-semibold text-sm ${
                      isActive ? 'text-brand-700 dark:text-blue-400' : 'text-neutral-700 dark:text-gray-300'
                    }`}>
                      {item.label}
                    </span>
                    {/* Badge for task count */}
                    {item.badge && (
                      <span className="bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                        {item.badge}
                      </span>
                    )}
                    {/* Status indicator for integrations */}
                    {item.status && (
                      <div className={`w-2 h-2 rounded-full ${
                        item.status === 'connected' ? 'bg-green-500' : 
                        item.status === 'disconnected' ? 'bg-gray-400' : 'bg-yellow-500'
                      }`}></div>
                    )}
                  </div>
                  <ChevronRight className={`w-4 h-4 transition-all duration-200 ${
                    isActive 
                      ? 'text-brand-500 dark:text-blue-400 translate-x-0 opacity-100' 
                      : 'text-neutral-400 dark:text-gray-500 -translate-x-1 opacity-0 group-hover:translate-x-0 group-hover:opacity-100'
                  }`} />
                </div>
                <p className={`text-xs mt-0.5 ${
                  isActive ? 'text-brand-600 dark:text-blue-300' : 'text-neutral-500 dark:text-gray-400'
                }`}>
                  {item.description}
                </p>
              </div>
            </Link>
          );
        })}
      </nav>

      {/* Bottom Section */}
      <div className="absolute bottom-0 left-0 right-0 p-4">
        {/* Version Info */}
        <div className="mt-3 text-center">
          <p className="text-xs text-neutral-400 font-medium">TtalKkak v1.0.0</p>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;