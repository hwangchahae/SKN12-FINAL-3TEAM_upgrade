import { DashboardData } from '../types';

export const mockDashboardData: DashboardData = {
  meetingStats: {
    totalMeetings: 12,
    avgOnlineTime: 20,
    accuracy: 95,
  },
  tasks: [
    // 완료된 작업
    {
      id: '1',
      title: 'WBS 문서 정리',
      date: '2024-07-04',
      category: 'completed',
    },
    {
      id: '2',
      title: '고구려정 지시사항(초안)',
      date: '2024-07-04',
      category: 'completed',
    },
    {
      id: '3',
      title: '고구려정 업무시',
      date: '2024-07-11',
      category: 'completed',
    },
    // 진행 중인 작업
    {
      id: '4',
      title: '프로젝트 기획서',
      date: '2024-07-04',
      category: 'inProgress',
    },
    {
      id: '5',
      title: '주간 데이터',
      date: '2024-07-11',
      category: 'inProgress',
    },
    {
      id: '6',
      title: '데이터베이스 설계서',
      date: '2024-07-11',
      category: 'inProgress',
    },
    {
      id: '7',
      title: '데이터 주차 프로그램',
      date: '2024-07-11',
      category: 'inProgress',
    },
    // 예약된 작업
    {
      id: '8',
      title: '인증서를 데이터 전문가 교육',
      date: '2024-07-18',
      category: 'scheduled',
    },
    {
      id: '9',
      title: '인증서를 학습 교육',
      date: '2024-07-18',
      category: 'scheduled',
    },
    {
      id: '10',
      title: '학습한 인증서는 전문',
      date: '2024-07-18',
      category: 'scheduled',
    },
    {
      id: '11',
      title: '주말 데이터 및 설치지 요청',
      date: '2024-07-18',
      category: 'scheduled',
    },
    {
      id: '12',
      title: '시스템 아키텍처',
      date: '2024-07-18',
      category: 'scheduled',
    },
  ],
  notifications: [
    {
      id: '1',
      message: '마감 임박한 업무',
      type: 'warning',
    },
    {
      id: '2',
      message: '미확인 중요한 메시지',
      type: 'info',
    },
    {
      id: '3',
      message: '시스템 공지',
      type: 'info',
    },
  ],
  recentSummaries: [
    {
      id: '1',
      title: 'AI 모델 리뷰',
      date: '2024-07-11',
      taskCount: 2,
    },
    {
      id: '2',
      title: 'UI/UX 개선 회의',
      date: '2024-07-10',
      taskCount: 1,
    },
  ],
  user: {
    id: '1',
    name: '사용자',
    role: '개발자',
  },
}; 