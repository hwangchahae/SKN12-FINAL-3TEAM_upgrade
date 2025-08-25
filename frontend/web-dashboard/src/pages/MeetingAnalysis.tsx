import { useState, useEffect } from 'react';
import { Search, Edit3, Download, Plus, ChevronRight, FileText, Calendar, Clock, Users, CheckCircle, Star, Eye, MessageSquare, Target, X } from 'lucide-react';
import { motion } from 'framer-motion';
import { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, UnderlineType } from 'docx';
import { saveAs } from 'file-saver';
import { toast } from 'sonner';
import { useQuery } from '@tanstack/react-query';
import { projectAPI, taskAPI, Task } from '../services/api';

const MeetingAnalysis = () => {
  const [searchFilters, setSearchFilters] = useState({
    meetingName: '',
    date: '',
    participants: ''
  });

  // API 데이터 가져오기
  const { data: projects = [], isLoading: projectsLoading } = useQuery(
    ['projects'],
    projectAPI.getProjects
  );

  const { data: tasks = [], isLoading: tasksLoading } = useQuery(
    ['tasks'], 
    () => taskAPI.getTasks()
  );


  // 기본 체크리스트 데이터
  const defaultChecklist = [
    { id: 1, text: '모델 결과 정리', completed: false },
    { id: 2, text: '피드백 정리', completed: false },
    { id: 3, text: '회의실 예약', completed: false }
  ];

  const [checklist, setChecklist] = useState<Array<{id: number, text: string, completed: boolean}>>(() => {
    const savedChecklist = localStorage.getItem('meetingAnalysis-checklist');
    return savedChecklist ? JSON.parse(savedChecklist) : defaultChecklist;
  });

  const [showEditModal, setShowEditModal] = useState(false);
  const [showNewTaskModal, setShowNewTaskModal] = useState(false);
  const [showEditNextMeetingModal, setShowEditNextMeetingModal] = useState(false);
  const [showEditChecklistModal, setShowEditChecklistModal] = useState(false);
  const [showMeetingDetailModal, setShowMeetingDetailModal] = useState(false);
  const [selectedMeetingId, setSelectedMeetingId] = useState(1);

  // 실제 데이터 기반 회의 요약
  const currentProject = projects.length > 0 ? projects[0] : null;
  const editedSummary = currentProject ? {
    purpose: currentProject.overview || '프로젝트 진행 상황 점검',
    mainContent: currentProject.content?.summary || '프로젝트 진행사항 및 이슈 논의',
    decisions: currentProject.content?.actionItems?.map((item: any) => 
      `${item.title} (담당: ${item.assignee || '미지정'})`
    ).join(', ') || '결정사항이 기록되지 않았습니다.'
  } : {
    purpose: '프로젝트가 없습니다.',
    mainContent: '먼저 프로젝트를 생성해주세요.',
    decisions: '결정사항 없음'
  };

  // 실제 태스크를 파생업무로 변환
  const derivedTasks = (tasks as Task[]).map((task: Task) => ({
    id: task.id,
    name: task.title,
    assignee: task.assignee?.name || '미지정',
    dueDate: task.dueDate ? new Date(task.dueDate).toISOString().split('T')[0] : '',
    status: task.status === 'TODO' ? '예정' : 
            task.status === 'IN_PROGRESS' ? '진행 중' : '완료',
    statusColor: task.status === 'TODO' ? 'bg-orange-500' : 
                 task.status === 'IN_PROGRESS' ? 'bg-blue-500' : 'bg-green-500',
    priority: task.priority === 'HIGH' ? '상' : 
              task.priority === 'MEDIUM' ? '중' : '하'
  }));

  // 다음 회의 정보
  const nextMeetingInfo = {
    title: currentProject ? `▶ 다음 회의: ${currentProject.title} 진행상황 검토` : '▶ 다음 회의: 미정',
    host: '프로젝트 매니저',
    preparations: currentProject ? 
      `${currentProject.title} 관련 진행상황 보고서, 이슈 및 해결방안` : 
      '프로젝트를 먼저 생성해주세요.'
  };

  // localStorage 저장
  useEffect(() => {
    localStorage.setItem('meetingAnalysis-checklist', JSON.stringify(checklist));
  }, [checklist]);

  // 프로젝트 데이터를 회의 데이터로 변환
  const meetings = projects.map((project, index) => ({
    id: index + 1,
    name: `${project.title} 프로젝트 회의`,
    date: new Date(project.createdAt).toISOString().split('T')[0],
    participants: ['프로젝트 매니저', '개발팀', '기획팀'],
    status: '완료',
    record: {
      purpose: project.overview,
      mainContent: project.content?.summary || '프로젝트 진행사항 및 이슈 논의',
      decisions: project.content?.actionItems?.map((item: any) => 
        `• ${item.title} (담당: ${item.assignee || '미지정'}, 마감: ${item.dueDate || '미정'})`
      ).join('\n') || '결정사항이 기록되지 않았습니다.',
      basicInfo: {
        meetingName: `${project.title} 프로젝트 회의`,
        date: new Date(project.createdAt).toLocaleDateString('ko-KR'),
        participants: '프로젝트 매니저, 개발팀, 기획팀',
        host: '프로젝트 매니저'
      },
      detailedContent: {
        uiStatus: project.content?.uiRequirements || 'UI 요구사항이 정의되지 않았습니다.',
        apiStatus: project.content?.apiRequirements || 'API 요구사항이 정의되지 않았습니다.',
        performance: project.content?.performanceRequirements || '성능 요구사항이 정의되지 않았습니다.',
        technicalIssues: project.content?.technicalChallenges || '기술적 이슈가 기록되지 않았습니다.'
      }
    }
  }));

  // 필터링된 회의 목록
  const filteredMeetings = meetings.filter(meeting => {
    const matchesName = !searchFilters.meetingName || 
      meeting.name.toLowerCase().includes(searchFilters.meetingName.toLowerCase());
    const matchesDate = !searchFilters.date || meeting.date === searchFilters.date;
    const matchesParticipants = !searchFilters.participants || 
      meeting.participants.some(p => p.toLowerCase().includes(searchFilters.participants.toLowerCase()));
    
    return matchesName && matchesDate && matchesParticipants;
  });


  // 회의 기록 다운로드
  const downloadMeetingRecord = async (meeting: any) => {
    try {
      const doc = new Document({
        sections: [{
          properties: {},
          children: [
            new Paragraph({
              text: meeting.record.basicInfo.meetingName,
              heading: HeadingLevel.HEADING_1,
              alignment: AlignmentType.CENTER
            }),
            new Paragraph({
              children: [
                new TextRun({
                  text: "회의 정보",
                  bold: true,
                  underline: { type: UnderlineType.SINGLE }
                })
              ]
            }),
            new Paragraph({
              text: `일시: ${meeting.record.basicInfo.date}`
            }),
            new Paragraph({
              text: `주최자: ${meeting.record.basicInfo.host}`
            }),
            new Paragraph({
              text: `참석자: ${meeting.record.basicInfo.participants}`
            }),
            new Paragraph({ text: "" }),
            new Paragraph({
              children: [
                new TextRun({
                  text: "회의 목적",
                  bold: true,
                  underline: { type: UnderlineType.SINGLE }
                })
              ]
            }),
            new Paragraph({
              text: meeting.record.purpose
            }),
            new Paragraph({ text: "" }),
            new Paragraph({
              children: [
                new TextRun({
                  text: "주요 내용",
                  bold: true,
                  underline: { type: UnderlineType.SINGLE }
                })
              ]
            }),
            new Paragraph({
              text: meeting.record.mainContent
            }),
            new Paragraph({ text: "" }),
            new Paragraph({
              children: [
                new TextRun({
                  text: "결정사항 및 액션 아이템",
                  bold: true,
                  underline: { type: UnderlineType.SINGLE }
                })
              ]
            }),
            new Paragraph({
              text: meeting.record.decisions
            })
          ]
        }]
      });

      const blob = await Packer.toBlob(doc);
      saveAs(blob, `${meeting.name}_회의록.docx`);
      toast.success('회의록이 다운로드되었습니다! 📄');
    } catch (error) {
      console.error('Download failed:', error);
      toast.error('다운로드 중 오류가 발생했습니다.');
    }
  };

  // 로딩 상태 처리
  if (projectsLoading || tasksLoading) {
    return (
      <div className="flex-1 p-6 bg-white dark:bg-gray-900 overflow-y-auto">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-1">회의 분석</h1>
          <p className="text-sm text-gray-500 dark:text-gray-300">AI가 분석한 회의 내용을 확인하세요</p>
        </div>
        <div className="flex items-center justify-center py-20">
          <div className="text-gray-500">데이터를 불러오는 중...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 p-6 bg-white dark:bg-gray-900 overflow-y-auto">
      {/* 헤더 */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-1">회의 분석</h1>
        <p className="text-sm text-gray-500 dark:text-gray-300">AI가 분석한 회의 내용을 확인하세요</p>
      </div>


      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 왼쪽: 회의 목록 */}
        <div className="lg:col-span-2 space-y-6">
          {/* 검색 및 필터 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-4">
            <div className="flex items-center gap-4 mb-4">
              <Search className="w-5 h-5 text-gray-400" />
              <h3 className="font-semibold text-gray-900 dark:text-white">회의 검색</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <input
                type="text"
                placeholder="회의명 검색"
                value={searchFilters.meetingName}
                onChange={(e) => setSearchFilters({...searchFilters, meetingName: e.target.value})}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
              <input
                type="date"
                value={searchFilters.date}
                onChange={(e) => setSearchFilters({...searchFilters, date: e.target.value})}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
              <input
                type="text"
                placeholder="참석자 검색"
                value={searchFilters.participants}
                onChange={(e) => setSearchFilters({...searchFilters, participants: e.target.value})}
                className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
            </div>
          </div>

          {/* 회의 목록 */}
          <div className="space-y-4">
            {filteredMeetings.length === 0 ? (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-8 text-center">
                <MessageSquare className="w-16 h-16 mx-auto mb-4 opacity-50 text-gray-400" />
                <h3 className="text-lg font-medium mb-2 text-gray-900 dark:text-white">회의 데이터가 없습니다</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">프로젝트를 생성하거나 음성 파일을 업로드해주세요.</p>
              </div>
            ) : (
              filteredMeetings.map((meeting) => (
                <motion.div
                  key={meeting.id}
                  className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border-l-4 p-6 cursor-pointer transition-all ${
                    selectedMeetingId === meeting.id 
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                      : 'border-gray-300 dark:border-gray-600 hover:border-blue-300 dark:hover:border-blue-500'
                  }`}
                  onClick={() => setSelectedMeetingId(meeting.id)}
                  whileHover={{ scale: 1.01 }}
                  whileTap={{ scale: 0.99 }}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-3">
                        <h3 className="font-semibold text-lg text-gray-900 dark:text-white">{meeting.name}</h3>
                        <span className={`px-3 py-1 text-xs font-medium rounded-full ${
                          meeting.status === '완료' 
                            ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' 
                            : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                        }`}>
                          {meeting.status}
                        </span>
                      </div>
                      
                      <div className="flex items-center gap-6 text-sm text-gray-600 dark:text-gray-400 mb-3">
                        <div className="flex items-center gap-2">
                          <Calendar className="w-4 h-4" />
                          <span>{meeting.date}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Clock className="w-4 h-4" />
                          <span>14:00 - 15:30</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Users className="w-4 h-4" />
                          <span>{meeting.participants.length}명 참석</span>
                        </div>
                      </div>
                      
                      <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-2">
                        {meeting.record.purpose}
                      </p>
                    </div>
                    
                    <div className="flex items-center gap-2 ml-4">
                      <motion.button 
                        onClick={(e) => {
                          e.stopPropagation();
                          setShowMeetingDetailModal(true);
                        }}
                        className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                        title="상세보기"
                      >
                        <Eye className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                      </motion.button>
                      <motion.button 
                        onClick={(e) => {
                          e.stopPropagation();
                          downloadMeetingRecord(meeting);
                        }}
                        className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                        title="다운로드"
                      >
                        <Download className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                      </motion.button>
                      <ChevronRight className={`w-5 h-5 text-gray-400 transition-transform ${
                        selectedMeetingId === meeting.id ? 'rotate-90' : ''
                      }`} />
                    </div>
                  </div>
                </motion.div>
              ))
            )}
          </div>
        </div>

        {/* 오른쪽: 회의 상세 */}
        <div className="space-y-6">
          {/* 회의 요약 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                  <FileText className="w-5 h-5 text-blue-600" />
                  회의 요약
                </h3>
                <button
                  onClick={() => setShowEditModal(true)}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                  title="편집"
                >
                  <Edit3 className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                </button>
              </div>
            </div>
            
            <div className="p-4 space-y-4">
              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">회의 목적</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                  {editedSummary.purpose}
                </p>
              </div>
              
              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">주요 내용</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                  {editedSummary.mainContent}
                </p>
              </div>
              
              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">결정사항 및 액션 아이템</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                  {editedSummary.decisions}
                </p>
              </div>
            </div>
          </div>

          {/* 파생업무 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                  <Target className="w-5 h-5 text-green-600" />
                  파생업무
                </h3>
                <button
                  onClick={() => setShowNewTaskModal(true)}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                  title="업무 추가"
                >
                  <Plus className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                </button>
              </div>
            </div>
            
            <div className="p-4 space-y-3">
              {derivedTasks.length === 0 ? (
                <p className="text-sm text-gray-500 text-center py-8">
                  등록된 업무가 없습니다
                </p>
              ) : (
                derivedTasks.map((task) => (
                  <motion.div 
                    key={task.id} 
                    className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                    whileHover={{ scale: 1.01 }}
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-sm text-gray-900 dark:text-white">{task.name}</span>
                        <span className={`px-2 py-1 text-xs text-white rounded-full ${task.statusColor}`}>
                          {task.status}
                        </span>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                        <span className="flex items-center gap-1">
                          <Users className="w-3 h-3" />
                          {task.assignee}
                        </span>
                        {task.dueDate && (
                          <span className="flex items-center gap-1">
                            <Calendar className="w-3 h-3" />
                            {task.dueDate}
                          </span>
                        )}
                        <span className="flex items-center gap-1">
                          <Star className="w-3 h-3" />
                          {task.priority}
                        </span>
                      </div>
                    </div>
                  </motion.div>
                ))
              )}
            </div>
          </div>

          {/* 체크리스트 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-purple-600" />
                  체크리스트
                </h3>
                <button
                  onClick={() => setShowEditChecklistModal(true)}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                  title="체크리스트 편집"
                >
                  <Edit3 className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                </button>
              </div>
            </div>
            
            <div className="p-4 space-y-2">
              {checklist.map((item) => (
                <motion.div 
                  key={item.id} 
                  className="flex items-center gap-3 p-2 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-lg transition-colors"
                  whileHover={{ scale: 1.01 }}
                >
                  <input
                    type="checkbox"
                    checked={item.completed}
                    onChange={(e) => {
                      setChecklist(checklist.map((c: {id: number, text: string, completed: boolean}) => 
                        c.id === item.id ? { ...c, completed: e.target.checked } : c
                      ));
                    }}
                    className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
                  />
                  <span className={`text-sm flex-1 ${
                    item.completed 
                      ? 'text-gray-400 dark:text-gray-500 line-through' 
                      : 'text-gray-700 dark:text-gray-300'
                  }`}>
                    {item.text}
                  </span>
                </motion.div>
              ))}
            </div>
          </div>

          {/* 다음 회의 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                  <Clock className="w-5 h-5 text-orange-600" />
                  다음 회의
                </h3>
                <button
                  onClick={() => setShowEditNextMeetingModal(true)}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                  title="다음 회의 편집"
                >
                  <Edit3 className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                </button>
              </div>
            </div>
            
            <div className="p-4 space-y-3">
              <h4 className="font-medium text-gray-900 dark:text-white">{nextMeetingInfo.title}</h4>
              <div className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
                <p className="flex items-center gap-2">
                  <span className="font-medium">주최자:</span> 
                  <span>{nextMeetingInfo.host}</span>
                </p>
                <p className="flex items-start gap-2">
                  <span className="font-medium">준비사항:</span> 
                  <span>{nextMeetingInfo.preparations}</span>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 회의 상세보기 모달 */}
      {showMeetingDetailModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <motion.div 
            className="bg-white dark:bg-gray-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
          >
            <div className="p-6 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">회의 상세보기</h2>
                <button
                  onClick={() => setShowMeetingDetailModal(false)}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                </button>
              </div>
            </div>
            
            <div className="p-6">
              {meetings.find(m => m.id === selectedMeetingId) && (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="font-medium text-gray-900 dark:text-white mb-2">기본 정보</h3>
                      <div className="space-y-2 text-sm">
                        <p><span className="font-medium">회의명:</span> {meetings.find(m => m.id === selectedMeetingId)?.record.basicInfo.meetingName}</p>
                        <p><span className="font-medium">일시:</span> {meetings.find(m => m.id === selectedMeetingId)?.record.basicInfo.date}</p>
                        <p><span className="font-medium">주최자:</span> {meetings.find(m => m.id === selectedMeetingId)?.record.basicInfo.host}</p>
                        <p><span className="font-medium">참석자:</span> {meetings.find(m => m.id === selectedMeetingId)?.record.basicInfo.participants}</p>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="font-medium text-gray-900 dark:text-white mb-2">상세 내용</h3>
                      <div className="space-y-2 text-sm">
                        <p><span className="font-medium">UI 현황:</span> {meetings.find(m => m.id === selectedMeetingId)?.record.detailedContent.uiStatus}</p>
                        <p><span className="font-medium">API 상황:</span> {meetings.find(m => m.id === selectedMeetingId)?.record.detailedContent.apiStatus}</p>
                        <p><span className="font-medium">성능:</span> {meetings.find(m => m.id === selectedMeetingId)?.record.detailedContent.performance}</p>
                        <p><span className="font-medium">기술적 이슈:</span> {meetings.find(m => m.id === selectedMeetingId)?.record.detailedContent.technicalIssues}</p>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="font-medium text-gray-900 dark:text-white mb-2">회의 내용</h3>
                    <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                      <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-line">
                        {meetings.find(m => m.id === selectedMeetingId)?.record.mainContent}
                      </p>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="font-medium text-gray-900 dark:text-white mb-2">결정사항 및 액션 아이템</h3>
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                      <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-line">
                        {meetings.find(m => m.id === selectedMeetingId)?.record.decisions}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
};

export default MeetingAnalysis;