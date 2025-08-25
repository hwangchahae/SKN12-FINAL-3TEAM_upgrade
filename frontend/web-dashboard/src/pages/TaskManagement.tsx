import { useState, useEffect } from 'react';
import { Search, Plus, Calendar, CheckCircle, Clock, Star, Edit3, Trash2, FileText } from 'lucide-react';
import { motion } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'; // ✅ 추가
import { taskAPI, Task, projectAPI } from '../services/api'; // ✅ 추가
import { toast } from 'sonner';

const TaskManagement = () => {
  // 에러 처리를 위한 try-catch
  try {
  const [filters, setFilters] = useState({
    assignee: '전체',
    status: '전체',
    search: ''
  });
  
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [showNewTaskModal, setShowNewTaskModal] = useState(false);
  const [editingTaskForModal, setEditingTaskForModal] = useState<Task | null>(null);
  const [isEditingTask, setIsEditingTask] = useState(false);
  const [editedTaskData, setEditedTaskData] = useState<any>(null);
  const [showCalendarModal, setShowCalendarModal] = useState(false);
  const [calendarDate, setCalendarDate] = useState(new Date());
  const [showDayDetailModal, setShowDayDetailModal] = useState(false);
  const [selectedDayTasks, setSelectedDayTasks] = useState<any[]>([]);
  const [selectedDayInfo, setSelectedDayInfo] = useState({ date: '', day: 0 });
  const [searchInput, setSearchInput] = useState('');

  // 상태별 카드 클릭 필터링 함수
  const handleStatusFilter = (status: string) => {
    setFilters({...filters, status: status});
  };

  // 검색 실행 함수
  const handleSearch = () => {
    if (searchInput.trim()) {
      setFilters({...filters, search: searchInput.trim()});
    } else {
      setFilters({...filters, search: ''});
    }
  };

  // ✅ 새로운 버전 (교체)
  const addNewTask = (taskData: any) => {
    // 프로젝트가 없으면 생성 불가
    if (!projects || projects.length === 0) {
      toast.error('먼저 프로젝트를 생성해주세요.');
      return;
    }
    
    const newTask = {
      title: taskData.title,
      description: taskData.description || '',
      status: taskData.status === 'todo' ? 'TODO' : 
              taskData.status === 'progress' ? 'IN_PROGRESS' : 'DONE',
      priority: taskData.priority === '상' ? 'HIGH' : 
                taskData.priority === '하' ? 'LOW' : 'MEDIUM',
      dueDate: taskData.dueDate || undefined,
      assigneeId: taskData.assignee || undefined,
      projectId: projects[0].id // 첫 번째 프로젝트 사용
    };
    
    createTaskMutation.mutate(newTask);
  };

  const updateTask = (taskData: any) => {
    if (editingTaskForModal) {
      const updates = {
        title: taskData.title,
        description: taskData.description || '',
        status: taskData.status === 'todo' ? 'TODO' : 
                taskData.status === 'progress' ? 'IN_PROGRESS' : 'DONE',
        dueDate: taskData.dueDate || undefined,
        assigneeId: taskData.assignee || undefined
      };
      
      updateTaskMutation.mutate({ 
        taskId: editingTaskForModal.id, 
        updates 
      });
    }
  };
  

  const queryClient = useQueryClient();
  
  // 프로젝트 목록 가져오기
  const { data: projects = [] } = useQuery({
    queryKey: ['projects'],
    queryFn: projectAPI.getProjects
  });

  // 태스크 목록 가져오기
  const { data: tasks = [], isLoading: tasksLoading } = useQuery({
    queryKey: ['tasks'],
    queryFn: () => taskAPI.getTasks()
  });

  const createTaskMutation = useMutation({
    mutationFn: (newTask: any) => taskAPI.createTask(newTask),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      toast.success('새 업무가 추가되었습니다! ✨');
    },
    onError: (error) => {
      console.error('Task creation failed:', error);
      toast.error('업무 생성에 실패했습니다.');
    }
  });

  const updateTaskMutation = useMutation({
    mutationFn: ({ taskId, updates }: { taskId: string; updates: any }) =>
      taskAPI.updateTask(taskId, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      toast.success('업무가 수정되었습니다! ✏️');
    },
    onError: (error) => {
      console.error('Task update failed:', error);
      toast.error('업무 수정에 실패했습니다.');
    }
  });

  const deleteTaskMutation = useMutation({
    mutationFn: (taskId: string) => taskAPI.deleteTask(taskId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      toast.success('업무가 삭제되었습니다! 🗑️');
    },
    onError: (error) => {
      console.error('Task deletion failed:', error);
      toast.error('업무 삭제에 실패했습니다.');
    }
  });

  // 필터링된 업무 목록
  const filteredTasks = tasks.filter(task => {
    const assigneeName = task.assignee?.name || '미지정';
    const statusKorean = task.status === 'DONE' ? '완료' : 
                        task.status === 'IN_PROGRESS' ? '진행 중' : '예정';
    
    const matchesAssignee = filters.assignee === '전체' || assigneeName === filters.assignee;
    const matchesStatus = filters.status === '전체' || statusKorean === filters.status;
    const matchesSearch = task.title.toLowerCase().includes(filters.search.toLowerCase()) ||
                        assigneeName.toLowerCase().includes(filters.search.toLowerCase());
    
    return matchesAssignee && matchesStatus && matchesSearch;
  });



  // 상태 아이콘 반환
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'DONE': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'IN_PROGRESS': return <Clock className="w-4 h-4 text-blue-500" />;
      case 'TODO': return <Calendar className="w-4 h-4 text-orange-500" />;
      default: return null;
    }
  };



  // 상태 변경 함수
  const handleStatusChange = (taskId: string, newStatus: string) => {
    const apiStatus = newStatus === '완료' ? 'DONE' : 
                    newStatus === '진행 중' ? 'IN_PROGRESS' : 'TODO';
    
    updateTaskMutation.mutate({ 
      taskId, 
      updates: { status: apiStatus } 
    });
  };

  // 업무 편집 시작
  const startEditingTask = () => {
    if (selectedTask) {
      setEditedTaskData({
        name: selectedTask.title,
        assignee: selectedTask.assignee,
        dueDate: selectedTask.dueDate,
        description: selectedTask.description
      });
      setIsEditingTask(true);
    }
  };

  // 업무 편집 저장
  const saveTaskEdit = () => {
    if (selectedTask && editedTaskData) {
      updateTaskMutation.mutate({ 
        taskId: selectedTask.id, 
        updates: editedTaskData 
      });
      setSelectedTask(null);
      setIsEditingTask(false);
      setEditedTaskData(null);
    }
  };

  // 업무 편집 취소
  const cancelTaskEdit = () => {
    setIsEditingTask(false);
    setEditedTaskData(null);
  };

  // 이전 달로 이동
  const goToPreviousMonth = () => {
    setCalendarDate(prev => new Date(prev.getFullYear(), prev.getMonth() - 1, 1));
  };

  // 다음 달로 이동
  const goToNextMonth = () => {
    setCalendarDate(prev => new Date(prev.getFullYear(), prev.getMonth() + 1, 1));
  };

  // 날짜 클릭 시 해당 날짜의 업무 상세보기
  const handleDayClick = (day: number, tasksForDay: any[]) => {
    if (tasksForDay.length > 0) {
      const dateStr = `${calendarDate.getFullYear()}-${String(calendarDate.getMonth() + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
      setSelectedDayTasks(tasksForDay);
      setSelectedDayInfo({ 
        date: dateStr, 
        day: day 
      });
      setShowDayDetailModal(true);
    }
  };



  // 업무 삭제 함수
  const handleDeleteTask = (taskId: string) => {
    if (window.confirm('정말로 이 업무를 삭제하시겠습니까?')) {
      deleteTaskMutation.mutate(taskId);
      if (selectedTask?.id === taskId) {
        setSelectedTask(null);
      }
    }
  };

  // 디버깅용 로그
  console.log('TaskManagement - tasksLoading:', tasksLoading);
  console.log('TaskManagement - tasks:', tasks);
  console.log('TaskManagement - projects:', projects);

  // 로딩 상태 처리
  if (tasksLoading) {
    return (
      <div className="flex h-full bg-gray-100 dark:bg-gray-900">
        <div className="flex-1 p-6 overflow-auto">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 mb-6">
            <h1 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">업무 관리</h1>
            <div className="flex items-center justify-center py-20">
              <div className="text-gray-500">데이터를 불러오는 중...</div>
            </div>
          </div>
        </div>
      </div>
    );
  }





  return (
    <div className="flex h-full bg-gray-100 dark:bg-gray-900">
      {/* Main Content */}
      <div className="flex-1 p-6 overflow-auto">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 mb-6">
          <h1 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">업무 관리</h1>
          
          {/* 통계 카드 */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <motion.div 
              className="bg-gradient-to-r from-blue-500 to-blue-600 p-4 rounded-lg text-white cursor-pointer"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => handleStatusFilter('전체')}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm opacity-90">전체 업무</p>
                  <p className="text-2xl font-bold">{tasks.length}</p>
                </div>
                <FileText className="w-8 h-8 opacity-80" />
              </div>
            </motion.div>
            
            <motion.div 
              className="bg-gradient-to-r from-green-500 to-green-600 p-4 rounded-lg text-white cursor-pointer"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => handleStatusFilter('완료')}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm opacity-90">완료</p>
                  <p className="text-2xl font-bold">{tasks.filter(t => t.status === 'DONE').length}</p>
                </div>
                <CheckCircle className="w-8 h-8 opacity-80" />
              </div>
            </motion.div>
            
            <motion.div 
              className="bg-gradient-to-r from-yellow-500 to-yellow-600 p-4 rounded-lg text-white cursor-pointer"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => handleStatusFilter('진행 중')}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm opacity-90">진행 중</p>
                  <p className="text-2xl font-bold">{tasks.filter(t => t.status === 'IN_PROGRESS').length}</p>
                </div>
                <Clock className="w-8 h-8 opacity-80" />
              </div>
            </motion.div>
            
            <motion.div 
              className="bg-gradient-to-r from-orange-500 to-orange-600 p-4 rounded-lg text-white cursor-pointer"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => handleStatusFilter('예정')}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm opacity-90">예정</p>
                  <p className="text-2xl font-bold">{tasks.filter(t => t.status === 'TODO').length}</p>
                </div>
                <Calendar className="w-8 h-8 opacity-80" />
              </div>
            </motion.div>
          </div>

          {/* Filters */}
          <div className="flex gap-4 mb-6">
            <select 
              value={filters.assignee}
              onChange={(e) => setFilters({...filters, assignee: e.target.value})}
              className="px-4 py-2 border border-gray-300 rounded-lg bg-white hover:border-blue-400 transition-colors"
            >
              <option value="전체">전체 담당자</option>
              <option value="김미정">김미정</option>
              <option value="이철수">이철수</option>
              <option value="박영희">박영희</option>
              <option value="정수민">정수민</option>
            </select>
            
            <select 
              value={filters.status}
              onChange={(e) => setFilters({...filters, status: e.target.value})}
              className="px-4 py-2 border border-gray-300 rounded-lg bg-white hover:border-blue-400 transition-colors"
            >
              <option value="전체">전체 상태</option>
              <option value="완료">완료</option>
              <option value="진행 중">진행 중</option>
              <option value="예정">예정</option>
            </select>
            
            <div className="relative flex-1">
              <input
                type="text"
                placeholder="업무명 또는 담당자 검색..."
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    handleSearch();
                  }
                }}
                className="w-full p-2 border border-gray-300 rounded-lg pr-10 hover:border-blue-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all"
              />
              <Search className="absolute right-3 top-2.5 w-4 h-4 text-gray-400" />
            </div>
            
            <motion.button 
              onClick={handleSearch}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg flex items-center gap-2 transition-colors"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Search className="w-4 h-4" />
              검색
            </motion.button>
          </div>
          
          {/* 필터링 결과 표시 */}
          <div className="mb-4 text-sm text-gray-600">
            총 {filteredTasks.length}개의 업무가 있습니다.
            {filteredTasks.length !== tasks.length && (
              <span className="ml-2 text-blue-600">
                (전체 {tasks.length}개 중 필터링됨)
              </span>
            )}
          </div>
        </div>

        {/* Task Cards */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold">업무 목록</h2>
            <motion.button 
              onClick={() => setShowNewTaskModal(true)}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center gap-2 transition-colors"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Plus className="w-4 h-4" />
              새 업무 추가
            </motion.button>
          </div>
          <div className="space-y-3">
            {filteredTasks.map((task) => {
              // 마감일 지났는지 확인 (완료 제외)
              const isOverdue = task.status !== '완료' && new Date(task.dueDate) < new Date();
              
              return (
                <motion.div 
                  key={task.id} 
                  className="border border-gray-200 rounded-lg p-4 hover:shadow-md cursor-pointer transition-all"
                  onClick={() => setSelectedTask(task)}
                  whileHover={{ scale: 1.01 }}
                  layout
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1" onClick={() => setSelectedTask(task)}>
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="font-medium text-gray-900">{task.title}</h3>
                      </div>
                      
                      <div className="flex items-center gap-4 text-sm text-gray-600">
                        <div className="flex items-center gap-1">
                          <span>👤</span>
                          <span>{task.assignee?.name || '미지정'}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Calendar className="w-4 h-4" />
                          <span className={isOverdue ? 'text-red-600 font-semibold' : ''}>
                            {task.dueDate}
                            {isOverdue && <span className="ml-1">⚠️</span>}
                          </span>
                        </div>
                        <div className="flex items-center gap-1">
                          {getStatusIcon(task.status)}
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            task.status === 'TODO' ? 'bg-orange-100 text-orange-700' :
                            task.status === 'IN_PROGRESS' ? 'bg-blue-100 text-blue-700' :
                            'bg-green-100 text-green-700'
                          }`}>
                            {task.status === 'TODO' ? '예정' : 
                             task.status === 'IN_PROGRESS' ? '진행 중' : '완료'}
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditingTaskForModal(task);
                        setShowNewTaskModal(true);
                      }}
                      className="ml-3 p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                    >
                      <Edit3 className="w-4 h-4" />
                    </button>
                  </div>
                </motion.div>
              );
            })}
            
            {filteredTasks.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>조건에 맞는 업무가 없습니다.</p>
                <p className="text-sm">다른 필터를 선택하거나 새 업무를 추가해보세요.</p>
              </div>
            )}
          </div>
        </div>

        {/* 새 업무 추가/수정 모달 */}
        {showNewTaskModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 w-96 max-w-[90vw]">
              <h3 className="text-lg font-semibold mb-4">
                {editingTaskForModal ? '업무 수정' : '새 업무 추가'}
              </h3>
              <form onSubmit={(e) => {
                e.preventDefault();
                const formData = new FormData(e.target as HTMLFormElement);
                const taskData = {
                  title: formData.get('title') as string,
                  assignee: formData.get('assignee') as string,
                  dueDate: formData.get('dueDate') as string,
                  priority: formData.get('priority') as string,
                  description: formData.get('description') as string,
                  status: formData.get('status') as string
                };
                
                if (editingTaskForModal) {
                  updateTask(taskData);
                  toast.success('업무가 수정되었습니다! ✏️');
                } else {
                  addNewTask(taskData);
                  toast.success('새 업무가 추가되었습니다! ✨');
                }
                
                setShowNewTaskModal(false);
                setEditingTaskForModal(null);
              }}>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">업무명 *</label>
                    <input
                      name="title"
                      type="text"
                      placeholder="업무명을 입력하세요"
                      defaultValue={editingTaskForModal?.name || ''}
                      className="w-full p-2 border border-gray-300 rounded-lg"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">담당자</label>
                    <input
                      name="assignee"
                      type="text"
                      placeholder="담당자를 입력하세요"
                      defaultValue={editingTaskForModal?.assignee || ''}
                      className="w-full p-2 border border-gray-300 rounded-lg"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">마감일</label>
                    <input
                      name="dueDate"
                      type="date"
                      defaultValue={editingTaskForModal?.dueDate || ''}
                      className="w-full p-2 border border-gray-300 rounded-lg"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">업무 상태 *</label>
                    <select 
                      name="status" 
                      className="w-full p-2 border border-gray-300 rounded-lg" 
                      defaultValue={
                        editingTaskForModal?.status === '예정' ? 'todo' :
                        editingTaskForModal?.status === '진행 중' ? 'progress' :
                        editingTaskForModal?.status === '완료' ? 'done' : 'todo'
                      }
                      required
                    >
                      <option value="todo">예정</option>
                      <option value="progress">진행중</option>
                      <option value="done">완료</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">업무 설명</label>
                    <textarea
                      name="description"
                      placeholder="업무에 대한 설명을 입력하세요"
                      defaultValue={editingTaskForModal?.description || ''}
                      className="w-full p-2 border border-gray-300 rounded-lg h-20 resize-none"
                    />
                  </div>
                </div>
                
                <div className="flex gap-3 mt-6">
                  <button
                    type="button"
                    onClick={() => {
                      setShowNewTaskModal(false);
                      setEditingTaskForModal(null);
                    }}
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    취소
                  </button>
                  <button
                    type="submit"
                    className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    {editingTaskForModal ? '수정' : '추가'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>

      {/* Right Sidebar - Task Details */}
      <div className="w-80 bg-white shadow-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">업무 상세 정보</h3>
          {!isEditingTask && (
            <button
              onClick={startEditingTask}
              className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <Edit3 className="w-4 h-4" />
            </button>
          )}
        </div>
{selectedTask ? (
          <motion.div 
            className="space-y-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="border-b pb-4">
              {isEditingTask ? (
                <input
                  type="text"
                  value={editedTaskData?.name || ''}
                  onChange={(e) => setEditedTaskData(prev => ({ ...prev, name: e.target.value }))}
                  className="w-full font-medium text-gray-900 mb-2 p-2 border border-gray-300 rounded-lg"
                  placeholder="업무명"
                />
              ) : (
                <h4 className="font-medium text-gray-900 mb-2">{selectedTask.title}</h4>
              )}
              <div className="flex items-center gap-2 mb-3">
                {getStatusIcon(selectedTask.status)}
                <span className={`px-2 py-1 text-xs rounded-full ${
                  selectedTask.status === 'TODO' ? 'bg-orange-100 text-orange-700' :
                  selectedTask.status === 'IN_PROGRESS' ? 'bg-blue-100 text-blue-700' :
                  'bg-green-100 text-green-700'
                }`}>
                  {selectedTask.status === 'TODO' ? '예정' : 
                   selectedTask.status === 'IN_PROGRESS' ? '진행 중' : '완료'}
                </span>
              </div>
              <div className="flex items-center gap-4 text-sm text-gray-600">
                <div className="flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  {isEditingTask ? (
                    <input
                      type="date"
                      value={editedTaskData?.dueDate || ''}
                      onChange={(e) => setEditedTaskData(prev => ({ ...prev, dueDate: e.target.value }))}
                      className="p-1 border border-gray-300 rounded text-sm"
                    />
                  ) : (
                    <span>{selectedTask.dueDate}</span>
                  )}
                </div>
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">담당자</label>
              <div className="flex items-center gap-2">
                <span>👤</span>
                {isEditingTask ? (
                  <input
                    type="text"
                    value={typeof editedTaskData?.assignee === 'string' ? editedTaskData.assignee : editedTaskData?.assignee?.name || ''}
                    onChange={(e) => setEditedTaskData(prev => ({ ...prev, assignee: e.target.value }))}
                    className="flex-1 text-sm text-gray-900 p-2 border border-gray-300 rounded-lg"
                    placeholder="담당자"
                  />
                ) : (
                  <span className="text-sm text-gray-900">{selectedTask.assignee?.name || '미지정'}</span>
                )}
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">설명</label>
              {isEditingTask ? (
                <textarea
                  value={editedTaskData?.description || ''}
                  onChange={(e) => setEditedTaskData(prev => ({ ...prev, description: e.target.value }))}
                  className="w-full text-sm text-gray-600 p-3 border border-gray-300 rounded-lg"
                  rows={4}
                  placeholder="업무 설명"
                />
              ) : (
                <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg">
                  {selectedTask.description}
                </div>
              )}
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">상태 변경</label>
              <select
                value={selectedTask.status}
                onChange={(e) => handleStatusChange(selectedTask.id, e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-lg"
              >
                <option value="예정">예정</option>
                <option value="진행 중">진행 중</option>
                <option value="완료">완료</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">액션</label>
              <div className="space-y-2">
                {isEditingTask ? (
                  <>
                    <motion.button 
                      onClick={saveTaskEdit}
                      className="w-full px-4 py-2 bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100 transition-colors flex items-center justify-center gap-2"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <CheckCircle className="w-4 h-4" />
                      저장
                    </motion.button>
                    <motion.button 
                      onClick={cancelTaskEdit}
                      className="w-full px-4 py-2 bg-gray-50 text-gray-600 rounded-lg hover:bg-gray-100 transition-colors flex items-center justify-center gap-2"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      취소
                    </motion.button>
                  </>
                ) : (
                  <motion.button 
                    onClick={() => handleDeleteTask(selectedTask.id)}
                    className="w-full px-4 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors flex items-center justify-center gap-2"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Trash2 className="w-4 h-4" />
                    업무 삭제
                  </motion.button>
                )}
              </div>
            </div>
          </motion.div>
        ) : (
          <div className="text-center text-gray-500 mt-8">
            <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p className="mb-2">업무를 선택하면</p>
            <p>상세 정보가 표시됩니다.</p>
          </div>
        )}
        
        <div className="mt-8 pt-6 border-t">
          <h4 className="text-md font-semibold mb-4 flex items-center gap-2">
            <Star className="w-4 h-4 text-yellow-500" />
            빠른 액션
          </h4>
          <div className="space-y-2">
            <motion.button 
              onClick={() => setShowCalendarModal(true)}
              className="w-full px-4 py-2 bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100 transition-colors flex items-center justify-center gap-2"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Calendar className="w-4 h-4" />
              업무 캘린더
            </motion.button>

          </div>
        </div>
      </div>

      {/* 업무 캘린더 모달 */}
      {showCalendarModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <motion.div 
            className="bg-white rounded-xl p-6 w-full max-w-4xl max-h-[80vh] overflow-auto"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold flex items-center gap-2">
                <Calendar className="w-5 h-5 text-blue-600" />
                업무 캘린더
              </h3>
              <button
                onClick={() => setShowCalendarModal(false)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                ✕
              </button>
            </div>

            {/* 달력 헤더 */}
            <div className="mb-6">
              <div className="flex items-center justify-between mb-4">
                <button
                  onClick={goToPreviousMonth}
                  className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  ◀
                </button>
                <h4 className="text-lg font-semibold">
                  {calendarDate.getFullYear()}년 {calendarDate.getMonth() + 1}월
                </h4>
                <button
                  onClick={goToNextMonth}
                  className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  ▶
                </button>
              </div>
              
              {/* 요일 헤더 */}
              <div className="grid grid-cols-7 gap-1 mb-2">
                {['일', '월', '화', '수', '목', '금', '토'].map(day => (
                  <div key={day} className="p-3 text-center font-semibold text-gray-600 bg-gray-100 rounded">
                    {day}
                  </div>
                ))}
              </div>
              
              {/* 달력 그리드 */}
              <div className="grid grid-cols-7 gap-1">
                {(() => {
                  const now = new Date();
                  const firstDay = new Date(calendarDate.getFullYear(), calendarDate.getMonth(), 1);
                  const lastDay = new Date(calendarDate.getFullYear(), calendarDate.getMonth() + 1, 0);
                  const firstDayOfWeek = firstDay.getDay();
                  const daysInMonth = lastDay.getDate();
                  
                  const days = [];
                  
                  // 이전 달의 빈 칸들
                  for (let i = 0; i < firstDayOfWeek; i++) {
                    days.push(
                      <div key={`empty-${i}`} className="p-2 h-24 bg-gray-50 rounded border opacity-50">
                      </div>
                    );
                  }
                  
                  // 현재 달의 날짜들
                  for (let day = 1; day <= daysInMonth; day++) {
                    const dateStr = `${calendarDate.getFullYear()}-${String(calendarDate.getMonth() + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
                    const tasksForDay = tasks.filter(task => task.dueDate === dateStr);
                    
                    // 디버깅: 모든 업무가 있는 날에 표시
                    const debugTasks = tasks.filter(task => task.dueDate === dateStr);
                    if (debugTasks.length > 0) {
                      console.log(`${dateStr} (${day}일) 업무:`, debugTasks.map(t => t.name));
                    }
                    const isToday = (
                      day === now.getDate() && 
                      calendarDate.getMonth() === now.getMonth() && 
                      calendarDate.getFullYear() === now.getFullYear()
                    );
                    
                    days.push(
                      <div 
                        key={day} 
                        className={`p-2 h-24 bg-white rounded border transition-all hover:bg-gray-50 ${
                          isToday ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                        } ${tasksForDay.length > 0 ? 'cursor-pointer hover:border-blue-300' : ''}`}
                        onClick={() => handleDayClick(day, tasksForDay)}
                      >
                        <div className={`text-sm font-medium mb-1 ${
                          isToday ? 'text-blue-600' : 'text-gray-900'
                        }`}>
                          {day}
                        </div>
                        <div className="space-y-1 overflow-hidden">
                          {/* 디버깅: 업무가 있는 모든 날에 표시 */}
                          {tasksForDay.length > 0 && (
                            <div className="text-xs text-red-500 font-bold">
                              ●{tasksForDay.length}개
                            </div>
                          )}
                          {tasksForDay.slice(0, 2).map(task => {
                            const taskDate = new Date(task.dueDate);
                            const today = new Date();
                            today.setHours(0, 0, 0, 0);
                            const isOverdue = taskDate < today;
                            return (
                              <div
                                key={task.id}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setSelectedTask(task);
                                  setShowCalendarModal(false);
                                }}
                                className={`text-xs p-1 rounded cursor-pointer truncate transition-all hover:scale-105 ${
                                  isOverdue
                                    ? 'bg-red-100 text-red-700 border border-red-200'
                                    : task.status === '완료'
                                    ? 'bg-green-100 text-green-700 border border-green-200'
                                    : 'bg-blue-100 text-blue-700 border border-blue-200'
                                }`}
                                title={task.title}
                              >
                                {task.title}
                              </div>
                            );
                          })}
                          {tasksForDay.length > 2 && (
                            <div className="text-xs text-gray-500 text-center">
                              +{tasksForDay.length - 2}개 더
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  }
                  
                  return days;
                })()}
              </div>
            </div>

            {tasks.filter(task => task.dueDate).length === 0 && (
              <div className="text-center text-gray-500 py-8">
                <Calendar className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>마감일이 설정된 업무가 없습니다.</p>
              </div>
            )}
          </motion.div>
        </div>
      )}

      {/* 날짜별 업무 상세보기 모달 */}
      {showDayDetailModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <motion.div 
            className="bg-white rounded-xl p-6 w-full max-w-2xl max-h-[70vh] overflow-auto"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold flex items-center gap-2">
                <Calendar className="w-5 h-5 text-blue-600" />
                {calendarDate.getFullYear()}년 {calendarDate.getMonth() + 1}월 {selectedDayInfo.day}일 업무
              </h3>
              <button
                onClick={() => setShowDayDetailModal(false)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4">
              {selectedDayTasks.map(task => {
                const taskDate = new Date(task.dueDate);
                const today = new Date();
                today.setHours(0, 0, 0, 0);
                const isOverdue = taskDate < today;
                
                return (
                  <motion.div
                    key={task.id}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                      isOverdue 
                        ? 'border-red-200 bg-red-50 hover:bg-red-100' 
                        : task.status === '완료'
                        ? 'border-green-200 bg-green-50 hover:bg-green-100'
                        : 'border-blue-200 bg-blue-50 hover:bg-blue-100'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    onClick={() => {
                      setSelectedTask(task);
                      setShowDayDetailModal(false);
                      setShowCalendarModal(false);
                    }}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold text-gray-900">{task.title}</h4>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                        task.status === 'TODO' ? 'bg-orange-100 text-orange-700' :
                        task.status === 'IN_PROGRESS' ? 'bg-blue-100 text-blue-700' :
                        'bg-green-100 text-green-700'
                      }`}>
                        {task.status === 'TODO' ? '예정' : 
                         task.status === 'IN_PROGRESS' ? '진행 중' : '완료'}
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 text-sm text-gray-600 mb-3">
                      <div className="flex items-center gap-2">
                        <span>👤</span>
                        <span>{task.assignee}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Calendar className="w-4 h-4" />
                        <span className={isOverdue ? 'text-red-600 font-semibold' : ''}>
                          {task.dueDate}
                          {isOverdue && ' (마감일 지남)'}
                        </span>
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-700 bg-gray-100 p-3 rounded">
                      {task.description}
                    </p>
                  </motion.div>
                );
              })}
            </div>

            <div className="mt-6 text-center">
              <p className="text-sm text-gray-500">
                업무를 클릭하면 상세보기로 이동합니다
              </p>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
  } catch (error) {
    console.error('TaskManagement 컴포넌트 에러:', error);
    return (
      <div className="flex h-full bg-gray-100 dark:bg-gray-900">
        <div className="flex-1 p-6 overflow-auto">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 mb-6">
            <h1 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">업무 관리</h1>
            <div className="flex items-center justify-center py-20">
              <div className="text-red-500">페이지 로드 중 오류가 발생했습니다. 콘솔을 확인해주세요.</div>
            </div>
          </div>
        </div>
      </div>
    );
  }
};

export default TaskManagement; 