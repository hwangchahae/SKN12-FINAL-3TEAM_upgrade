import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { 
  Volume2,
  FileText,
  LayoutDashboard,
  Plus,
  RefreshCw,
  Edit3,
  ExternalLink
} from 'lucide-react';
import {
  DndContext,
  DragEndEvent,
  DragOverEvent,
  DragStartEvent,
  DragOverlay,
  PointerSensor,
  useSensor,
  useSensors,
  closestCenter,
  useDroppable,
} from '@dnd-kit/core';
import {
  SortableContext,
  verticalListSortingStrategy,
  useSortable,
  arrayMove,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { dashboardAPI, projectAPI, taskAPI, userAPI, subscribeToRealTimeUpdates } from '../services/api';
import { toast } from 'sonner';


interface KanbanItem {
  id: string;
  content: string;
  date: string;
  assignee?: string;
  priority?: string;
  originalTask?: any;
}

interface KanbanColumn {
  name: string;
  items: KanbanItem[];
}

interface KanbanColumns {
  todo: KanbanColumn;
  progress: KanbanColumn;
  done: KanbanColumn;
  [key: string]: KanbanColumn; // 이것이 핵심! 동적 키 접근 허용
}

type ColumnId = 'todo' | 'progress' | 'done';

// 날짜가 지났는지 확인하는 함수
const isOverdue = (dateString: string, columnId: string) => {
  // 완료된 업무는 제외
  if (columnId === 'done') return false;
  
  try {
    const today = new Date();
    const taskDate = new Date();
    
    // MM.DD 형식을 YYYY-MM-DD로 변환
    const [month, day] = dateString.split('.');
    taskDate.setMonth(parseInt(month) - 1); // 월은 0부터 시작
    taskDate.setDate(parseInt(day));
    taskDate.setHours(23, 59, 59, 999); // 하루 끝으로 설정
    
    return today > taskDate;
  } catch (error) {
    return false; // 날짜 형식이 잘못된 경우 false 반환
  }
};



// @dnd-kit을 사용한 드래그 앤 드롭 컴포넌트
const TaskCard = ({ task, columnId, onTaskSelect, isOverTarget }: any) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ 
    id: task.id,
    transition: {
      duration: 150,
      easing: 'cubic-bezier(0.25, 1, 0.5, 1)',
    },
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  const getColumnBgColor = (colId: string) => {
    const colors: Record<string, string> = {
      'todo': 'bg-accent-amber/10',
      'progress': 'bg-accent-blue/10', 
      'done': 'bg-accent-green/10'
    };
    return colors[colId] || 'bg-white';
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`group rounded-lg ${getColumnBgColor(columnId)} ${
        isDragging ? 'opacity-90 rotate-1 scale-105 shadow-xl z-50 border-2 border-blue-400' : 
        isOverTarget ? 'ring-2 ring-blue-400 ring-offset-2 bg-blue-50/50 scale-105 border-blue-300' : 'hover:shadow-md shadow-sm'
      } transition-all duration-200 border border-gray-200 cursor-grab active:cursor-grabbing`}
      {...attributes}
      {...listeners}
    >
      {/* 전체 카드가 드래그 영역 */}
      <div className="px-6 py-6 w-full">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <div className="w-1 h-4 bg-current opacity-30 rounded-full group-hover:opacity-60 transition-opacity" />
              <span className="font-medium text-sm text-gray-900 line-clamp-2">{task.content}</span>
            </div>
            <div className="flex items-center justify-between mt-3">
              <span className={`text-xs ${isOverdue(task.date, columnId) ? 'text-red-600 font-semibold' : 'text-gray-500'}`}>
                📅 {task.date}
              </span>
              {task.assignee && (
                <span className="text-xs px-2 py-1 bg-white/50 rounded-full">
                  👤 {task.assignee}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* 상세보기 버튼 - 드래그 영역 밖 */}
      <div className="flex items-center justify-start px-3 pb-3 pt-0 border-t border-white/30">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onTaskSelect({...task, status: columnId});
          }}
          onMouseDown={(e) => e.stopPropagation()}
          onPointerDown={(e) => e.stopPropagation()}
          className="text-xs text-blue-600 hover:text-blue-800 font-medium px-3 py-1 rounded bg-white/50 hover:bg-white/70 transition-colors cursor-pointer"
        >
          상세보기
        </button>
      </div>
    </div>
  );
};

const DroppableColumn = ({ colId, items, onTaskSelect, isOver: isOverProp, overItemId }: { colId: string; items: any[]; onTaskSelect: any; isOver?: boolean; overItemId?: string | null }) => {
  const { setNodeRef, isOver: isDroppableOver } = useDroppable({
    id: colId,
  });

  const isOver = isOverProp || isDroppableOver;

  return (
    <div
      ref={setNodeRef}
      className={`min-h-[600px] rounded-b-xl p-6 transition-all duration-200 ${
        isOver ? 'bg-blue-50/50 ring-2 ring-offset-2 ring-blue-400 scale-105' : 'bg-neutral-50 hover:bg-neutral-100'
      }`}
    >
      <SortableContext items={items.map(item => item.id)} strategy={verticalListSortingStrategy}>
        <div className="space-y-4 min-h-[500px]">
          {items.map((item, _index) => (
            <div key={item.id} className="mb-4">
              <TaskCard 
                task={item} 
                columnId={colId}
                onTaskSelect={onTaskSelect}
                isOverTarget={overItemId === item.id}
              />
              {/* 드래그 중 삽입 위치 표시 */}
              {isOver && overItemId === item.id && (
                <div className="h-2 bg-blue-400 rounded-full my-2 opacity-60 animate-pulse" />
              )}
            </div>
          ))}
          {/* 빈 컬럼에 드롭할 때의 표시 */}
          {isOver && items.length === 0 && (
            <div className="h-32 border-2 border-dashed border-blue-400 rounded-lg flex items-center justify-center bg-blue-50/30">
              <span className="text-blue-600 font-medium">여기에 드롭하세요</span>
            </div>
          )}
        </div>
      </SortableContext>
    </div>
  );
};



const MainContent = () => {
  const [viewMode, setViewMode] = useState<'dashboard' | 'kanban'>('dashboard');

  const [isUploading, setIsUploading] = useState(false);
  const [isDocumentUploading, setIsDocumentUploading] = useState(false);
  const [showNewTaskModal, setShowNewTaskModal] = useState(false);
  const [selectedTask, setSelectedTask] = useState<any>(null);
  const [activeTask, setActiveTask] = useState<any>(null);
  const [overColumnId, setOverColumnId] = useState<string | null>(null);
  const [overItemId, setOverItemId] = useState<string | null>(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [taskToDelete, setTaskToDelete] = useState<any>(null);
  const [isEditingTask, setIsEditingTask] = useState(false);
  const [editedTaskData, setEditedTaskData] = useState<any>({});

  // React Query로 데이터 패칭
  const { data: stats, isLoading: statsLoading, refetch: refetchStats, error: statsError } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: dashboardAPI.getStats,
    refetchInterval: 30000, // 30초마다 자동 갱신
  });

  const { data: tasks = [], isLoading: tasksLoading, refetch: refetchTasks, error: tasksError } = useQuery({
    queryKey: ['tasks'],
    queryFn: () => taskAPI.getTasks(),
  });

  const { data: users = [], isLoading: usersLoading, error: usersError } = useQuery({
    queryKey: ['users'],
    queryFn: userAPI.getUsers,
  });


  // 실시간 업데이트 구독
  useEffect(() => {
    const unsubscribe = subscribeToRealTimeUpdates({
      onTaskUpdate: () => {
        refetchTasks();
        refetchStats();
      },
      onProjectCreate: () => {
        refetchStats();
      },
    });

    return unsubscribe;
  }, [refetchTasks, refetchStats]);

  // ⬇️ 여기에 새로 추가할 디버깅 로그 ⬇️
  useEffect(() => {
    console.log('=== API 데이터 상태 체크 ===');
    console.log('📊 Stats:', { data: stats, loading: statsLoading, error: statsError });
    console.log('📋 Tasks:', { data: tasks, count: tasks?.length, loading: tasksLoading, error: tasksError });
    console.log('👥 Users:', { data: users, count: users?.length, loading: usersLoading, error: usersError });
    console.log('========================');
  }, [stats, tasks, users, statsLoading, tasksLoading, usersLoading, statsError, tasksError, usersError]);


  // 칸반보드용 초기 데이터
  // Tailwind 동적 클래스 문제 해결: 색상 클래스 분리
  const columnColors: Record<ColumnId, { bg: string; text: string }> = {
    done: {
      bg: 'bg-accent-green/10',
      text: 'text-accent-green',
    },
    progress: {
      bg: 'bg-accent-blue/10',
      text: 'text-accent-blue',
    },
    todo: {
      bg: 'bg-accent-amber/10',
      text: 'text-accent-amber',
    },
  };
  const [columns, setColumns] = useState<KanbanColumns>({
    todo: { name: '해야할 일', items: [] },
    progress: { name: '진행중', items: [] },
    done: { name: '완료', items: [] }
  });

  // API 데이터를 칸반보드에 매핑하는 useEffect 추가
  useEffect(() => {
    if (tasks && tasks.length > 0) {
      console.log('🔄 업무 데이터를 칸반보드에 매핑 중...', tasks);
      
      const newColumns: KanbanColumns = {
        todo: { name: '해야할 일', items: [] },
        progress: { name: '진행중', items: [] },
        done: { name: '완료', items: [] }
      };

      tasks.forEach(task => {
        // 날짜 포맷팅 함수
        const formatDate = (dateString?: string) => {
          if (!dateString) return '미정';
          try {
            const date = new Date(dateString);
            const month = (date.getMonth() + 1).toString().padStart(2, '0');
            const day = date.getDate().toString().padStart(2, '0');
            return `${month}.${day}`;
          } catch (error) {
            return '미정';
          }
        };

        const kanbanItem: KanbanItem = {
          id: task.id,
          content: task.title,
          date: formatDate(task.dueDate),
          assignee: task.assignee?.name || '미지정',
          priority: task.priority?.toLowerCase() || 'medium',
          originalTask: task
        };

        // 백엔드 상태를 칸반 컬럼에 매핑
        if (task.status === 'TODO') {
          newColumns.todo.items.push(kanbanItem);
        } else if (task.status === 'IN_PROGRESS') {
          newColumns.progress.items.push(kanbanItem);
        } else if (task.status === 'DONE') {
          newColumns.done.items.push(kanbanItem);
        }
      });

      console.log('✅ 칸반보드 업데이트:', newColumns);
      setColumns(newColumns);
    } else {
      console.log('❌ 업무 데이터가 없습니다:', { tasks, tasksLoading });
    }
  }, [tasks]);


  // @dnd-kit 센서 설정 - 더 자연스러운 드래그 경험
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 5, // 드래그 시작 거리를 줄여서 더 민감하게
        delay: 100, // 짧은 지연시간으로 빠른 반응
      },
    })
  );

  // 드래그 시작
  const handleDragStart = (event: DragStartEvent) => {
    const { active } = event;
    // 현재 드래그되는 아이템 찾기
    for (const [columnId, column] of Object.entries(columns)) {
      const task = column.items.find(item => item.id === active.id);
      if (task) {
        setActiveTask(task);
        break;
      }
    }
  };

  // 드래그 오버 - 삽입 위치 시각적 피드백
  const handleDragOver = (event: DragOverEvent) => {
    const { over } = event;
    if (over) {
      // 컬럼 위에 있는 경우
      const column = columns[over.id as string];
      if (column) {
        setOverColumnId(over.id as string);
        setOverItemId(null);
      } else {
        // 특정 아이템 위에 있는 경우
        setOverItemId(over.id as string);
        // 해당 아이템이 속한 컬럼 찾기
        for (const [columnId, col] of Object.entries(columns)) {
          const item = col.items.find(item => item.id === over.id);
          if (item) {
            setOverColumnId(columnId);
            break;
          }
        }
      }
    } else {
      // 드래그가 컬럼 밖으로 나간 경우
      setOverColumnId(null);
      setOverItemId(null);
    }
  };

  // 드래그 종료 - 자리 양보 방식으로 삽입
  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    setActiveTask(null);
    setOverColumnId(null);
    setOverItemId(null);
    if (!over) return;

    const activeId = active.id as string;
    const overId = over.id as string;

    // 소스 컬럼과 아이템 찾기
    let sourceColumnId = '';
    let sourceItem = null;
    let sourceIndex = -1;
    for (const [columnId, column] of Object.entries(columns)) {
      const idx = column.items.findIndex(item => item.id === activeId);
      if (idx !== -1) {
        sourceColumnId = columnId;
        sourceItem = column.items[idx];
        sourceIndex = idx;
        break;
      }
    }
    if (!sourceItem) return;

    // 타겟 컬럼과 인덱스 찾기
    let targetColumnId = '';
    let targetIndex = -1;
    
    // 컬럼 자체에 드롭한 경우
    if (columns[overId]) {
      targetColumnId = overId;
      targetIndex = 0; // 맨 위에 삽입
    } else {
      // 특정 아이템 위에 드롭한 경우
      for (const [columnId, column] of Object.entries(columns)) {
        const idx = column.items.findIndex(item => item.id === overId);
        if (idx !== -1) {
          targetColumnId = columnId;
          targetIndex = idx; // 해당 아이템 위치에 삽입 (기존 아이템들이 아래로 밀려남)
          break;
        }
      }
    }

    if (!targetColumnId) return;

    // 같은 컬럼 내 이동
    if (sourceColumnId === targetColumnId) {
      if (sourceIndex === targetIndex) return;
      
      const items = [...columns[sourceColumnId].items];
      const newItems = arrayMove(items, sourceIndex, targetIndex);
      setColumns({
        ...columns,
        [sourceColumnId]: { ...columns[sourceColumnId], items: newItems }
      });
    } else {
      // 다른 컬럼으로 이동 - 자리 양보 방식으로 삽입
      const sourceItems = [...columns[sourceColumnId].items];
      sourceItems.splice(sourceIndex, 1); // 소스에서 제거
      
      const targetItems = [...columns[targetColumnId].items];
      
      // 자리 양보 방식: 삽입 위치의 기존 아이템들이 한 칸씩 밀려나서 자리를 비워줌
      const newTargetItems = [
        ...targetItems.slice(0, targetIndex), // 삽입 위치 이전 아이템들 (그대로 유지)
        sourceItem, // 삽입할 아이템 (빈 자리에 들어감)
        ...targetItems.slice(targetIndex) // 삽입 위치 이후 아이템들 (한 칸씩 밀려남)
      ];
      
      setColumns({
        ...columns,
        [sourceColumnId]: { ...columns[sourceColumnId], items: sourceItems },
        [targetColumnId]: { ...columns[targetColumnId], items: newTargetItems }
      });

      // 백엔드 상태 업데이트
      if (sourceItem?.originalTask) {
        const newStatus = 
          targetColumnId === 'todo' ? 'TODO' :
          targetColumnId === 'progress' ? 'IN_PROGRESS' : 'DONE';
        
        // 비동기로 백엔드 업데이트
        taskAPI.updateTaskStatus(sourceItem.originalTask.id, newStatus)
          .then(() => {
            toast.success('업무 상태가 업데이트되었습니다! ✅');
            refetchTasks(); // 데이터 새로고침
          })
          .catch((error) => {
            console.error('Failed to update task status:', error);
            toast.error('상태 업데이트에 실패했습니다. 😞');
          });
      }
    }
  };



  // 업무 추가 함수
  const addNewTask = (taskData: any) => {
    const newTask = {
      id: Date.now().toString(),
      content: taskData.title,
      date: taskData.dueDate,
      assignee: taskData.assignee,
      priority: taskData.priority
    };

    // 선택된 상태에 따라 해당 컬럼에 추가
    const targetColumn = taskData.status || 'todo'; // 기본값은 'todo'

    setColumns((prev: any) => ({
      ...prev,
      [targetColumn]: {
        ...prev[targetColumn],
        items: [newTask, ...prev[targetColumn].items] // 새 업무를 맨 위에 추가하여 기존 업무들이 아래로 밀려남
      }
    }));
  };

  // 업무 삭제 함수
  const deleteTask = (taskId: string) => {
    setColumns(prev => {
      const newColumns = { ...prev };
      Object.keys(newColumns).forEach(colId => {
        newColumns[colId] = {
          ...newColumns[colId],
          items: newColumns[colId].items.filter(item => item.id !== taskId)
        };
      });
      return newColumns;
    });
  };

  // 업무 수정 함수
  const updateTask = (taskId: string, updatedData: any) => {
    setColumns(prev => {
      const newColumns = { ...prev };
      Object.keys(newColumns).forEach(colId => {
        newColumns[colId] = {
          ...newColumns[colId],
          items: newColumns[colId].items.map(item => 
            item.id === taskId ? { ...item, ...updatedData } : item
          )
        };
      });
      return newColumns;
    });
  };

  // MM.DD 형식을 YYYY-MM-DD로 변환
  const formatDateForInput = (dateString: string) => {
    try {
      const [month, day] = dateString.split('.');
      const currentYear = new Date().getFullYear();
      return `${currentYear}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
    } catch (error) {
      return '';
    }
  };

  // YYYY-MM-DD를 MM.DD 형식으로 변환
  const formatDateFromInput = (dateString: string) => {
    try {
      const date = new Date(dateString);
      const month = (date.getMonth() + 1).toString();
      const day = date.getDate().toString();
      return `${month.padStart(2, '0')}.${day.padStart(2, '0')}`;
    } catch (error) {
      return dateString;
    }
  };

  // 편집 모드 시작
  const startEditingTask = (task: any) => {
    setIsEditingTask(true);
    setEditedTaskData({
      content: task.content,
      date: task.date,
      dateForInput: formatDateForInput(task.date), // 달력 피커용 날짜
      assignee: task.assignee || '',
      priority: task.priority || '중간'
    });
  };

  // 편집 저장
  const saveTaskEdit = () => {
    if (selectedTask) {
      // 달력에서 선택한 날짜를 MM.DD 형식으로 변환
      const finalTaskData = {
        ...editedTaskData,
        date: editedTaskData.dateForInput ? formatDateFromInput(editedTaskData.dateForInput) : editedTaskData.date
      };
      
      updateTask(selectedTask.id, finalTaskData);
      setSelectedTask({ ...selectedTask, ...finalTaskData });
      setIsEditingTask(false);
      toast.success('업무가 수정되었습니다! ✏️');
    }
  };

  // 편집 취소
  const cancelTaskEdit = () => {
    setIsEditingTask(false);
    setEditedTaskData({});
  };

  // 음성 파일 업로드 핸들러
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      const result = await projectAPI.createFromAudio(file);
      if (result.success) {
        refetchStats();
        refetchTasks();
        toast.success(`음성 파일 "${file.name}"이 성공적으로 처리되었습니다! 🎤`, {
          description: '새로운 업무들이 생성되었습니다.'
        });
      }
    } catch (error) {
      console.error('Upload failed:', error);
      toast.error('음성 파일 처리 중 오류가 발생했습니다. 😞');
    } finally {
      setIsUploading(false);
    }
  };

  // 문서 파일 업로드 핸들러
  const handleDocumentUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsDocumentUploading(true);
    try {
      // 문서 분석 API 호출 (예시)
      // const result = await projectAPI.analyzeDocument(file);
      
      // 임시로 성공 처리
      setTimeout(() => {
        setIsDocumentUploading(false);
        toast.success(`문서 "${file.name}"이 성공적으로 분석되었습니다! 📄`, {
          description: `크기: ${(file.size / 1024).toFixed(1)}KB • 새로운 업무들이 추출되었습니다.`
        });
        
        // 예시로 새 업무 추가
        const newTask = {
          id: Date.now().toString(),
          content: `문서 분석 결과: ${file.name.split('.')[0]}`,
          date: new Date().toLocaleDateString('ko-KR').replace(/\./g, '').replace(/ /g, '.'),
          assignee: '자동 할당',
          priority: '중간'
        };

        setColumns((prev: any) => ({
          ...prev,
          todo: {
            ...prev.todo,
            items: [...prev.todo.items, newTask]
          }
        }));

        refetchStats();
        refetchTasks();
      }, 2000);
    } catch (error) {
      console.error('Document analysis failed:', error);
      toast.error('문서 분석 중 오류가 발생했습니다. 😞');
      setIsDocumentUploading(false);
    }
  };

  if (statsLoading || tasksLoading || usersLoading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-neutral-50">
        <div className="flex flex-col items-center space-y-4">
          <RefreshCw className="w-8 h-8 text-brand-500 animate-spin" />
          <p className="text-neutral-600 font-medium">데이터를 불러오는 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-neutral-50 dark:bg-gray-900 min-h-screen">
      {/* 헤더 */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 border-b border-neutral-200 dark:border-gray-700 px-8 py-6"
      >
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-2xl font-bold text-neutral-900 dark:text-white">오늘의 프로젝트 현황</h2>
            <p className="text-neutral-600 dark:text-gray-300 mt-1">AI 기반 실시간 업무 관리 대시보드</p>
          </div>
          
          {/* 뷰 모드 전환 및 랜딩페이지 버튼 */}
          <div className="flex items-center space-x-4">
            <div className="flex bg-neutral-100 rounded-2xl p-1">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setViewMode('dashboard')}
                className={`flex items-center px-4 py-2 rounded-xl transition-all duration-200 ${
                  viewMode === 'dashboard' 
                    ? 'bg-white shadow-soft text-brand-700' 
                    : 'text-neutral-600 hover:text-neutral-800'
                }`}
              >
                <LayoutDashboard className="w-4 h-4 mr-2" />
                대시보드
              </motion.button>
            </div>
            
            {/* 랜딩페이지 버튼 */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => window.location.href = '/landing'}
              className="flex items-center px-4 py-2 bg-brand-600 text-white rounded-xl hover:bg-brand-700 transition-all duration-200 shadow-soft"
            >
              <ExternalLink className="w-4 h-4 mr-2" />
              랜딩페이지
            </motion.button>
          </div>
        </div>


      </motion.div>

      {/* 메인 콘텐츠 */}
      <div className="p-8">
        <div className="space-y-6">
          {/* 메인 영역 - 칸반보드 (전체 화면) */}
          <div className="space-y-6">
            {/* 빠른 액션 */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="grid grid-cols-2 md:grid-cols-4 gap-4"
            >
              <h3 className="text-lg font-bold text-neutral-900 mb-4">빠른 액션</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <motion.label
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="relative group cursor-pointer"
                >
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={handleFileUpload}
                    className="hidden"
                    disabled={isUploading}
                  />
                  <div className="flex flex-col items-center p-6 bg-gradient-to-br from-brand-50 to-brand-100 rounded-2xl border-2 border-dashed border-brand-300 transition-all duration-200 group-hover:border-brand-400 group-hover:shadow-medium">
                    {isUploading ? (
                      <RefreshCw className="w-8 h-8 text-brand-600 animate-spin mb-3" />
                    ) : (
                      <Volume2 className="w-8 h-8 text-brand-600 mb-3" />
                    )}
                    <span className="font-semibold text-brand-700">
                      {isUploading ? '처리 중...' : '음성 파일 업로드'}
                    </span>
                    <span className="text-sm text-brand-600 mt-1">회의록 자동 생성</span>
                  </div>
                </motion.label>

                <motion.label
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="relative group cursor-pointer"
                >
                  <input
                    type="file"
                    accept=".pdf,.doc,.docx,.txt,.ppt,.pptx,.xls,.xlsx"
                    onChange={handleDocumentUpload}
                    className="hidden"
                    disabled={isDocumentUploading}
                  />
                  <div className="flex flex-col items-center p-6 bg-gradient-to-br from-neutral-50 to-neutral-100 rounded-2xl border-2 border-dashed border-neutral-300 transition-all duration-200 group-hover:border-neutral-400 group-hover:shadow-medium">
                    {isDocumentUploading ? (
                      <RefreshCw className="w-8 h-8 text-neutral-600 animate-spin mb-3" />
                    ) : (
                      <FileText className="w-8 h-8 text-neutral-600 mb-3" />
                    )}
                    <span className="font-semibold text-neutral-700">
                      {isDocumentUploading ? '분석 중...' : '문서 분석'}
                    </span>
                    <span className="text-sm text-neutral-500 mt-1">PDF, DOC, TXT 업로드</span>
                  </div>
                </motion.label>
              </div>
            </motion.div>

            {/* 업무 현황 - 여기에 KanbanBoard 삽입 */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="bg-white rounded-2xl p-6 shadow-soft border border-neutral-200"
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-bold text-neutral-900">업무 현황</h3>
                <div className="flex items-center space-x-3">
                  {/* 새 업무 추가 버튼 */}
                  <button
                    onClick={() => setShowNewTaskModal(true)}
                    className="flex items-center gap-2 px-4 py-2 bg-brand-500 text-white rounded-xl hover:bg-brand-600 transition-colors text-sm font-medium"
                  >
                    <Plus className="w-4 h-4" />
                    새 업무 추가
                  </button>
                </div>
              </div>
              
              {/* 새로운 요소를 위한 공간 - 여기에 추가하면 칸반보드가 자동으로 아래로 밀려남 */}
              <div className="space-y-4 mb-6">
                {/* 새로운 요소들을 여기에 추가하세요 */}
                {/* 예시: 새로운 요소가 추가되면 칸반보드가 자동으로 아래로 밀려납니다 */}
              </div>
              
              {/* 칸반보드 스타일 - @dnd-kit 드래그 앤 드롭 - 아래쪽에 배치 */}
              <DndContext
                sensors={sensors}
                collisionDetection={closestCenter}
                onDragStart={handleDragStart}
                onDragOver={handleDragOver}
                onDragEnd={handleDragEnd}
              >
                <div className="grid grid-cols-3 gap-4 min-h-[700px]">
                  {['todo', 'progress', 'done'].map((colId) => {
                    const col = columns[colId];
                    return (
                    <div key={colId} className="w-full">
                      {/* 컬럼 헤더 */}
                      <div className={`rounded-t-xl px-4 py-3 ${columnColors[colId as ColumnId].bg} border-b-2 ${
                        colId === 'todo' ? 'border-accent-amber' : 
                        colId === 'progress' ? 'border-accent-blue' : 'border-accent-green'
                      }`}>
                        <div className="flex items-center justify-between">
                          <h4 className={`font-bold text-lg ${columnColors[colId as ColumnId].text}`}>
                            {col.name}
                          </h4>
                          <span className={`text-sm font-medium px-2 py-1 rounded-full ${
                            colId === 'todo' ? 'bg-accent-amber/20 text-accent-amber' : 
                            colId === 'progress' ? 'bg-accent-blue/20 text-accent-blue' : 'bg-accent-green/20 text-accent-green'
                          }`}>
                            {col.items.length}
                          </span>
                        </div>
                      </div>
                      
                      {/* 컬럼 내용 - @dnd-kit 드래그 앤 드롭 */}
                      <DroppableColumn 
                        colId={colId} 
                        items={col.items} 
                        onTaskSelect={setSelectedTask}
                        isOver={overColumnId === colId}
                        overItemId={overItemId}
                      />
                    </div>
                  );
                })}
                </div>

                {/* 드래그 오버레이 */}
                <DragOverlay dropAnimation={{
                  duration: 200,
                  easing: 'cubic-bezier(0.18, 0.67, 0.6, 1.22)',
                }}>
                  {activeTask ? <TaskCard task={activeTask} columnId="todo" onTaskSelect={() => {}} /> : null}
                </DragOverlay>
              </DndContext>
            </motion.div>
          </div>


        </div>
      </div>

      {/* 새 업무 추가 모달 */}
      {showNewTaskModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-96 max-w-[90vw]">
            <h3 className="text-lg font-semibold mb-4">새 업무 추가</h3>
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
              addNewTask(taskData);
              setShowNewTaskModal(false);
              toast.success('새 업무가 추가되었습니다! ✨');
            }}>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">업무명 *</label>
                  <input
                    name="title"
                    type="text"
                    placeholder="업무명을 입력하세요"
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
                    className="w-full p-2 border border-gray-300 rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">마감일</label>
                  <input
                    name="dueDate"
                    type="date"
                    className="w-full p-2 border border-gray-300 rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">업무 상태 *</label>
                  <select name="status" className="w-full p-2 border border-gray-300 rounded-lg" required>
                    <option value="todo">해야할 일</option>
                    <option value="progress">진행중</option>
                    <option value="done">완료</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">업무 설명</label>
                  <textarea
                    name="description"
                    placeholder="업무에 대한 설명을 입력하세요"
                    className="w-full p-2 border border-gray-300 rounded-lg h-20 resize-none"
                  />
                </div>
              </div>
              
              <div className="flex gap-3 mt-6">
                <button
                  type="button"
                  onClick={() => setShowNewTaskModal(false)}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  취소
                </button>
                <button
                  type="submit"
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  추가
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* 업무 상세 모달 */}
      {selectedTask && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-96 max-w-[90vw]">
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-lg font-semibold">업무 상세</h3>
              <div className="flex items-center gap-2">
                {!isEditingTask && (
                  <button
                    onClick={() => startEditingTask(selectedTask)}
                    className="p-2 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                    title="편집"
                  >
                    <Edit3 className="w-4 h-4" />
                  </button>
                )}
                <button
                  onClick={() => {
                    setSelectedTask(null);
                    setIsEditingTask(false);
                    setEditedTaskData({});
                  }}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">업무명</label>
                {isEditingTask ? (
                  <input
                    type="text"
                    value={editedTaskData.content || ''}
                    onChange={(e) => setEditedTaskData((prev: any) => ({ ...prev, content: e.target.value }))}
                    className="w-full p-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="업무명을 입력하세요"
                  />
                ) : (
                  <div className="p-2 bg-gray-50 rounded-lg text-sm">{selectedTask.content}</div>
                )}
              </div>
              
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium mb-1">마감일</label>
                  {isEditingTask ? (
                    <input
                      type="date"
                      value={editedTaskData.dateForInput || ''}
                      onChange={(e) => setEditedTaskData((prev: any) => ({ ...prev, content: e.target.value }))}
                      className="w-full p-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  ) : (
                    <div className="p-2 bg-gray-50 rounded-lg text-sm">📅 {selectedTask.date}</div>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">현재 상태</label>
                  <div className="p-2 bg-gray-50 rounded-lg text-sm">
                    {selectedTask.status === 'todo' && '📋 해야할 일'}
                    {selectedTask.status === 'progress' && '⚡ 진행중'}
                    {selectedTask.status === 'done' && '✅ 완료'}
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">담당자</label>
                {isEditingTask ? (
                  <input
                    type="text"
                    value={editedTaskData.assignee || ''}
                    onChange={(e) => setEditedTaskData((prev: any) => ({ ...prev, content: e.target.value }))}
                    className="w-full p-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="담당자를 입력하세요"
                  />
                ) : (
                  <div className="p-2 bg-gray-50 rounded-lg text-sm">👤 {selectedTask.assignee || '미지정'}</div>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">상태 변경</label>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => {
                      // 상태 변경 로직
                      const newColumns = { ...columns };
                      // 현재 위치에서 제거
                      Object.keys(newColumns).forEach(colId => {
                        newColumns[colId].items = newColumns[colId].items.filter(item => item.id !== selectedTask.id);
                      });
                      // 새 위치에 추가
                      newColumns.todo.items.push(selectedTask);
                      setColumns(newColumns);
                      setSelectedTask(null);
                    }}
                    className="p-2 text-xs bg-amber-100 text-amber-700 rounded-lg hover:bg-amber-200 transition-colors"
                  >
                    📋 해야할 일
                  </button>
                  <button
                    onClick={() => {
                      const newColumns = { ...columns };
                      Object.keys(newColumns).forEach(colId => {
                        newColumns[colId].items = newColumns[colId].items.filter(item => item.id !== selectedTask.id);
                      });
                      newColumns.progress.items.push(selectedTask);
                      setColumns(newColumns);
                      setSelectedTask(null);
                    }}
                    className="p-2 text-xs bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
                  >
                    ⚡ 진행중
                  </button>
                  <button
                    onClick={() => {
                      const newColumns = { ...columns };
                      Object.keys(newColumns).forEach(colId => {
                        newColumns[colId].items = newColumns[colId].items.filter(item => item.id !== selectedTask.id);
                      });
                      newColumns.done.items.push(selectedTask);
                      setColumns(newColumns);
                      setSelectedTask(null);
                    }}
                    className="p-2 text-xs bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors"
                  >
                    ✅ 완료
                  </button>
                </div>
              </div>
            </div>
            
            <div className="flex gap-3 mt-6">
              {isEditingTask ? (
                <>
                  <button
                    onClick={cancelTaskEdit}
                    className="flex-1 px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 transition-colors"
                  >
                    취소
                  </button>
                  <button
                    onClick={saveTaskEdit}
                    className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    저장
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => {
                      setTaskToDelete(selectedTask);
                      setShowDeleteModal(true);
                    }}
                    className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                  >
                    삭제
                  </button>
                  <button
                    onClick={() => setSelectedTask(null)}
                    className="flex-1 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                  >
                    닫기
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 삭제 확인 모달 */}
      {showDeleteModal && taskToDelete && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[60]">
          <div className="bg-white rounded-lg p-6 w-96 max-w-[90vw] shadow-xl">
            <div className="text-center">
              {/* 경고 아이콘 */}
              <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-red-100 mb-4">
                <svg className="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
              </div>

              {/* 제목 */}
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                업무 삭제
              </h3>

              {/* 메시지 */}
              <p className="text-sm text-gray-500 mb-2">
                다음 업무를 정말로 삭제하시겠습니까?
              </p>
              <p className="text-sm font-medium text-gray-900 bg-gray-50 p-2 rounded mb-6">
                "{taskToDelete.content}"
              </p>

              {/* 버튼들 */}
              <div className="flex gap-3">
                <button
                  onClick={() => {
                    setShowDeleteModal(false);
                    setTaskToDelete(null);
                  }}
                  className="flex-1 px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 transition-colors font-medium"
                >
                  취소
                </button>
                <button
                  onClick={() => {
                    deleteTask(taskToDelete.id);
                    setShowDeleteModal(false);
                    setTaskToDelete(null);
                    setSelectedTask(null);
                  }}
                  className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium"
                >
                  삭제
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MainContent;