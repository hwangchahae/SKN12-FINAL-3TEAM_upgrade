import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  DndContext,
  DragEndEvent,
  DragOverlay,
  DragStartEvent,
  PointerSensor,
  useSensor,
  useSensors,
  closestCenter,
  DragOverEvent,
} from '@dnd-kit/core';
import {
  SortableContext,
  verticalListSortingStrategy,
  arrayMove,
  insertAtIndex,
  removeAtIndex,
  rectIntersection,
} from '@dnd-kit/sortable';
import {
  useSortable,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { 
  Clock, 
  User, 
  Calendar, 
  AlertCircle, 
  CheckCircle, 
  Target,
  Plus,
  MoreHorizontal 
} from 'lucide-react';
import { taskAPI, Task } from '../services/api';

interface KanbanColumn {
  id: string;
  title: string;
  status: Task['status'];
  color: string;
  bgColor: string;
  tasks: Task[];
}

const TaskCard: React.FC<{ 
  task: Task; 
  isDragging?: boolean; 
  isOver?: boolean;
  isInsertAfter?: boolean;
  shouldPushDown?: boolean;
  index: number;
}> = ({ task, isDragging, isOver = false, isInsertAfter = false, shouldPushDown = false, index }) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging: isSortableDragging,
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

  const formatDate = (dateString?: string) => {
    if (!dateString) return null;
    return new Date(dateString).toLocaleDateString('ko-KR', {
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <motion.div
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      data-task-id={task.id}
      initial={{ opacity: 0, y: 10 }}
      animate={{ 
        opacity: shouldPushDown ? 0.5 : 1, 
        y: shouldPushDown ? 100 : 0,
        scale: shouldPushDown ? 0.95 : 1,
        x: shouldPushDown ? 10 : 0
      }}
      exit={{ opacity: 0, y: -10 }}
      whileHover={{ y: -2, scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      transition={{ 
        duration: shouldPushDown ? 0.2 : 0.2,
        ease: shouldPushDown ? 'easeOut' : 'easeInOut'
      }}
      className={`bg-white rounded-2xl p-4 shadow-soft border border-neutral-200 cursor-grab active:cursor-grabbing transition-all duration-200 hover:shadow-medium ${
        isDragging || isSortableDragging ? 'opacity-90 rotate-1 scale-105 shadow-xl z-[100]' : ''
      } ${
        isOver ? 'ring-2 ring-blue-400 ring-offset-2 bg-blue-50 border-blue-300' : ''
      } ${
        isInsertAfter ? 'border-t-4 border-t-blue-400' : ''
      } ${
        shouldPushDown ? 'opacity-50 z-10' : ''
      }`}
    >
      {/* 태스크 헤더 */}
      <div className="flex items-start justify-between mb-3">
        <h4 className="font-semibold text-neutral-900 text-sm leading-tight flex-1 pr-2">
          {task.title}
        </h4>
        <button className="text-neutral-400 hover:text-neutral-600 transition-colors">
          <MoreHorizontal className="w-4 h-4" />
        </button>
      </div>

      {/* 태스크 설명 */}
      {task.description && (
        <p className="text-xs text-neutral-600 mb-3 line-clamp-2 leading-relaxed">
          {task.description}
        </p>
      )}

      {/* 메타데이터 */}
      <div className="space-y-2">
        {/* 담당자 */}
        {task.assignee && (
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 bg-brand-100 rounded-full flex items-center justify-center">
              <User className="w-3 h-3 text-brand-600" />
            </div>
            <span className="text-xs text-neutral-600 font-medium">
              {task.assignee.name}
            </span>
          </div>
        )}

        {/* 마감일 */}
        {task.dueDate && (
          <div className="flex items-center space-x-1">
            <Calendar className="w-3 h-3 text-neutral-400" />
            <span className="text-xs text-neutral-500">
              {formatDate(task.dueDate)}
            </span>
          </div>
        )}

        {/* 예상 시간 */}
        {task.metadata?.estimatedHours && (
          <div className="flex items-center space-x-1">
            <Clock className="w-3 h-3 text-neutral-400" />
            <span className="text-xs text-neutral-500">
              {task.metadata.estimatedHours}시간 예상
            </span>
          </div>
        )}
      </div>
    </motion.div>
  );
};

const KanbanColumn: React.FC<{ 
  column: KanbanColumn; 
  onAddTask: (columnId: string) => void;
  isOver?: boolean;
  overTaskId?: string | null;
  insertAfter?: string | null;
  dragOverIndex?: number | null;
}> = ({ 
  column, 
  onAddTask,
  isOver = false,
  overTaskId = null,
  insertAfter = null,
  dragOverIndex = null
}) => {
  const {
    setNodeRef,
  } = useSortable({ id: column.id });

  const getColumnIcon = (status: Task['status']) => {
    switch (status) {
      case 'TODO':
        return <AlertCircle className="w-5 h-5" />;
      case 'IN_PROGRESS':
        return <Clock className="w-5 h-5" />;
      case 'DONE':
        return <CheckCircle className="w-5 h-5" />;
      default:
        return <Target className="w-5 h-5" />;
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* 컬럼 헤더 */}
      <div className={`${column.bgColor} rounded-2xl p-4 mb-4 border border-opacity-20 transition-all duration-200 ${
        isOver ? 'ring-2 ring-offset-2 ring-blue-400 scale-105' : ''
      }`} style={{ borderColor: column.color }}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-xl ${column.bgColor.replace('/10', '/20')}`} style={{ color: column.color }}>
              {getColumnIcon(column.status)}
            </div>
            <div>
              <h3 className="font-bold text-neutral-900">{column.title}</h3>
              <p className="text-sm text-neutral-600">{column.tasks.length}개 업무</p>
            </div>
          </div>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onAddTask(column.id)}
            className="p-2 bg-white rounded-xl shadow-soft border border-neutral-200 text-neutral-600 hover:text-neutral-800 transition-colors"
          >
            <Plus className="w-4 h-4" />
          </motion.button>
        </div>
      </div>

      {/* 태스크 리스트 */}
      <div 
        ref={setNodeRef} 
        className={`flex-1 space-y-3 min-h-[200px] transition-all duration-200 rounded-2xl p-2 ${
          isOver ? 'bg-blue-50 border-2 border-dashed border-blue-300' : ''
        }`}
      >
        <SortableContext items={column.tasks.map(task => task.id)} strategy={verticalListSortingStrategy}>
          {column.tasks.map((task, index) => {
            const shouldPushDown = dragOverIndex !== null && index >= dragOverIndex;
            const showInsertLine = dragOverIndex === index && !insertAfter;
            const showInsertLineAfter = dragOverIndex === index + 1 && insertAfter === task.id;
            
            return (
              <React.Fragment key={task.id}>
                {showInsertLine && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 6 }}
                    exit={{ opacity: 0, height: 0 }}
                    className="bg-blue-500 rounded-full mx-2 my-2 shadow-lg"
                    style={{ height: '6px' }}
                  />
                )}
                <TaskCard 
                  task={task} 
                  index={index}
                  isOver={overTaskId === task.id}
                  isInsertAfter={insertAfter === task.id}
                  shouldPushDown={shouldPushDown}
                />
                {showInsertLineAfter && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 6 }}
                    exit={{ opacity: 0, height: 0 }}
                    className="bg-blue-500 rounded-full mx-2 my-2 shadow-lg"
                    style={{ height: '6px' }}
                  />
                )}
              </React.Fragment>
            );
          })}
        </SortableContext>
        
        {column.tasks.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center justify-center h-32 border-2 border-dashed border-neutral-200 rounded-2xl text-neutral-400"
          >
            {isOver && dragOverIndex === 0 && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 6 }}
                exit={{ opacity: 0, height: 0 }}
                className="bg-blue-500 rounded-full mx-2 my-2 shadow-lg mb-4"
                style={{ height: '6px', width: '80%' }}
              />
            )}
            <Target className="w-8 h-8 mb-2" />
            <p className="text-sm font-medium">업무가 없습니다</p>
            <p className="text-xs">새 업무를 추가해보세요</p>
          </motion.div>
        )}
      </div>
    </div>
  );
};

const KanbanBoard: React.FC = () => {
  const [activeTask, setActiveTask] = useState<Task | null>(null);
  const [overColumnId, setOverColumnId] = useState<string | null>(null);
  const [overTaskId, setOverTaskId] = useState<string | null>(null);
  const [insertAfter, setInsertAfter] = useState<string | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
  const queryClient = useQueryClient();

  // 태스크 데이터 가져오기
  const { data: tasks = [], isLoading } = useQuery({
    queryKey: ['tasks'],
    queryFn: () => taskAPI.getTasks(),
  });

  // 태스크 상태 업데이트 mutation
  const updateTaskMutation = useMutation({
    mutationFn: ({ taskId, status }: { taskId: string; status: Task['status'] }) =>
      taskAPI.updateTaskStatus(taskId, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
  });

  // 컬럼 설정
  const columns: KanbanColumn[] = [
    {
      id: 'todo',
      title: '대기 중',
      status: 'TODO',
      color: '#F59E0B',
      bgColor: 'bg-accent-amber/10',
      tasks: tasks.filter(task => task.status === 'TODO'),
    },
    {
      id: 'in-progress',
      title: '진행 중',
      status: 'IN_PROGRESS',
      color: '#3B82F6',
      bgColor: 'bg-accent-blue/10',
      tasks: tasks.filter(task => task.status === 'IN_PROGRESS'),
    },
    {
      id: 'done',
      title: '완료',
      status: 'DONE',
      color: '#10B981',
      bgColor: 'bg-accent-green/10',
      tasks: tasks.filter(task => task.status === 'DONE'),
    },
  ];

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 3, // 더 민감하게
        delay: 50, // 더 빠른 반응
      },
    })
  );

  const handleDragStart = (event: DragStartEvent) => {
    const { active } = event;
    const task = tasks.find(t => t.id === active.id);
    setActiveTask(task || null);
  };

  const handleDragOver = (event: DragOverEvent) => {
    const { over, active } = event;
    
    if (!over || active.id === over.id) {
      setOverColumnId(null);
      setOverTaskId(null);
      setInsertAfter(null);
      setDragOverIndex(null);
      return;
    }

    // 컬럼 위에 있는지 확인
    const column = columns.find(col => col.id === over.id);
    if (column) {
      setOverColumnId(column.id);
      setOverTaskId(null);
      setInsertAfter(null);
      setDragOverIndex(0); // 컬럼의 맨 위에 삽입
      return;
    }
    
    // 태스크 위에 있는지 확인
    const task = tasks.find(t => t.id === over.id);
    if (task) {
      setOverTaskId(task.id);
      const taskColumn = columns.find(col => col.tasks.some(t => t.id === task.id));
      if (taskColumn) {
        setOverColumnId(taskColumn.id);
        
        const taskIndex = taskColumn.tasks.findIndex(t => t.id === task.id);
        
        // 마우스 위치에 따라 삽입 위치 결정
        if (event.activatorEvent) {
          const mouseEvent = event.activatorEvent as MouseEvent;
          const taskElement = document.querySelector(`[data-task-id="${task.id}"]`) as HTMLElement;
          
          if (taskElement) {
            const rect = taskElement.getBoundingClientRect();
            const mouseY = mouseEvent.clientY;
            const taskTop = rect.top;
            const taskBottom = rect.bottom;
            const taskHeight = rect.height;
            
            // 카드의 상단 1/4 지점과 하단 1/4 지점을 기준으로 판단 (더 민감하게)
            const upperThreshold = taskTop + taskHeight * 0.25;
            const lowerThreshold = taskBottom - taskHeight * 0.25;
            
            if (mouseY < upperThreshold) {
              // 카드 위쪽에 드래그 - 해당 위치에 삽입
              setInsertAfter(null);
              setDragOverIndex(taskIndex);
            } else if (mouseY > lowerThreshold) {
              // 카드 아래쪽에 드래그 - 다음 위치에 삽입
              setInsertAfter(task.id);
              setDragOverIndex(taskIndex + 1);
            } else {
              // 카드 중앙에 드래그 - 해당 위치에 삽입 (겹치지 않도록)
              setInsertAfter(null);
              setDragOverIndex(taskIndex);
            }
          }
        }
      }
    }
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    
    if (!over) {
          setActiveTask(null);
    setOverColumnId(null);
    setOverTaskId(null);
    setInsertAfter(null);
    setDragOverIndex(null);
      return;
    }

    const activeTaskId = active.id as string;
    const overId = over.id as string;

    // 컬럼으로 드롭한 경우
    const targetColumn = columns.find(col => col.id === overId);
    if (targetColumn) {
      updateTaskMutation.mutate({
        taskId: activeTaskId,
        status: targetColumn.status,
      });
    }

    // 태스크 위에 드롭한 경우 - 해당 위치에 삽입
    const overTask = tasks.find(t => t.id === overId);
    if (overTask) {
      const sourceColumn = columns.find(col => col.tasks.some(t => t.id === activeTaskId));
      const targetColumn = columns.find(col => col.tasks.some(t => t.id === overId));
      
      if (sourceColumn && targetColumn) {
        // 같은 컬럼 내에서 순서만 변경
        if (sourceColumn.id === targetColumn.id) {
          const oldIndex = sourceColumn.tasks.findIndex(t => t.id === activeTaskId);
          const newIndex = targetColumn.tasks.findIndex(t => t.id === overId);
          
          if (oldIndex !== newIndex) {
            const newTasks = arrayMove(sourceColumn.tasks, oldIndex, newIndex);
            // 여기서 실제로는 API 호출로 순서 업데이트
            console.log('Reorder in same column:', { oldIndex, newIndex, newTasks });
          }
        } else {
          // 다른 컬럼으로 이동
          updateTaskMutation.mutate({
            taskId: activeTaskId,
            status: targetColumn.status,
          });
        }
      }
    }

    setActiveTask(null);
    setOverColumnId(null);
    setOverTaskId(null);
  };

  const handleAddTask = (columnId: string) => {
    // TODO: 새 태스크 추가 모달 열기
    console.log('Add task to column:', columnId);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="loading-spinner w-8 h-8"></div>
      </div>
    );
  }

  return (
    <div className="h-full">
      {/* 헤더 */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6"
      >
        <h2 className="text-2xl font-bold text-neutral-900 mb-2">칸반 보드</h2>
        <p className="text-neutral-600">드래그 앤 드롭으로 업무 상태를 변경하세요</p>
      </motion.div>

      {/* 칸반 보드 */}
      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragStart={handleDragStart}
        onDragOver={handleDragOver}
        onDragEnd={handleDragEnd}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 h-full">
          {columns.map((column) => (
            <motion.div
              key={column.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * columns.indexOf(column) }}
              className="flex flex-col"
            >
              <KanbanColumn 
                column={column} 
                onAddTask={handleAddTask}
                isOver={overColumnId === column.id}
                overTaskId={overTaskId}
                insertAfter={insertAfter}
                dragOverIndex={dragOverIndex}
              />
            </motion.div>
          ))}
        </div>

        {/* 드래그 오버레이 */}
        <DragOverlay dropAnimation={{
          duration: 200,
          easing: 'cubic-bezier(0.18, 0.67, 0.6, 1.22)',
        }}>
          {activeTask ? <TaskCard task={activeTask} isDragging /> : null}
        </DragOverlay>
      </DndContext>
    </div>
  );
};

export default KanbanBoard;