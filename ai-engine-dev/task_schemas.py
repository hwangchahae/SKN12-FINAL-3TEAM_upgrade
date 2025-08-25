"""
TtalKkak 태스크 스키마 정의
Task Master의 Zod 스키마를 Pydantic으로 구현
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

class TaskPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

class SubTask(BaseModel):
    """서브태스크 모델"""
    id: int = Field(..., description="서브태스크 ID")
    title: str = Field(..., min_length=1, description="서브태스크 제목")
    description: str = Field(..., min_length=1, description="서브태스크 설명")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="서브태스크 상태")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="서브태스크 우선순위")
    estimated_hours: Optional[int] = Field(None, ge=0, description="예상 소요 시간")
    completed_at: Optional[datetime] = Field(None, description="완료 시간")
    
    class Config:
        use_enum_values = True

class TaskItem(BaseModel):
    """메인 태스크 모델 (Task Master 스타일)"""
    id: int = Field(..., description="태스크 ID")
    title: str = Field(..., min_length=1, description="태스크 제목")
    description: str = Field(..., min_length=1, description="태스크 상세 설명")
    details: Optional[str] = Field(None, description="추가 세부사항")
    
    # 우선순위 및 상태
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="태스크 우선순위")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="태스크 상태")
    
    # 할당 및 일정
    assignee: Optional[str] = Field(None, description="담당자")
    deadline: Optional[str] = Field(None, description="마감일")
    estimated_hours: Optional[int] = Field(None, ge=0, description="예상 소요 시간")
    
    # 복잡도 및 의존성
    complexity: int = Field(default=1, ge=1, le=10, description="복잡도 (1-10)")
    dependencies: List[int] = Field(default_factory=list, description="의존성 태스크 ID 목록")
    
    # 검증 및 테스트
    test_strategy: Optional[str] = Field(None, description="테스트 전략")
    acceptance_criteria: List[str] = Field(default_factory=list, description="수락 기준")
    
    # 서브태스크
    subtasks: List[SubTask] = Field(default_factory=list, description="서브태스크 목록")
    
    # 메타데이터
    tags: List[str] = Field(default_factory=list, description="태그 목록")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    updated_at: Optional[datetime] = Field(None, description="수정 시간")
    
    class Config:
        use_enum_values = True

class MeetingAnalysisResult(BaseModel):
    """회의 분석 결과 모델"""
    summary: str = Field(..., min_length=1, description="회의 요약")
    
    # 액션 아이템 (TaskItem 형태로 구조화)
    action_items: List[TaskItem] = Field(default_factory=list, description="액션 아이템 목록")
    
    # 회의 결과
    decisions: List[str] = Field(default_factory=list, description="중요한 결정사항")
    next_steps: List[str] = Field(default_factory=list, description="다음 단계")
    key_points: List[str] = Field(default_factory=list, description="핵심 포인트")
    
    # 참석자 정보
    participants: List[str] = Field(default_factory=list, description="참석자 목록")
    
    # 후속 조치
    follow_up: Dict[str, Any] = Field(default_factory=dict, description="후속 조치 정보")
    
    # 메타데이터
    metadata: Dict[str, Any] = Field(default_factory=dict, description="분석 메타데이터")
    
    @validator('action_items')
    def validate_action_items(cls, v):
        """액션 아이템 검증"""
        if not isinstance(v, list):
            return []
        
        # ID 재할당
        for i, item in enumerate(v):
            if hasattr(item, 'id'):
                item.id = i + 1
        
        return v

class ComplexityAnalysis(BaseModel):
    """복잡도 분석 결과"""
    overall_score: int = Field(..., ge=1, le=10, description="전체 복잡도 점수")
    
    # 세부 복잡도 요소
    technical_complexity: int = Field(..., ge=1, le=3, description="기술적 복잡도")
    business_complexity: int = Field(..., ge=1, le=3, description="비즈니스 복잡도")
    resource_requirements: int = Field(..., ge=1, le=3, description="자원 요구사항")
    risk_factors: int = Field(..., ge=1, le=3, description="위험 요소")
    implementation_scope: int = Field(..., ge=1, le=3, description="구현 범위")
    
    # 상세 분석
    analysis_details: Dict[str, str] = Field(default_factory=dict, description="세부 분석 내용")
    recommendations: List[str] = Field(default_factory=list, description="추천사항")
    
    class Config:
        use_enum_values = True

class TaskExpansionRequest(BaseModel):
    """태스크 확장 요청"""
    task_id: int = Field(..., description="확장할 태스크 ID")
    num_subtasks: int = Field(default=3, ge=1, le=10, description="생성할 서브태스크 수")
    additional_context: Optional[str] = Field(None, description="추가 컨텍스트")
    focus_areas: List[str] = Field(default_factory=list, description="집중할 영역")

class TaskExpansionResult(BaseModel):
    """태스크 확장 결과"""
    original_task: TaskItem = Field(..., description="원본 태스크")
    expanded_subtasks: List[SubTask] = Field(..., description="확장된 서브태스크")
    expansion_reasoning: str = Field(..., description="확장 근거")
    estimated_total_hours: Optional[int] = Field(None, description="전체 예상 소요시간")

class PipelineRequest(BaseModel):
    """전체 파이프라인 요청"""
    audio_filename: str = Field(..., description="오디오 파일명")
    meeting_title: Optional[str] = Field(None, description="회의 제목")
    num_tasks: int = Field(default=5, ge=1, le=20, description="생성할 태스크 수")
    additional_context: Optional[str] = Field(None, description="추가 컨텍스트")
    auto_expand_tasks: bool = Field(default=False, description="태스크 자동 확장 여부")

class PipelineResult(BaseModel):
    """전체 파이프라인 결과"""
    success: bool = Field(..., description="처리 성공 여부")
    step: str = Field(..., description="현재 처리 단계")
    
    # 각 단계 결과
    transcription: Optional[Dict[str, Any]] = Field(None, description="전사 결과")
    analysis: Optional[MeetingAnalysisResult] = Field(None, description="분석 결과")
    
    # 에러 정보
    error: Optional[str] = Field(None, description="에러 메시지")
    
    # 메타데이터
    processing_time: Optional[float] = Field(None, description="처리 시간 (초)")
    model_info: Dict[str, Any] = Field(default_factory=dict, description="사용된 모델 정보")

# 스키마 검증 유틸리티
def validate_task_dependencies(tasks: List[TaskItem]) -> List[TaskItem]:
    """태스크 의존성 검증 및 정리"""
    task_ids = {task.id for task in tasks}
    
    for task in tasks:
        # 존재하지 않는 의존성 제거
        valid_dependencies = [dep for dep in task.dependencies if dep in task_ids]
        task.dependencies = valid_dependencies
        
        # 순환 의존성 검사 (간단한 버전)
        if task.id in task.dependencies:
            task.dependencies.remove(task.id)
    
    return tasks

def calculate_task_complexity_advanced(task: TaskItem) -> int:
    """고급 복잡도 계산 (Task Master 알고리즘)"""
    complexity_score = 0
    
    # 기본 복잡도 (설명 길이)
    complexity_score += len(task.description) / 100
    
    # 의존성 복잡도
    complexity_score += len(task.dependencies) * 1.5
    
    # 서브태스크 복잡도
    complexity_score += len(task.subtasks) * 1.2
    
    # 키워드 기반 복잡도
    high_complexity_keywords = [
        "개발", "구현", "시스템", "통합", "데이터베이스", "API", "아키텍처",
        "설계", "분석", "연구", "조사", "검토", "승인", "협의", "프로토타입"
    ]
    
    medium_complexity_keywords = [
        "작성", "문서", "계획", "준비", "정리", "업데이트", "수정", "확인",
        "테스트", "검증", "배포", "설치", "구성"
    ]
    
    for keyword in high_complexity_keywords:
        if keyword in task.description or keyword in task.title:
            complexity_score += 2
    
    for keyword in medium_complexity_keywords:
        if keyword in task.description or keyword in task.title:
            complexity_score += 1
    
    # 우선순위 가중치
    priority_weights = {
        TaskPriority.HIGH: 1.5,
        TaskPriority.MEDIUM: 1.0,
        TaskPriority.LOW: 0.7
    }
    
    complexity_score *= priority_weights.get(task.priority, 1.0)
    
    # 1-10 스케일로 정규화
    normalized_score = min(max(int(complexity_score), 1), 10)
    
    return normalized_score

def generate_task_id_sequence(existing_tasks: List[TaskItem]) -> int:
    """새로운 태스크 ID 생성"""
    if not existing_tasks:
        return 1
    
    max_id = max(task.id for task in existing_tasks)
    return max_id + 1

# JSON 스키마 예시 (프롬프트에서 사용)
TASK_SCHEMA_EXAMPLE = {
    "summary": "프로젝트 킥오프 회의에서 개발 계획과 역할 분담을 논의했습니다.",
    "action_items": [
        {
            "id": 1,
            "title": "요구사항 문서 작성",
            "description": "클라이언트 요구사항을 정리하여 상세 문서를 작성한다",
            "priority": "high",
            "assignee": "김개발",
            "deadline": "2025-01-15",
            "complexity": 6
        },
        {
            "id": 2,
            "title": "개발 환경 설정",
            "description": "로컬 및 스테이징 환경을 구성한다",
            "priority": "medium",
            "assignee": "이운영",
            "deadline": "2025-01-20",
            "complexity": 4
        }
    ],
    "decisions": [
        "React를 프론트엔드 프레임워크로 선택",
        "PostgreSQL을 메인 데이터베이스로 사용"
    ],
    "next_steps": [
        "다음 주 화요일 기술 리뷰 미팅 예정",
        "요구사항 문서 완성 후 클라이언트 검토 요청"
    ],
    "key_points": [
        "프로젝트 기간은 3개월로 확정",
        "팀 구성원은 총 5명",
        "매주 화요일 정기 미팅 진행"
    ]
}