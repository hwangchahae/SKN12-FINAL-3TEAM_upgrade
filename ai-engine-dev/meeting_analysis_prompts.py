"""
TtalKkak 회의 분석 프롬프트 시스템
Task Master의 검증된 프롬프트 구조를 회의록 분석에 맞게 적용
"""

def generate_meeting_analysis_system_prompt(num_tasks: int = 5) -> str:
    """회의 분석을 위한 시스템 프롬프트 생성"""
    return f"""You are an AI assistant specialized in analyzing meeting transcripts and generating actionable tasks.

**Your Role:**
- Meeting Analysis Expert
- Task Generation Specialist
- Priority Assessment Analyst
- Korean Language Processing Expert

**Analysis Guidelines:**
1. Create exactly {num_tasks} actionable tasks from the meeting content
2. Each task should be specific, measurable, and implementable
3. Order tasks by priority and logical dependencies
4. Include clear validation criteria for each task
5. Set appropriate priority levels (high/medium/low)
6. Identify responsible parties when mentioned
7. Extract realistic deadlines from context
8. Focus on concrete next steps rather than abstract concepts

**Task Quality Standards:**
- Tasks must be atomic and focused on a single outcome
- Include specific deliverables and success criteria
- Consider dependencies between tasks
- Provide clear testing/validation approaches
- Be actionable within a reasonable timeframe

**Priority Assessment:**
- HIGH: Critical for project success, blocking other tasks, immediate attention required
- MEDIUM: Important but not urgent, can be scheduled within normal workflow
- LOW: Nice to have, can be deferred if resources are limited

**Language Processing:**
- Process Korean meeting content accurately
- Maintain context and nuance in Korean business communication
- Extract action items that may be implied rather than explicitly stated
- Handle Korean business terminology and hierarchy appropriately

**Output Format:**
Always respond with valid JSON structure following the exact schema provided.
No additional text or explanation outside the JSON response.
"""

def generate_meeting_analysis_user_prompt(transcript: str, additional_context: str = "") -> str:
    """회의 분석을 위한 사용자 프롬프트 생성"""
    context_section = f"\n\n**Additional Context:**\n{additional_context}" if additional_context else ""
    
    return f"""Analyze the following meeting transcript and extract actionable tasks:

**Meeting Transcript:**
{transcript}
{context_section}

**Analysis Requirements:**
1. Identify all action items, decisions, and next steps
2. Extract or infer responsible parties (assignees)
3. Determine realistic deadlines based on context
4. Assess priority levels for each task
5. Create a comprehensive meeting summary
6. Identify key decisions and their implications

**Task Generation Instructions:**
- Focus on concrete, actionable items
- Include both explicitly mentioned tasks and implied responsibilities
- Consider follow-up actions that naturally flow from decisions
- Group related activities into coherent tasks
- Ensure each task has clear success criteria

**Response Format:**
Provide a JSON response with the exact structure specified in the schema.
All text should be in Korean unless technical terms require English.
"""

def generate_task_expansion_system_prompt(num_subtasks: int = 3) -> str:
    """태스크 확장을 위한 시스템 프롬프트 (Task Master 스타일)"""
    return f"""You are an AI assistant specialized in breaking down complex tasks into detailed subtasks.

**Your Role:**
- Task Decomposition Expert
- Implementation Planning Specialist
- Risk Assessment Analyst

**Expansion Guidelines:**
1. Create exactly {num_subtasks} subtasks for the given task
2. Each subtask should be a concrete, implementable step
3. Order subtasks logically based on dependencies
4. Include validation criteria for each subtask
5. Consider potential risks and mitigation strategies
6. Focus on practical implementation details

**Subtask Quality Standards:**
- Each subtask must be specific and measurable
- Include clear acceptance criteria
- Consider technical and business constraints
- Provide realistic time estimates
- Include necessary resources and dependencies

**Implementation Focus:**
- Break down complex work into manageable chunks
- Consider both technical and non-technical aspects
- Include testing and validation steps
- Plan for potential obstacles and solutions
- Focus on deliverable outcomes

**Output Format:**
Always respond with valid JSON structure following the exact schema provided.
No additional text or explanation outside the JSON response.
"""

def generate_complexity_analysis_prompt(task_description: str) -> str:
    """복잡도 분석을 위한 프롬프트 생성"""
    return f"""Analyze the complexity of the following task:

**Task Description:**
{task_description}

**Complexity Assessment Criteria:**
1. **Technical Complexity**: Required skills, technologies, integration points
2. **Business Complexity**: Stakeholder involvement, decision points, approval processes
3. **Resource Requirements**: Time, people, tools, budget considerations
4. **Risk Factors**: Dependencies, unknowns, potential blockers
5. **Implementation Scope**: Size of changes, impact on existing systems

**Analysis Instructions:**
- Evaluate each criterion on a scale of 1-3 (Low/Medium/High)
- Consider both apparent and hidden complexities
- Factor in typical organizational constraints
- Assess based on standard development practices
- Include justification for each rating

**Output Format:**
Provide a JSON response with complexity scores and detailed reasoning.
"""

# 회의 분석 결과 후처리 유틸리티
def validate_meeting_analysis(analysis_result: dict) -> dict:
    """회의 분석 결과 검증 및 후처리"""
    validated = {
        "summary": analysis_result.get("summary", "회의 요약이 생성되지 않았습니다."),
        "action_items": [],
        "decisions": analysis_result.get("decisions", []),
        "next_steps": analysis_result.get("next_steps", []),
        "key_points": analysis_result.get("key_points", []),
        "metadata": {
            "total_tasks": 0,
            "high_priority": 0,
            "medium_priority": 0,
            "low_priority": 0
        }
    }
    
    # 액션 아이템 검증 및 정규화
    for item in analysis_result.get("action_items", []):
        if isinstance(item, dict) and "task" in item:
            task = {
                "task": item["task"],
                "assignee": item.get("assignee", "미지정"),
                "deadline": item.get("deadline", "미정"),
                "priority": item.get("priority", "medium").lower(),
                "status": "pending",
                "complexity": calculate_task_complexity(item["task"])
            }
            
            # 우선순위 정규화
            if task["priority"] not in ["high", "medium", "low"]:
                task["priority"] = "medium"
            
            validated["action_items"].append(task)
            validated["metadata"]["total_tasks"] += 1
            validated["metadata"][f"{task['priority']}_priority"] += 1
    
    return validated

def calculate_task_complexity(task_description: str) -> int:
    """태스크 복잡도 계산 (Task Master 알고리즘 적용)"""
    complexity_score = 0
    
    # 기본 복잡도 (설명 길이)
    complexity_score += len(task_description) / 50
    
    # 키워드 기반 복잡도
    high_complexity_keywords = [
        "개발", "구현", "시스템", "통합", "데이터베이스", "API", "아키텍처",
        "설계", "분석", "연구", "조사", "검토", "승인", "협의"
    ]
    
    medium_complexity_keywords = [
        "작성", "문서", "계획", "준비", "정리", "업데이트", "수정", "확인"
    ]
    
    for keyword in high_complexity_keywords:
        if keyword in task_description:
            complexity_score += 2
    
    for keyword in medium_complexity_keywords:
        if keyword in task_description:
            complexity_score += 1
    
    # 1-10 스케일로 정규화
    normalized_score = min(max(int(complexity_score), 1), 10)
    
    return normalized_score

# 프롬프트 템플릿 상수
MEETING_ANALYSIS_SCHEMA = {
    "summary": "회의의 핵심 내용을 2-3줄로 요약",
    "action_items": [
        {
            "task": "구체적인 업무 내용",
            "assignee": "담당자명 또는 부서",
            "deadline": "예상 마감일 (YYYY-MM-DD 또는 상대적 기간)",
            "priority": "high/medium/low",
            "details": "추가 상세 정보 또는 요구사항"
        }
    ],
    "decisions": ["중요한 결정사항들"],
    "next_steps": ["다음 단계 또는 후속 조치들"],
    "key_points": ["회의의 핵심 포인트들"],
    "participants": ["참석자 목록 (추출 가능한 경우)"],
    "follow_up": {
        "next_meeting": "다음 회의 일정 또는 필요성",
        "required_resources": "필요한 자원이나 지원",
        "potential_blockers": "잠재적 장애물이나 이슈"
    }
}