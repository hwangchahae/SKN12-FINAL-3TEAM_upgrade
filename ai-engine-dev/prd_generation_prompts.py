"""
TtalKkak PRD 생성 프롬프트 시스템
회의록 → 기획안 → Task Master PRD 형식 변환
"""

# 노션용 기획안 템플릿
NOTION_PROJECT_TEMPLATE = """
## 프로젝트 개요

**프로젝트명**: {project_name}

**목적**: {project_purpose}

**수행기간**: {project_period}

**담당자**: {project_manager}

## 핵심 목표

{core_objectives}

## 세부 내용

- **핵심 아이디어**
    
    {core_idea}
    
- **아이디어 기술**
    
    {idea_description}
    
- **실행 계획**
    
    {execution_plan}

## 기대 효과

{expected_effects}
"""

# Task Master PRD 템플릿
TASK_MASTER_PRD_TEMPLATE = """
# Overview  
{overview}

# Core Features  
{core_features}

# User Experience  
{user_experience}

# Technical Architecture  
{technical_architecture}

# Development Roadmap  
{development_roadmap}

# Logical Dependency Chain
{logical_dependency_chain}

# Risks and Mitigations  
{risks_and_mitigations}

# Appendix  
{appendix}
"""

def generate_notion_project_prompt(meeting_transcript: str) -> str:
    """노션 기획안 생성 프롬프트"""
    return f"""
다음 회의 전사본을 바탕으로 노션에 업로드할 프로젝트 기획안을 작성하세요.

**회의 전사본:**
{meeting_transcript}

**작성 지침:**
1. 회의에서 논의된 내용을 바탕으로 체계적인 기획안을 작성
2. 프로젝트명은 회의 내용을 바탕으로 적절히 명명
3. 목적과 목표는 명확하고 구체적으로 작성
4. 실행 계획은 실현 가능한 단계별로 구성
5. 기대 효과는 정량적/정성적 결과를 포함
6. 모든 내용은 한국어로 작성

**응답 형식:**
다음 JSON 형식으로 응답하세요:
{{
    "project_name": "프로젝트명",
    "project_purpose": "프로젝트의 주요 목적",
    "project_period": "예상 수행 기간 (예: 2025.01.01 ~ 2025.03.31)",
    "project_manager": "담당자명 (회의에서 언급된 경우)",
    "core_objectives": [
        "목표 1: 구체적인 목표",
        "목표 2: 구체적인 목표",
        "목표 3: 구체적인 목표"
    ],
    "core_idea": "핵심 아이디어 설명",
    "idea_description": "아이디어의 기술적/비즈니스적 설명",
    "execution_plan": "단계별 실행 계획과 일정",
    "expected_effects": [
        "기대효과 1: 자세한 설명",
        "기대효과 2: 자세한 설명",
        "기대효과 3: 자세한 설명"
    ]
}}
"""

def generate_task_master_prd_prompt(notion_project: dict) -> str:
    """Task Master PRD 변환 프롬프트"""
    return f"""
다음 기획안을 Task Master가 사용하는 PRD 형식으로 변환하세요.

**기획안 정보:**
- 프로젝트명: {notion_project.get('project_name', '')}
- 목적: {notion_project.get('project_purpose', '')}
- 핵심 아이디어: {notion_project.get('core_idea', '')}
- 실행 계획: {notion_project.get('execution_plan', '')}
- 핵심 목표: {', '.join(notion_project.get('core_objectives', []))}
- 기대 효과: {', '.join(notion_project.get('expected_effects', []))}

**변환 지침:**
1. Overview: 프로젝트의 전체적인 개요, 해결하는 문제, 타겟 사용자, 가치 제안
2. Core Features: 주요 기능들을 구체적으로 나열하고 각각의 중요성과 작동 방식 설명
3. User Experience: 사용자 여정, 페르소나, 주요 플로우, UI/UX 고려사항
4. Technical Architecture: 시스템 구성요소, 데이터 모델, API, 인프라 요구사항
5. Development Roadmap: MVP 요구사항, 향후 개선사항, 단계별 개발 범위
6. Logical Dependency Chain: 개발 순서, 기초 기능 우선순위, 점진적 개발 방식
7. Risks and Mitigations: 기술적 도전, MVP 정의, 자원 제약사항
8. Appendix: 추가 정보, 연구 결과, 기술 사양

**응답 형식:**
다음 JSON 형식으로 응답하세요:
{{
    "overview": "프로젝트 전체 개요",
    "core_features": "주요 기능들 상세 설명",
    "user_experience": "사용자 경험 설계",
    "technical_architecture": "기술 아키텍처 설명",
    "development_roadmap": "개발 로드맵",
    "logical_dependency_chain": "논리적 의존성 체인",
    "risks_and_mitigations": "위험 요소 및 완화 방안",
    "appendix": "추가 정보 및 참고 자료"
}}
"""

def format_notion_project(project_data: dict) -> str:
    """노션 기획안 포맷팅"""
    
    # 핵심 목표 포맷팅
    core_objectives_formatted = "\n".join([f"- {obj}" for obj in project_data.get('core_objectives', [])])
    
    # 기대 효과 포맷팅
    expected_effects_formatted = "\n".join([f"- {effect}" for effect in project_data.get('expected_effects', [])])
    
    return f"""## 프로젝트 개요

**프로젝트명**: {project_data.get('project_name', '[프로젝트명 입력]')}

**목적**: {project_data.get('project_purpose', '[프로젝트의 주요 목적]')}

**수행기간**: {project_data.get('project_period', '[시작일] ~ [종료일]')}

**담당자**: {project_data.get('project_manager', '[담당자명]')}

## 핵심 목표

{core_objectives_formatted}

## 세부 내용

- **핵심 아이디어**
    
    {project_data.get('core_idea', '핵심아이디어')}
    
- **아이디어 기술**
    
    {project_data.get('idea_description', '아이디어의 기술')}
    
- **실행 계획**
    
    {project_data.get('execution_plan', '단계별 실행 계획과 일정을 작성하세요.')}

## 기대 효과

{expected_effects_formatted}
"""

def format_task_master_prd(prd_data: dict) -> str:
    """Task Master PRD 포맷팅"""
    # 백슬래시 문제 해결을 위해 문자열 변수 사용
    apostrophe = "'"
    overview_default = f"[Provide a high-level overview of your product here. Explain what problem it solves, who it{apostrophe}s for, and why it{apostrophe}s valuable.]"
    features_default = f"[List and describe the main features of your product. For each feature, include: What it does, Why it{apostrophe}s important, How it works at a high level]"
    risks_default = f"[Identify potential risks and how they{apostrophe}ll be addressed: Technical challenges, Figuring out the MVP that we can build upon, Resource constraints]"
    
    return f"""# Overview  
{prd_data.get('overview', overview_default)}

# Core Features  
{prd_data.get('core_features', features_default)}

# User Experience  
{prd_data.get('user_experience', '[Describe the user journey and experience. Include: User personas, Key user flows, UI/UX considerations]')}

# Technical Architecture  
{prd_data.get('technical_architecture', '[Outline the technical implementation details: System components, Data models, APIs and integrations, Infrastructure requirements]')}

# Development Roadmap  
{prd_data.get('development_roadmap', '[Break down the development process into phases: MVP requirements, Future enhancements, Do not think about timelines whatsoever -- all that matters is scope and detailing exactly what needs to be build in each phase so it can later be cut up into tasks]')}

# Logical Dependency Chain
{prd_data.get('logical_dependency_chain', '[Define the logical order of development: Which features need to be built first (foundation), Getting as quickly as possible to something usable/visible front end that works, Properly pacing and scoping each feature so it is atomic but can also be built upon and improved as development approaches]')}

# Risks and Mitigations  
{prd_data.get('risks_and_mitigations', risks_default)}

# Appendix  
{prd_data.get('appendix', '[Include any additional information: Research findings, Technical specifications]')}
"""

# 2단계 프로세스 검증 함수
def validate_notion_project(project_data: dict) -> dict:
    """노션 기획안 데이터 검증"""
    required_fields = [
        'project_name', 'project_purpose', 'core_objectives', 
        'core_idea', 'execution_plan', 'expected_effects'
    ]
    
    validated = {}
    for field in required_fields:
        if field not in project_data or not project_data[field]:
            if field == 'core_objectives' or field == 'expected_effects':
                validated[field] = [f"[{field} 입력 필요]"]
            else:
                validated[field] = f"[{field} 입력 필요]"
        else:
            validated[field] = project_data[field]
    
    # 선택적 필드들
    optional_fields = ['project_period', 'project_manager', 'idea_description']
    for field in optional_fields:
        validated[field] = project_data.get(field, f"[{field} 입력 필요]")
    
    return validated

def validate_task_master_prd(prd_data: dict) -> dict:
    """Task Master PRD 데이터 검증"""
    required_sections = [
        'overview', 'core_features', 'user_experience', 
        'technical_architecture', 'development_roadmap',
        'logical_dependency_chain', 'risks_and_mitigations', 'appendix'
    ]
    
    validated = {}
    for section in required_sections:
        if section not in prd_data or not prd_data[section]:
            validated[section] = f"[{section} 섹션 내용 필요]"
        else:
            validated[section] = prd_data[section]
    
    return validated

# 프롬프트 생성 유틸리티
def generate_two_stage_prompts(meeting_transcript: str):
    """2단계 프로세스 프롬프트 생성"""
    return {
        "stage1_notion": generate_notion_project_prompt(meeting_transcript),
        "stage2_prd": None  # 1단계 결과를 받은 후 생성
    }

# 응답 스키마
NOTION_PROJECT_SCHEMA = {
    "project_name": "프로젝트명",
    "project_purpose": "프로젝트의 주요 목적",
    "project_period": "예상 수행 기간",
    "project_manager": "담당자명",
    "core_objectives": [
        "목표 1: 구체적인 목표",
        "목표 2: 구체적인 목표",
        "목표 3: 구체적인 목표"
    ],
    "core_idea": "핵심 아이디어 설명",
    "idea_description": "아이디어의 기술적/비즈니스적 설명",
    "execution_plan": "단계별 실행 계획과 일정",
    "expected_effects": [
        "기대효과 1: 자세한 설명",
        "기대효과 2: 자세한 설명",
        "기대효과 3: 자세한 설명"
    ]
}

TASK_MASTER_PRD_SCHEMA = {
    "overview": "프로젝트 전체 개요",
    "core_features": "주요 기능들 상세 설명",
    "user_experience": "사용자 경험 설계",
    "technical_architecture": "기술 아키텍처 설명",
    "development_roadmap": "개발 로드맵",
    "logical_dependency_chain": "논리적 의존성 체인",
    "risks_and_mitigations": "위험 요소 및 완화 방안",
    "appendix": "추가 정보 및 참고 자료"
}