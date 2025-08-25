/**
 * JIRA 연동 서비스
 * Task Master에서 생성된 업무를 JIRA 이슈로 자동 생성
 */

import { PrismaClient } from '@prisma/client';
// import fetch from 'node-fetch'; // Node.js 18+ has built-in fetch

interface JiraIssue {
  key: string;
  id: string;
  summary: string;
  description: string;
  status: string;
  priority: string;
  assignee?: {
    accountId: string;
    displayName: string;
  };
}

interface JiraCreateIssueRequest {
  summary: string;
  description: string;
  issueType: string;
  priority: string;
  assignee?: string | undefined;
  parentKey?: string | undefined; // 서브태스크인 경우
  startDate?: string | undefined; // YYYY-MM-DD 형식
  dueDate?: string | undefined; // YYYY-MM-DD 형식
  epicName?: string | undefined; // Epic의 경우 Epic Name
}

interface JiraCreateIssueResponse {
  key: string;
  id: string;
  self: string;
}

class JiraService {
  private prisma: PrismaClient;

  constructor(prisma: PrismaClient) {
    this.prisma = prisma;
  }

  /**
   * 토큰 복호화
   */
  private decrypt(encryptedText: string): string {
    return Buffer.from(encryptedText, 'base64').toString();
  }

  /**
   * 토큰 암호화
   */
  private encrypt(text: string): string {
    return Buffer.from(text).toString('base64');
  }

  /**
   * JIRA OAuth 토큰 갱신
   */
  private async refreshJiraToken(integration: any) {
    try {
      if (!integration.refreshToken) {
        console.log('❌ Refresh token이 없습니다.');
        return null;
      }

      const refreshToken = this.decrypt(integration.refreshToken);
      const clientId = process.env.JIRA_CLIENT_ID;
      const clientSecret = process.env.JIRA_CLIENT_SECRET;

      if (!clientId || !clientSecret) {
        console.log('❌ JIRA OAuth 앱 설정이 없습니다.');
        return null;
      }

      const response = await fetch('https://auth.atlassian.com/oauth/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          grant_type: 'refresh_token',
          client_id: clientId,
          client_secret: clientSecret,
          refresh_token: refreshToken,
        }),
      });

      if (!response.ok) {
        console.log('❌ 토큰 갱신 실패:', response.status);
        return null;
      }

      const tokens: any = await response.json();
      
      console.log('🔍 토큰 갱신 응답:', {
        access_token: tokens.access_token ? `${tokens.access_token.substring(0, 20)}...` : 'missing',
        refresh_token: tokens.refresh_token ? `${tokens.refresh_token.substring(0, 20)}...` : 'missing',
        token_type: tokens.token_type,
        expires_in: tokens.expires_in
      });
      
      // 갱신된 토큰 저장
      const updatedIntegration = await this.prisma.integration.update({
        where: { id: integration.id },
        data: {
          accessToken: this.encrypt(tokens.access_token),
          refreshToken: this.encrypt(tokens.refresh_token),
        },
      });

      console.log('✅ 갱신된 토큰 저장 완료:', {
        integrationId: updatedIntegration.id,
        hasNewAccessToken: !!updatedIntegration.accessToken,
        hasNewRefreshToken: !!updatedIntegration.refreshToken
      });

      return updatedIntegration;
    } catch (error) {
      console.error('❌ 토큰 갱신 오류:', error);
      return null;
    }
  }

  /**
   * 테넌트의 JIRA 설정 조회
   */
  async getJiraIntegration(tenantId: string, userId: string) {
    return await this.prisma.integration.findFirst({
      where: {
        tenantId,
        userId,
        serviceType: 'JIRA',
        isActive: true
      }
    });
  }

  /**
   * 테넌트의 기본 JIRA 설정 조회 (간단한 설정)
   */
  async getDefaultJiraProject(tenantId: string) {
    // 단순화된 구조에서는 Integration 테이블의 config에서 JIRA 설정을 가져옴
    const integration = await this.prisma.integration.findFirst({
      where: {
        tenantId,
        serviceType: 'JIRA',
        isActive: true
      }
    });

    if (!integration || !integration.config) {
      return null;
    }

    const config = integration.config as any;
    return {
      jiraProjectKey: config.defaultProjectKey || 'TK',
      jiraProjectId: config.defaultProjectId || '10000', // 기본 프로젝트 ID 추가
      defaultIssueType: config.defaultIssueType || 'Task',
      defaultPriority: config.defaultPriority || 'Medium',
      priorityMapping: config.priorityMapping || {
        HIGH: 'High',
        MEDIUM: 'Medium',
        LOW: 'Low'
      }
    };
  }

  /**
   * JIRA API 호출
   */
  private async callJiraAPI(
    integration: any,
    endpoint: string,
    method: string = 'GET',
    data?: any
  ) {
    const config = integration.config || {};
    const cloudId = config.cloudId || config.site_id || config.site?.id;
    
    if (!cloudId) {
      console.error('❌ JIRA cloudId not found in config:', JSON.stringify(config, null, 2));
      throw new Error('JIRA cloudId not configured');
    }

    const url = `https://api.atlassian.com/ex/jira/${cloudId}/rest/api/3${endpoint}`;
    const headers = {
      'Authorization': `Bearer ${this.decrypt(integration.accessToken)}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-Atlassian-Token': 'no-check'
    };
    
    console.log('🔍 JIRA API 호출:', {
      url: url,
      method: method,
      endpoint: endpoint,
      hasData: !!data
    });

    const response = await fetch(url, {
      method,
      headers,
      ...(data && { body: JSON.stringify(data) })
    });

    if (!response.ok) {
      const errorText = await response.text();
      if (response.status === 401) {
        // 토큰 갱신 시도
        console.log('🔄 JIRA 토큰 만료, 갱신 시도 중...');
        const refreshed = await this.refreshJiraToken(integration);
        if (refreshed) {
          console.log('✅ JIRA 토큰 갱신 성공, 재시도 중...');
          
          const newToken = this.decrypt(refreshed.accessToken!);
          console.log('🔍 재시도용 새 토큰:', {
            tokenPreview: `${newToken.substring(0, 20)}...`,
            tokenLength: newToken.length
          });
          
          // 갱신된 토큰으로 재시도
          const newHeaders = {
            'Authorization': `Bearer ${newToken}`,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-Atlassian-Token': 'no-check'
          };
          // 갱신된 토큰 간단한 테스트 먼저 진행
          const testCloudId = integration.config.cloudId || integration.config.site_id || integration.config.site?.id;
          const testUrl = `https://api.atlassian.com/ex/jira/${testCloudId}/rest/api/3/myself`;
          console.log('🔍 토큰 검증 시도 URL:', testUrl);
          const testResponse = await fetch(testUrl, {
            method: 'GET',
            headers: newHeaders
          });
          
          if (testResponse.ok) {
            const userInfo: any = await testResponse.json();
            console.log('✅ 갱신된 토큰 검증 성공:', {
              accountId: userInfo.accountId,
              displayName: userInfo.displayName,
              emailAddress: userInfo.emailAddress
            });
          } else {
            const testError = await testResponse.text();
            console.error('❌ 갱신된 토큰 검증 실패:', testResponse.status, testError);
          }
          
          // 프로젝트 목록도 확인해보기
          const projectsUrl = `https://api.atlassian.com/ex/jira/${testCloudId}/rest/api/3/project`;
          const projectsResponse = await fetch(projectsUrl, {
            method: 'GET',
            headers: newHeaders
          });
          
          if (projectsResponse.ok) {
            const projects: any = await projectsResponse.json();
            console.log('✅ 프로젝트 목록 조회 성공:', projects.map((p: any) => ({ key: p.key, name: p.name })));
          } else {
            const projectsError = await projectsResponse.text();
            console.error('❌ 프로젝트 목록 조회 실패:', projectsResponse.status, projectsError);
          }
          
          const retryResponse = await fetch(url, {
            method,
            headers: newHeaders,
            ...(data && { body: JSON.stringify(data) })
          });
          if (retryResponse.ok) {
            return await retryResponse.json();
          } else {
            const retryErrorText = await retryResponse.text();
            console.error('❌ 토큰 갱신 후 재시도도 실패:', retryResponse.status, retryErrorText);
          }
        }
        throw new Error(`JIRA 토큰이 만료되었습니다. 다시 연동해주세요.`);
      }
      throw new Error(`JIRA API error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    return await response.json();
  }

  /**
   * JIRA 이슈 생성
   */
  async createJiraIssue(
    tenantId: string,
    userId: string,
    request: JiraCreateIssueRequest
  ): Promise<JiraCreateIssueResponse> {
    const integration = await this.getJiraIntegration(tenantId, userId);
    if (!integration) {
      throw new Error('JIRA integration not found');
    }

    const jiraProject = await this.getDefaultJiraProject(tenantId);
    if (!jiraProject) {
      throw new Error('Default JIRA project not configured');
    }

    // 저장된 Start Date 필드들 가져오기
    const config = integration.config as any || {};
    const discoveredStartDateFields = config.startDateFields || [];

    // 우선순위 매핑
    const priorityMapping = jiraProject.priorityMapping || {
      HIGH: 'High',
      MEDIUM: 'Medium',
      LOW: 'Low'
    };

    // Epic의 경우 특별한 필드 처리
    // 프로젝트 키만 사용 (ID는 권한 문제로 사용하지 않음)
    const issueData: any = {
      fields: {
        project: {
          key: jiraProject.jiraProjectKey
        },
        summary: request.summary,
        description: {
          type: 'doc',
          version: 1,
          content: [
            {
              type: 'paragraph',
              content: [
                {
                  type: 'text',
                  text: request.description
                }
              ]
            }
          ]
        },
        issuetype: {
          name: request.issueType
        },
        priority: {
          name: priorityMapping[request.priority] || jiraProject.defaultPriority
        },
        ...(request.assignee && {
          assignee: {
            accountId: request.assignee
          }
        }),
        ...(request.dueDate && {
          duedate: request.dueDate // Due Date (표준 필드)
        })
      }
    };

    // Start Date 필드 설정 (가장 안전한 방법)
    if (request.startDate) {
      console.log(`📅 Start Date 설정: ${request.startDate}`);
      
      // 발견된 필드 중에서 'Start date' 필드만 사용 (가장 안전함)
      const startDateField = discoveredStartDateFields.find((field: any) => 
        field.name && field.name.toLowerCase() === 'start date'
      );
      
      if (startDateField && startDateField.id === 'customfield_10015') {
        issueData.fields.customfield_10015 = request.startDate;
        console.log(`📅 Start date (${startDateField.id})에 날짜 설정됨`);
      } else {
        console.log(`⚠️ 표준 Start date 필드를 사용합니다: ${request.startDate}`);
        // 가장 일반적이고 안전한 Start Date 필드만 사용
        issueData.fields.customfield_10015 = request.startDate;
      }
    }

    // Epic의 경우 Epic Name 필드 추가 (일부 JIRA 인스턴스에서 필요하지만 Task 타입에서는 무시됨)
    if (request.issueType === 'Epic' && request.epicName) {
      try {
        issueData.fields.customfield_10011 = request.epicName; // Epic Name (필드 ID는 환경마다 다를 수 있음)
      } catch (error) {
        console.log('⚠️ Epic Name 필드 설정 실패, 무시됨');
      }
    }

    // Sub-task의 경우에만 parent 필드 추가 (일반 Task는 parent 관계 없이 독립적으로 생성)
    if (request.parentKey && request.issueType.toLowerCase().includes('sub')) {
      issueData.fields.parent = {
        key: request.parentKey
      };
    }

    console.log('🔗 이슈 생성 정보:', {
      issueType: request.issueType,
      parentKey: request.parentKey,
      hasParent: !!issueData.fields.parent,
      summary: request.summary,
      startDate: request.startDate,
      dueDate: request.dueDate,
      hasStartDate: !!request.startDate,
      hasDueDate: !!request.dueDate
    });

    // 이슈 생성 및 필드 유효성 로깅
    console.log('📋 JIRA API 요청 데이터:', JSON.stringify(issueData, null, 2));
    
    try {
      const result = await this.callJiraAPI(integration, '/issue', 'POST', issueData) as any;
      
      console.log('✅ JIRA 이슈 생성 성공:', {
        key: result.key,
        id: result.id,
        hasStartDate: !!request.startDate,
        hasDueDate: !!request.dueDate
      });
      
      return {
        key: result.key,
        id: result.id,
        self: result.self
      };
    } catch (error) {
      console.error('❌ JIRA 이슈 생성 실패:', error);
      throw error;
    }
  }

  /**
   * TaskMaster에서 생성된 Task와 Subtask를 JIRA에 올바르게 매핑
   * TaskMaster TASK → JIRA Epic
   * TaskMaster SUBTASK → JIRA Task (Epic에 연결)
   */
  async syncTaskMasterToJira(
    tenantId: string, 
    userId: string, 
    projectData: {
      title: string;
      overview: string;
      tasks: Array<{
        title: string;
        description: string;
        priority: string;
        estimated_hours: number;
        complexity: string;
        startDate?: string;
        dueDate?: string;
        subtasks?: Array<{
          title: string;
          description: string;
          estimated_hours: number;
          startDate?: string;
          dueDate?: string;
        }>;
      }>;
    }
  ) {
    const results: any[] = [];
    const epics: string[] = [];
    
    try {
      console.log('🎫 TaskMaster → JIRA 매핑 시작:', projectData.title);
      
      // 기존 조회 로직을 건너뛰고 바로 새 프로젝트 생성
      console.log('🎫 새로운 JIRA 프로젝트를 생성합니다...');
      
      // 현재 사용자 정보 조회
      const userResult = await this.getCurrentJiraUser(tenantId, userId);
      const leadAccountId = userResult.success ? userResult.user?.accountId : undefined;
      
      // 고유한 프로젝트 키 생성 (타임스탬프 사용)
      const timestamp = Date.now().toString().slice(-6); // 마지막 6자리
      const projectKey = `TK${timestamp}`;
      const projectName = `TtalKkak_${timestamp}`;
      
      console.log('🎫 생성할 프로젝트:', { key: projectKey, name: projectName });
      
      // TtalKkak 프로젝트 새로 생성
      const createResult = await this.createJiraProject(tenantId, userId, {
        key: projectKey,
        name: projectName,
        description: `TtalKkak AI 프로젝트 관리 시스템 - ${new Date().toISOString()}`,
        leadAccountId
      });
      
      if (!createResult.success) {
        console.error('❌ JIRA 프로젝트 생성 실패:', createResult.error);
        return {
          success: false,
          error: `JIRA 프로젝트 생성에 실패했습니다: ${createResult.error}`,
          results
        };
      }
      
      const targetProject = createResult.project;
      if (!targetProject) {
        console.error('❌ 프로젝트 생성 결과에서 프로젝트 정보를 찾을 수 없습니다.');
        return {
          success: false,
          error: '프로젝트 생성은 성공했지만 프로젝트 정보를 가져올 수 없습니다.',
          results
        };
      }
      
      console.log('✅ JIRA 프로젝트 생성 완료:', targetProject);
      
      // 커스텀 필드 조회 (Start Date 필드 확인용)
      const customFieldsResult = await this.getProjectCustomFields(tenantId, userId, targetProject.key);
      let discoveredStartDateFields: any[] = [];
      if (customFieldsResult.success && customFieldsResult.startDateFields) {
        discoveredStartDateFields = customFieldsResult.startDateFields;
        console.log(`📅 Start Date 필드 발견: ${discoveredStartDateFields.length}개`);
      }
      
      // 기본 이슈 타입 사용 (조회하지 않고 Task만 사용)
      const issueTypes = {
        epic: 'Task',
        story: 'Task', 
        task: 'Task'
      };
      
      console.log('✅ 기본 이슈 타입 사용:', issueTypes);
      
      // 프로젝트 설정 업데이트
      const integration = await this.getJiraIntegration(tenantId, userId);
      if (integration && targetProject) {
        const currentConfig = integration.config as any || {};
        const updatedConfig = {
          ...currentConfig,
          defaultProjectKey: targetProject.key,
          defaultProjectId: targetProject.id,
          issueTypes: issueTypes,
          startDateFields: discoveredStartDateFields // 발견된 Start Date 필드들 저장
        };
        await this.prisma.integration.update({
          where: { id: integration.id },
          data: { config: updatedConfig }
        });
        console.log('✅ JIRA 프로젝트 설정 업데이트됨:', updatedConfig);
      }
      
      // TaskMaster의 각 TASK를 JIRA Epic으로 생성
      for (const task of projectData.tasks) {
        try {
          // 1. TaskMaster TASK → JIRA Epic 생성 (타임라인 표시를 위해)
          const epicIssue = await this.createJiraIssue(tenantId, userId, {
            summary: task.title,
            description: task.description,
            issueType: issueTypes.epic || 'Task',
            priority: task.priority || 'MEDIUM',
            epicName: task.title,
            startDate: task.startDate,
            dueDate: task.dueDate
          });
          
          console.log(`✅ Epic 생성 (TaskMaster Task): ${epicIssue.key} - ${task.title}`);
          epics.push(epicIssue.key);
          results.push({
            type: 'Epic',
            key: epicIssue.key,
            title: task.title,
            source: 'TaskMaster Task',
            success: true
          });
          
          // 2. TaskMaster SUBTASK들을 JIRA Story로 생성 (Epic에 연결)
          if (task.subtasks && task.subtasks.length > 0) {
            for (const subtask of task.subtasks) {
              try {
                // SubTask 없이 독립적인 Task로 생성 (타임라인에서 연관성 유지)
                const jiraTaskIssue = await this.createJiraIssue(tenantId, userId, {
                  summary: `[${task.title}] ${subtask.title}`, // 부모 Task 이름을 포함
                  description: `상위 업무: ${task.title}\n\n${subtask.description}`,
                  issueType: issueTypes.task || 'Task',
                  priority: 'MEDIUM',
                  // parentKey 제거 - 독립적인 Task로 생성
                  startDate: subtask.startDate,
                  dueDate: subtask.dueDate
                });
                
                console.log(`✅ Task 생성 (TaskMaster Subtask): ${jiraTaskIssue.key} - ${subtask.title}`);
                results.push({
                  type: 'Task',
                  key: jiraTaskIssue.key,
                  title: subtask.title,
                  parentKey: epicIssue.key,
                  source: 'TaskMaster Subtask',
                  success: true
                });
              } catch (subtaskError) {
                console.error(`❌ Task 생성 실패 (TaskMaster Subtask: ${subtask.title}):`, subtaskError);
                results.push({
                  type: 'Task',
                  title: subtask.title,
                  parentKey: epicIssue.key,
                  source: 'TaskMaster Subtask',
                  error: subtaskError instanceof Error ? subtaskError.message : 'Unknown error',
                  success: false
                });
              }
            }
          }
          
        } catch (taskError) {
          console.error(`❌ Epic 생성 실패 (TaskMaster Task: ${task.title}):`, taskError);
          results.push({
            type: 'Epic',
            title: task.title,
            source: 'TaskMaster Task',
            error: taskError instanceof Error ? taskError.message : 'Unknown error',
            success: false
          });
        }
      }
      
      const successCount = results.filter(r => r.success).length;
      const totalCount = results.length;
      const epicsCreated = results.filter(r => r.type === 'Epic' && r.success).length;
      const tasksCreated = results.filter(r => r.type === 'Task' && r.success).length;
      
      console.log(`✅ JIRA 매핑 완료: Epic ${epicsCreated}개, Task ${tasksCreated}개 (총 ${successCount}/${totalCount})`);
      
      return {
        success: true,
        epics: epics,
        projectKey: targetProject.key, // 새로 생성된 프로젝트 키 반환
        totalCreated: successCount,
        totalAttempted: totalCount,
        epicsCreated,
        tasksCreated,
        results
      };
      
    } catch (error) {
      console.error('❌ TaskMaster → JIRA 매핑 실패:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        results
      };
    }
  }

  /**
   * Task를 JIRA 이슈로 동기화
   */
  async syncTaskToJira(taskId: string, userId: string) {
    const task = await this.prisma.task.findUnique({
      where: { id: taskId },
      include: {
        tenant: true,
        assignee: true,
        parent: {
          include: {
            metadata: true
          }
        },
        children: true,
        metadata: true
      }
    });

    if (!task) {
      throw new Error('Task not found');
    }

    // 이미 동기화된 경우 스킵
    if (task.metadata?.jiraStatus === 'synced' && task.metadata?.jiraIssueKey) {
      return task.metadata.jiraIssueKey;
    }

    try {
      // 동기화 시작 - TaskMetadata 업데이트
      await this.prisma.taskMetadata.upsert({
        where: { taskId },
        update: { jiraStatus: 'syncing' },
        create: { taskId, jiraStatus: 'syncing' }
      });

      // 부모 태스크가 있는 경우 부모를 먼저 동기화
      let parentJiraKey = null;
      if (task.parent && !task.parent.metadata?.jiraIssueKey) {
        parentJiraKey = await this.syncTaskToJira(task.parent.id, userId);
      } else if (task.parent) {
        parentJiraKey = task.parent.metadata?.jiraIssueKey;
      }

      // JIRA 이슈 생성
      const jiraIssue = await this.createJiraIssue(task.tenantId, userId, {
        summary: task.title,
        description: task.description || '',
        issueType: parentJiraKey ? 'Sub-task' : 'Task',
        priority: task.priority,
        assignee: task.assignee?.jiraUserId || undefined,
        parentKey: parentJiraKey || undefined
      });

      // TaskMetadata 업데이트
      await this.prisma.taskMetadata.upsert({
        where: { taskId },
        update: {
          jiraIssueKey: jiraIssue.key,
          jiraStatus: 'synced'
        },
        create: {
          taskId,
          jiraIssueKey: jiraIssue.key,
          jiraStatus: 'synced'
        }
      });

      return jiraIssue.key;

    } catch (error) {
      // 동기화 실패
      await this.prisma.taskMetadata.upsert({
        where: { taskId },
        update: { jiraStatus: 'failed' },
        create: { taskId, jiraStatus: 'failed' }
      });

      throw error;
    }
  }

  /**
   * 프로젝트에서 생성된 모든 업무를 JIRA로 동기화
   */
  async syncProjectTasksToJira(projectId: string, userId: string) {
    const project = await this.prisma.project.findUnique({
      where: { id: projectId },
      include: {
        tasks: {
          include: {
            metadata: true
          },
          orderBy: { taskNumber: 'asc' }
        }
      }
    });

    if (!project) {
      throw new Error('Project not found');
    }

    const results = [];
    
    // 아직 동기화되지 않은 task만 필터링
    const pendingTasks = project.tasks.filter(task => 
      !task.metadata?.jiraStatus || 
      task.metadata.jiraStatus === 'pending' || 
      task.metadata.jiraStatus === 'failed'
    );
    
    // 부모 태스크부터 순차적으로 동기화 (taskNumber 순서)
    for (const task of pendingTasks) {
      try {
        const jiraKey = await this.syncTaskToJira(task.id, userId);
        results.push({ taskId: task.id, jiraKey, success: true });
      } catch (error) {
        results.push({ 
          taskId: task.id, 
          error: error instanceof Error ? error.message : 'Unknown error',
          success: false 
        });
      }
    }

    return results;
  }

  /**
   * JIRA 프로젝트 생성
   */
  async createJiraProject(tenantId: string, userId: string, projectData: {
    key: string;
    name: string;
    description?: string;
    projectTypeKey?: string;
    projectTemplateKey?: string;
    leadAccountId?: string;
  }) {
    const integration = await this.getJiraIntegration(tenantId, userId);
    if (!integration) {
      return { success: false, error: 'Integration not found' };
    }

    try {
      const projectPayload = {
        key: projectData.key,
        name: projectData.name,
        description: projectData.description || `${projectData.name} 프로젝트`,
        projectTypeKey: projectData.projectTypeKey || 'software',
        projectTemplateKey: projectData.projectTemplateKey || 'com.pyxis.greenhopper.jira:gh-kanban-template',
        leadAccountId: projectData.leadAccountId
      };

      console.log('🎫 JIRA 프로젝트 생성 시도:', projectPayload);
      const result: any = await this.callJiraAPI(integration, '/project', 'POST', projectPayload);
      
      console.log('✅ JIRA 프로젝트 생성 성공:', result);
      return { 
        success: true, 
        project: {
          key: result.key,
          id: result.id,
          name: projectData.name
        }
      };
    } catch (error) {
      console.error('❌ JIRA 프로젝트 생성 실패:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }

  /**
   * JIRA 사용자 정보 조회 (프로젝트 리드 설정용)
   */
  async getCurrentJiraUser(tenantId: string, userId: string) {
    const integration = await this.getJiraIntegration(tenantId, userId);
    if (!integration) {
      return { success: false, error: 'Integration not found' };
    }

    try {
      const userInfo: any = await this.callJiraAPI(integration, '/myself');
      return { 
        success: true, 
        user: {
          accountId: userInfo.accountId,
          displayName: userInfo.displayName,
          emailAddress: userInfo.emailAddress
        }
      };
    } catch (error) {
      console.error('❌ JIRA 사용자 정보 조회 실패:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }

  /**
   * 프로젝트의 이슈 타입 목록 조회
   */
  async getProjectIssueTypes(tenantId: string, userId: string, projectKey: string) {
    const integration = await this.getJiraIntegration(tenantId, userId);
    if (!integration) {
      return { success: false, error: 'Integration not found' };
    }

    try {
      const issueTypes: any = await this.callJiraAPI(integration, `/project/${projectKey}/issueTypes`);
      console.log('✅ JIRA 이슈 타입 목록:', issueTypes.map((it: any) => ({ id: it.id, name: it.name, subtask: it.subtask })));
      return { 
        success: true, 
        issueTypes: issueTypes.map((it: any) => ({ 
          id: it.id, 
          name: it.name,
          subtask: it.subtask || false
        }))
      };
    } catch (error) {
      console.error('❌ JIRA 이슈 타입 조회 실패:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }

  /**
   * JIRA 프로젝트 목록 조회
   */
  async getJiraProjects(tenantId: string, userId: string) {
    const integration = await this.getJiraIntegration(tenantId, userId);
    if (!integration) {
      return { success: false, error: 'Integration not found' };
    }

    try {
      const projects: any = await this.callJiraAPI(integration, '/project');
      console.log('✅ JIRA 프로젝트 목록:', projects.map((p: any) => ({ key: p.key, name: p.name })));
      return { 
        success: true, 
        projects: projects.map((p: any) => ({ 
          key: p.key, 
          name: p.name, 
          id: p.id 
        }))
      };
    } catch (error) {
      console.error('❌ JIRA 프로젝트 목록 조회 실패:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }

  /**
   * 프로젝트의 커스텀 필드 목록 조회 (Start Date 필드 찾기용)
   */
  async getProjectCustomFields(tenantId: string, userId: string, projectKey: string) {
    const integration = await this.getJiraIntegration(tenantId, userId);
    if (!integration) {
      return { success: false, error: 'Integration not found' };
    }

    try {
      const fields: any = await this.callJiraAPI(integration, '/field');
      
      // Start Date와 관련된 필드들 찾기 (더 정확한 필터링)
      const startDateFields = fields.filter((field: any) => 
        field.name && field.custom && (
          (field.name.toLowerCase().includes('start') && 
           field.name.toLowerCase().includes('date')) ||
          field.name.toLowerCase() === 'start date' ||
          field.name.toLowerCase() === 'target start' ||
          field.name.toLowerCase() === 'actual start'
        )
      );
      
      console.log('📅 발견된 Start Date 관련 필드들:', startDateFields.map((f: any) => ({ 
        id: f.id, 
        name: f.name, 
        custom: f.custom 
      })));
      
      return { 
        success: true, 
        startDateFields: startDateFields.map((f: any) => ({ 
          id: f.id, 
          name: f.name,
          custom: f.custom
        }))
      };
    } catch (error) {
      console.error('❌ JIRA 커스텀 필드 조회 실패:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }

  /**
   * JIRA 연동 상태 확인
   */
  async checkJiraConnection(tenantId: string, userId: string) {
    const integration = await this.getJiraIntegration(tenantId, userId);
    if (!integration) {
      return { connected: false, error: 'Integration not found' };
    }

    try {
      // JIRA API 호출 대신 단순히 토큰과 설정 존재 여부만 확인
      const config = integration.config as any || {};
      const siteUrl = config.site_url;
      const siteName = config.site_name;
      
      if (!siteUrl || !siteName) {
        return { connected: false, error: 'JIRA site configuration incomplete' };
      }
      
      if (!integration.accessToken) {
        return { connected: false, error: 'Access token not found' };
      }
      
      const accessToken = this.decrypt(integration.accessToken);
      if (!accessToken) {
        return { connected: false, error: 'Invalid access token' };
      }
      
      return { 
        connected: true, 
        site_name: siteName,
        site_url: siteUrl,
        user: { displayName: 'Connected User' }
      };
    } catch (error) {
      return { 
        connected: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }
}

export { JiraService };