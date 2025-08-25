/**
 * 스마트 업무 배정 서비스
 * 순수 스코어링 기반 알고리즘
 */

import { PrismaClient } from '@prisma/client';

interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  skills?: any[];
  availableHours?: number;
  preferredTypes?: string[];
  experienceLevel?: string;
  lastAssignedAt?: Date;
}

interface Task {
  id: string;
  title: string;
  description?: string;
  complexity?: string;
  estimatedHours?: number;
  priority: string;
  requiredSkills?: any[];
  taskType?: string;
  dueDate?: Date;
}

interface AssignmentResult {
  userId: string;
  score: number;
  reason: string;
  scoreBreakdown: {
    skillMatch: number;
    workload: number;
    experience: number;
    priority: number;
  };
  alternatives: Array<{
    userId: string;
    score: number;
  }>;
}

export class SmartTaskAssigner {
  private prisma: PrismaClient;

  constructor(prisma: PrismaClient) {
    this.prisma = prisma;
  }

  /**
   * 메인 배정 함수
   */
  async assignBestUser(task: Task, tenantId: string): Promise<AssignmentResult | null> {
    // 가용한 사용자 조회
    const availableUsers = await this.getAvailableUsers(tenantId);
    
    if (availableUsers.length === 0) {
      return null;
    }

    let bestUser: User | null = null;
    let highestScore = -1;
    const allScores: Array<{ user: User; score: number; breakdown: any }> = [];

    for (const user of availableUsers) {
      const skillScore = await this.calculateSkillMatchScore(task, user);
      const workloadScore = await this.calculateWorkloadScore(user);
      const experienceScore = await this.calculateExperienceScore(task, user);
      const priorityScore = this.calculatePriorityScore(task, user);

      const totalScore = (skillScore * 0.4) + 
                        (workloadScore * 0.3) + 
                        (experienceScore * 0.2) + 
                        (priorityScore * 0.1);

      const breakdown = {
        skillMatch: skillScore,
        workload: workloadScore,
        experience: experienceScore,
        priority: priorityScore
      };

      allScores.push({ user, score: totalScore, breakdown });

      if (totalScore > highestScore) {
        highestScore = totalScore;
        bestUser = user;
      }
    }

    if (!bestUser) {
      return null;
    }

    // 검증 로직
    const validatedUser = await this.validateAndAssign(task, bestUser);
    if (!validatedUser) {
      // 대체 사용자 찾기
      const alternatives = allScores
        .filter(s => s.user.id !== bestUser!.id)
        .sort((a, b) => b.score - a.score)
        .slice(0, 3);

      if (alternatives.length > 0) {
        bestUser = alternatives[0]!.user;
        highestScore = alternatives[0]!.score;
      }
    }

    // 결과 생성
    const bestUserScore = allScores.find(s => s.user.id === bestUser!.id)!;
    const alternatives = allScores
      .filter(s => s.user.id !== bestUser!.id)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map(s => ({ userId: s.user.id, score: s.score }));

    return {
      userId: bestUser.id,
      score: highestScore,
      reason: this.generateAssignmentReason(bestUserScore.breakdown, bestUser),
      scoreBreakdown: bestUserScore.breakdown,
      alternatives
    };
  }

  /**
   * 기술 매칭 스코어 계산 (40% 가중치)
   */
  private async calculateSkillMatchScore(task: Task, user: User): Promise<number> {
    const requiredSkills = this.extractSkillsFromTask(task);
    const userSkills = user.skills || [];

    if (requiredSkills.length === 0) {
      return 60; // 기본 점수
    }

    const matchedSkills = requiredSkills.filter(skill => 
      userSkills.some((userSkill: any) => 
        userSkill.name?.toLowerCase().includes(skill.toLowerCase())
      )
    );

    let score = (matchedSkills.length / requiredSkills.length) * 100;

    // 핵심 기술 보너스
    const coreSkillMatch = this.checkCoreSkillMatch(task, userSkills);
    if (coreSkillMatch) score += 20;

    // 경험 년수 고려
    const avgExperience = this.getAverageExperience(userSkills);
    if (avgExperience >= 3) score += 10;

    return Math.min(score, 100);
  }

  /**
   * 워크로드 스코어 계산 (30% 가중치)
   */
  private async calculateWorkloadScore(user: User): Promise<number> {
    const currentTasks = await this.prisma.task.findMany({
      where: {
        assigneeId: user.id,
        status: { not: 'DONE' }
      },
      include: {
        metadata: {
          select: {
            estimatedHours: true
          }
        }
      }
    });

    const taskCount = currentTasks.length;
    const totalHours = currentTasks.reduce((sum, task) => sum + (task.metadata?.estimatedHours || 0), 0);

    let score = 100 - (taskCount * 10) - (totalHours * 2);

    // 가용 시간 고려
    if (user.availableHours) {
      const utilizationRate = totalHours / user.availableHours;
      if (utilizationRate > 0.8) score -= 20;
    }

    return Math.max(score, 0);
  }

  /**
   * 과거 성과 스코어 계산 (20% 가중치)
   */
  private async calculateExperienceScore(task: Task, user: User): Promise<number> {
    const completedTasks = await this.prisma.task.findMany({
      where: {
        assigneeId: user.id,
        status: 'DONE'
      },
      include: {
        metadata: {
          select: {
            taskType: true,
            requiredSkills: true
          }
        }
      }
    });

    const similarTasks = this.findSimilarTasks(task, completedTasks);
    const totalAssigned = await this.prisma.task.count({
      where: { assigneeId: user.id }
    });

    const similarCount = similarTasks.length;
    const totalCompleted = completedTasks.length;
    const completionRate = totalAssigned > 0 ? totalCompleted / totalAssigned : 0;

    let score = (similarCount * 20) + (completionRate * 50) + (totalCompleted * 0.5);

    return Math.min(score, 100);
  }

  /**
   * 우선순위 최적화 스코어 계산 (10% 가중치)
   */
  private calculatePriorityScore(task: Task, user: User): number {
    let score = 60; // 기본 점수

    // 우선순위 & 역할 매칭
    if (task.priority === 'HIGH' && user.role === 'OWNER') score = 100;
    else if (task.priority === 'HIGH' && user.role === 'ADMIN') score = 80;
    else if (task.priority === 'MEDIUM') score = 60;
    else if (task.priority === 'LOW') score = 40;

    // 복잡도 & 경험 레벨 매칭
    if (task.complexity === 'high' && user.experienceLevel === 'senior') score += 20;
    else if (task.complexity === 'medium' && user.experienceLevel === 'mid') score += 10;
    else if (task.complexity === 'low') score += 5;

    return Math.min(score, 100);
  }

  /**
   * 배정 결과 검증
   */
  private async validateAndAssign(task: Task, user: User): Promise<User | null> {
    // 권한 검증
    if (!this.validateUserPermissions(user, task)) {
      return null;
    }

    // 워크로드 검증
    if (!(await this.validateWorkloadLimit(user))) {
      return null;
    }

    // 기술 검증
    if (!this.validateMinimumSkills(user, task)) {
      return null;
    }

    return user;
  }

  /**
   * 배정 로그 저장
   */
  async logAssignment(result: AssignmentResult, taskId: string): Promise<void> {
    await this.prisma.taskAssignmentLog.create({
      data: {
        taskId,
        userId: result.userId,
        assignmentScore: result.score,
        scoreBreakdown: result.scoreBreakdown,
        reason: result.reason,
        alternatives: result.alternatives,
        algorithmVersion: '1.0'
      }
    });
  }

  // === Helper Methods ===

  private async getAvailableUsers(tenantId: string): Promise<User[]> {
    return await this.prisma.user.findMany({
      where: { tenantId },
      select: {
        id: true,
        name: true,
        email: true,
        role: true,
        skills: true,
        availableHours: true,
        preferredTypes: true,
        experienceLevel: true,
        lastAssignedAt: true
      }
    }) as User[];
  }

  private extractSkillsFromTask(task: Task): string[] {
    const skills: string[] = [];
    
    // 제목과 설명에서 기술 키워드 추출
    const text = `${task.title} ${task.description || ''}`.toLowerCase();
    
    const techKeywords = [
      'javascript', 'typescript', 'react', 'node.js', 'python', 'java',
      'html', 'css', 'sql', 'mongodb', 'postgresql', 'api', 'rest',
      'graphql', 'docker', 'kubernetes', 'aws', 'azure', 'figma',
      'photoshop', 'ui', 'ux', 'design', 'frontend', 'backend',
      'fullstack', 'mobile', 'android', 'ios', 'flutter', 'react native'
    ];

    techKeywords.forEach(keyword => {
      if (text.includes(keyword)) {
        skills.push(keyword);
      }
    });

    // requiredSkills가 있으면 추가
    if (task.requiredSkills) {
      skills.push(...task.requiredSkills);
    }

    return [...new Set(skills)]; // 중복 제거
  }

  private checkCoreSkillMatch(task: Task, userSkills: any[]): boolean {
    const coreSkills = this.extractSkillsFromTask(task).slice(0, 2); // 상위 2개 기술
    return coreSkills.some(skill => 
      userSkills.some((userSkill: any) => 
        userSkill.name?.toLowerCase().includes(skill.toLowerCase()) && 
        userSkill.level === 'senior'
      )
    );
  }

  private getAverageExperience(userSkills: any[]): number {
    if (!userSkills.length) return 0;
    const totalYears = userSkills.reduce((sum, skill) => sum + (skill.years || 0), 0);
    return totalYears / userSkills.length;
  }

  private findSimilarTasks(task: Task, completedTasks: any[]): any[] {
    return completedTasks.filter(completed => {
      // 업무 유형 매칭
      if (task.taskType && completed.metadata?.taskType === task.taskType) return true;
      
      // 복잡도 매칭
      if (task.complexity && completed.complexity === task.complexity) return true;
      
      // 제목 유사도 (간단한 키워드 매칭)
      const taskWords = task.title.toLowerCase().split(' ');
      const completedWords = completed.title.toLowerCase().split(' ');
      const intersection = taskWords.filter(word => completedWords.includes(word));
      
      return intersection.length >= 2;
    });
  }

  private validateUserPermissions(user: User, task: Task): boolean {
    // 기본적으로 모든 사용자가 업무를 수행할 수 있음
    return true;
  }

  private async validateWorkloadLimit(user: User): Promise<boolean> {
    const currentTasks = await this.prisma.task.count({
      where: {
        assigneeId: user.id,
        status: { not: 'DONE' }
      }
    });

    // 최대 10개 업무까지 할당 가능
    return currentTasks < 10;
  }

  private validateMinimumSkills(user: User, task: Task): boolean {
    const requiredSkills = this.extractSkillsFromTask(task);
    if (requiredSkills.length === 0) return true;

    const userSkills = user.skills || [];
    const matchCount = requiredSkills.filter(skill => 
      userSkills.some((userSkill: any) => 
        userSkill.name?.toLowerCase().includes(skill.toLowerCase())
      )
    ).length;

    // 최소 30% 기술 매칭 필요
    return matchCount / requiredSkills.length >= 0.3;
  }

  private generateAssignmentReason(breakdown: any, user: User): string {
    const reasons = [];

    if (breakdown.skillMatch > 80) {
      reasons.push('높은 기술 매칭');
    }
    if (breakdown.workload > 70) {
      reasons.push('적절한 워크로드');
    }
    if (breakdown.experience > 60) {
      reasons.push('관련 업무 경험 보유');
    }
    if (breakdown.priority > 80) {
      reasons.push('역할 및 우선순위 적합');
    }

    const mainReason = reasons.length > 0 ? reasons.join(', ') : '종합적 판단';
    return `${user.name}: ${mainReason} (총 점수: ${Math.round((breakdown.skillMatch * 0.4) + (breakdown.workload * 0.3) + (breakdown.experience * 0.2) + (breakdown.priority * 0.1))})`;
  }
}