/**
 * Notion 서비스 - 사용자별 OAuth 기반 연동
 */

import { Client } from '@notionhq/client';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

interface MeetingData {
  title: string;
  overview: string;
  objectives: string[];
  tasks: Array<{
    title: string;
    description: string;
    complexity: string;
    estimated_hours: number;
    priority: string;
  }>;
  attendees?: string[];
  date?: string;
}

export class NotionService {
  private notion: Client;
  
  constructor(accessToken: string) {
    this.notion = new Client({
      auth: this.decrypt(accessToken),
    });
  }
  
  // 암호화/복호화 (간단한 버전)
  private decrypt(encryptedText: string): string {
    try {
      return Buffer.from(encryptedText, 'base64').toString();
    } catch (error) {
      console.error('토큰 복호화 실패:', error);
      throw new Error('Invalid access token');
    }
  }
  
  // 사용자별 NotionService 인스턴스 생성
  static async createForUser(tenantId: string, userId: string): Promise<NotionService | null> {
    try {
      const integration = await prisma.integration.findUnique({
        where: {
          tenantId_userId_serviceType: {
            tenantId,
            userId,
            serviceType: 'NOTION'
          }
        }
      });
      
      if (!integration || !integration.isActive || !integration.accessToken) {
        console.log(`❌ Notion 연동 없음: tenantId=${tenantId}, userId=${userId}`);
        return null;
      }
      
      console.log(`✅ Notion 연동 확인됨: ${(integration.config as any)?.workspace_name || 'Unknown Workspace'}`);
      return new NotionService(integration.accessToken);
    } catch (error) {
      console.error('NotionService 생성 오류:', error);
      return null;
    }
  }
  
  // 회의록 페이지 생성 (사용자 템플릿 형식에 맞춤)
  async createMeetingPage(meetingData: MeetingData): Promise<{ id: string; url: string }> {
    try {
      console.log('📝 Notion 페이지 생성 시작:', meetingData.title);
      
      // 먼저 사용자의 워크스페이스에서 가능한 부모 페이지 찾기
      const search = await this.notion.search({
        query: '',
        filter: {
          value: 'page',
          property: 'object'
        },
        page_size: 1
      });
      
      // 기본 부모 (워크스페이스 루트)를 사용
      const parent = search.results.length > 0 && search.results[0]?.id ? 
        { page_id: search.results[0].id } : 
        { workspace: true as const };
      
      const currentDate = meetingData.date || new Date().toLocaleDateString('ko-KR');
      
      // 페이지 생성 (사용자 템플릿 구조에 맞춤)
      const response = await this.notion.pages.create({
        parent,
        icon: {
          emoji: '🔔'
        },
        properties: {
          title: {
            title: [{ 
              text: { 
                content: `딸깍 - ${meetingData.title}` 
              } 
            }]
          }
        },
        children: [
          // 프로젝트 설명
          {
            object: 'block',
            type: 'heading_3',
            heading_3: {
              rich_text: [{ 
                type: 'text', 
                text: { content: '🎈SKN 12기 Final Project 3팀' } 
              }]
            }
          },
          {
            object: 'block',
            type: 'paragraph',
            paragraph: {
              rich_text: [{ 
                type: 'text', 
                text: { content: meetingData.overview || '음성 입력을 통해 자동으로 기획안을 구조화하고, AI Agent 기반으로 체계적인 업무 분담 및 팀원 배정을 수행하는 통합 워크플로우 시스템' } 
              }]
            }
          },
          
          // 네비게이션 콜아웃
          {
            object: 'block',
            type: 'callout',
            callout: {
              rich_text: [{ 
                type: 'text', 
                text: { content: 'Home    |   Calendar   |  Work  ' } 
              }],
              color: 'gray_background' as const
            }
          },
          
          // Work 섹션
          {
            object: 'block',
            type: 'callout',
            callout: {
              rich_text: [{ 
                type: 'text', 
                text: { content: 'Work' },
                annotations: { bold: true }
              }],
              icon: {
                emoji: '✅'
              },
              color: 'brown_background' as const
            }
          },
          
          // 회의 테이블 헤더
          {
            object: 'block',
            type: 'heading_3',
            heading_3: {
              rich_text: [{ 
                type: 'text', 
                text: { content: '회의' } 
              }]
            }
          }
        ]
      });
      
      // 회의 테이블 생성 (별도 추가)
      const meetingTableBlocks = [
        {
          object: 'block' as const,
          type: 'table' as const,
          table: {
            table_width: 6,
            has_column_header: true,
            has_row_header: false,
            children: [
              // 헤더 행
              {
                object: 'block' as const,
                type: 'table_row' as const,
                table_row: {
                  cells: [
                    [{ type: 'text' as const, text: { content: 'ID' } }],
                    [{ type: 'text' as const, text: { content: '주차구분' } }],
                    [{ type: 'text' as const, text: { content: '날짜' } }],
                    [{ type: 'text' as const, text: { content: '속성' } }],
                    [{ type: 'text' as const, text: { content: '상태' } }],
                    [{ type: 'text' as const, text: { content: '텍스트' } }]
                  ]
                }
              },
              // 데이터 행
              {
                object: 'block' as const,
                type: 'table_row' as const,
                table_row: {
                  cells: [
                    [{ type: 'text' as const, text: { content: '1' } }],
                    [{ type: 'text' as const, text: { content: new Date().toLocaleDateString('ko-KR', { month: 'long' }) + ' 주차' } }],
                    [{ type: 'text' as const, text: { content: currentDate } }],
                    [{ type: 'text' as const, text: { content: meetingData.title } }],
                    [{ type: 'text' as const, text: { content: '완료' } }],
                    [{ type: 'text' as const, text: { content: meetingData.overview.substring(0, 50) + '...' } }]
                  ]
                }
              }
            ]
          }
        }
      ];
      
      // Calendar 섹션
      const calendarBlocks = [
        {
          object: 'block' as const,
          type: 'callout' as const,
          callout: {
            rich_text: [{ 
              type: 'text' as const, 
              text: { content: '✱ Calendar' },
              annotations: { bold: true }
            }]
          }
        },
        {
          object: 'block' as const,
          type: 'heading_3' as const,
          heading_3: {
            rich_text: [{ 
              type: 'text' as const, 
              text: { content: '회의' } 
            }]
          }
        },
        // 같은 테이블 복사 (Calendar 뷰)
        ...meetingTableBlocks
      ];
      
      // 기획안 섹션
      const planningBlocks = [
        {
          object: 'block' as const,
          type: 'callout' as const,
          callout: {
            rich_text: [{ 
              type: 'text' as const, 
              text: { content: '✱ 기획안' },
              annotations: { bold: true }
            }]
          }
        },
        
        // 프로젝트 개요
        {
          object: 'block' as const,
          type: 'heading_2' as const,
          heading_2: {
            rich_text: [{ 
              type: 'text' as const, 
              text: { content: '프로젝트 개요' } 
            }]
          }
        },
        {
          object: 'block' as const,
          type: 'paragraph' as const,
          paragraph: {
            rich_text: [
              { 
                type: 'text' as const, 
                text: { content: '프로젝트명' },
                annotations: { bold: true }
              },
              { 
                type: 'text' as const, 
                text: { content: `: ${meetingData.title}` }
              }
            ]
          }
        },
        {
          object: 'block' as const,
          type: 'paragraph' as const,
          paragraph: {
            rich_text: [
              { 
                type: 'text' as const, 
                text: { content: '목적' },
                annotations: { bold: true }
              },
              { 
                type: 'text' as const, 
                text: { content: `: ${meetingData.overview}` }
              }
            ]
          }
        },
        {
          object: 'block' as const,
          type: 'paragraph' as const,
          paragraph: {
            rich_text: [
              { 
                type: 'text' as const, 
                text: { content: '수행기간' },
                annotations: { bold: true }
              },
              { 
                type: 'text' as const, 
                text: { content: `: ${currentDate} ~ [종료일]` }
              }
            ]
          }
        },
        {
          object: 'block' as const,
          type: 'paragraph' as const,
          paragraph: {
            rich_text: [
              { 
                type: 'text' as const, 
                text: { content: '담당자' },
                annotations: { bold: true }
              },
              { 
                type: 'text' as const, 
                text: { content: ': [담당자명]' }
              }
            ]
          }
        },
        
        // 핵심 목표
        {
          object: 'block' as const,
          type: 'heading_2' as const,
          heading_2: {
            rich_text: [{ 
              type: 'text' as const, 
              text: { content: '핵심 목표' } 
            }]
          }
        },
        ...meetingData.objectives.map(objective => ({
          object: 'block' as const,
          type: 'bulleted_list_item' as const,
          bulleted_list_item: {
            rich_text: [{ 
              type: 'text' as const, 
              text: { content: objective } 
            }]
          }
        })),
        
        // 세부 내용
        {
          object: 'block' as const,
          type: 'heading_2' as const,
          heading_2: {
            rich_text: [{ 
              type: 'text' as const, 
              text: { content: '세부 내용' } 
            }]
          }
        },
        {
          object: 'block' as const,
          type: 'toggle' as const,
          toggle: {
            rich_text: [{ 
              type: 'text' as const, 
              text: { content: '핵심 아이디어' },
              annotations: { bold: true }
            }],
            children: [
              {
                object: 'block' as const,
                type: 'paragraph' as const,
                paragraph: {
                  rich_text: [{ 
                    type: 'text' as const, 
                    text: { content: meetingData.overview } 
                  }]
                }
              }
            ]
          }
        },
        {
          object: 'block' as const,
          type: 'toggle' as const,
          toggle: {
            rich_text: [{ 
              type: 'text' as const, 
              text: { content: '실행 계획' },
              annotations: { bold: true }
            }],
            children: meetingData.tasks.map(task => ({
              object: 'block' as const,
              type: 'paragraph' as const,
              paragraph: {
                rich_text: [{ 
                  type: 'text' as const, 
                  text: { content: `${task.title}: ${task.description}` } 
                }]
              }
            }))
          }
        },
        
        // 예산 섹션
        {
          object: 'block' as const,
          type: 'heading_2' as const,
          heading_2: {
            rich_text: [{ 
              type: 'text' as const, 
              text: { content: '예산' } 
            }]
          }
        },
        {
          object: 'block' as const,
          type: 'table' as const,
          table: {
            table_width: 3,
            has_column_header: true,
            has_row_header: false,
            children: [
              {
                object: 'block' as const,
                type: 'table_row' as const,
                table_row: {
                  cells: [
                    [{ type: 'text' as const, text: { content: '항목' }, annotations: { bold: true } }],
                    [{ type: 'text' as const, text: { content: '금액' }, annotations: { bold: true } }],
                    [{ type: 'text' as const, text: { content: '비고' }, annotations: { bold: true } }]
                  ]
                }
              },
              {
                object: 'block' as const,
                type: 'table_row' as const,
                table_row: {
                  cells: [
                    [{ type: 'text' as const, text: { content: '개발 비용' } }],
                    [{ type: 'text' as const, text: { content: '₩0,000,000' } }],
                    [{ type: 'text' as const, text: { content: '예상 개발 비용' } }]
                  ]
                }
              },
              {
                object: 'block' as const,
                type: 'table_row' as const,
                table_row: {
                  cells: [
                    [{ type: 'text' as const, text: { content: '총계' } }],
                    [{ type: 'text' as const, text: { content: '₩0,000,000' } }],
                    [{ type: 'text' as const, text: { content: '' } }]
                  ]
                }
              }
            ]
          }
        },
        
        // 첨부 자료
        {
          object: 'block' as const,
          type: 'heading_2' as const,
          heading_2: {
            rich_text: [{ 
              type: 'text' as const, 
              text: { content: '첨부 자료' } 
            }]
          }
        },
        {
          object: 'block' as const,
          type: 'bulleted_list_item' as const,
          bulleted_list_item: {
            rich_text: [{ 
              type: 'text' as const, 
              text: { content: '생성된 업무 목록' } 
            }]
          }
        },
        ...meetingData.tasks.map(task => ({
          object: 'block' as const,
          type: 'bulleted_list_item' as const,
          bulleted_list_item: {
            rich_text: [{ 
              type: 'text' as const, 
              text: { content: `${task.title} (${task.complexity}, ${task.estimated_hours}h, ${task.priority})` } 
            }]
          }
        }))
      ];
      
      // 모든 블록들을 페이지에 추가
      const allBlocks = [...meetingTableBlocks, ...calendarBlocks, ...planningBlocks];
      
      await this.notion.blocks.children.append({
        block_id: response.id,
        children: allBlocks
      });
      
      console.log('✅ Notion 페이지 생성 완료:', (response as any).url);
      
      return {
        id: response.id,
        url: (response as any).url || `https://notion.so/${response.id.replace(/-/g, '')}`
      };
      
    } catch (error) {
      console.error('❌ Notion 페이지 생성 실패:', error);
      throw new Error(`Notion 페이지 생성 실패: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
  
  // 연동 상태 테스트
  async testConnection(): Promise<boolean> {
    try {
      await this.notion.users.me({});
      return true;
    } catch (error) {
      console.error('Notion 연결 테스트 실패:', error);
      return false;
    }
  }
  
  // 사용자별 연동 상태 확인 (정적 메서드)
  static async checkUserIntegration(tenantId: string, userId: string): Promise<{
    connected: boolean;
    workspace_name?: string;
    error?: string | undefined;
  }> {
    try {
      const integration = await prisma.integration.findUnique({
        where: {
          tenantId_userId_serviceType: {
            tenantId,
            userId,
            serviceType: 'NOTION'
          }
        }
      });
      
      if (!integration || !integration.isActive) {
        return { connected: false };
      }
      
      // 연결 테스트
      const notionService = new NotionService(integration.accessToken!);
      const isWorking = await notionService.testConnection();
      
      return {
        connected: isWorking,
        workspace_name: (integration.config as any)?.workspace_name,
        ...(isWorking ? {} : { error: 'Connection test failed' })
      };
      
    } catch (error) {
      return {
        connected: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }
}