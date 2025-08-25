/**
 * 단순화된 Multi-tenant 미들웨어
 * 새로운 단순화된 스키마에 맞게 수정
 */

import { Request, Response, NextFunction } from 'express';
import { PrismaClient, Tenant, User } from '@prisma/client';
// import { logger } from '../utils/logger';  // 임시 비활성화

const logger = {
  warn: (msg: string) => console.warn(msg),
  info: (msg: string) => console.info(msg),
  debug: (msg: string) => console.debug(msg),
  error: (msg: string, error?: any) => console.error(msg, error)
};

// Express Request 확장
declare global {
  namespace Express {
    interface Request {
      tenant?: Tenant;
      user?: User;
      tenantId?: string;
    }
  }
}

export class SimpleTenantMiddleware {
  private prisma: PrismaClient;

  constructor(prisma: PrismaClient) {
    this.prisma = prisma;
  }

  /**
   * 요청 헤더에서 테넌트 정보 추출 및 설정
   */
  extractTenant = async (
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> => {
    try {
      // 헤더에서 테넌트 slug 또는 ID 추출
      const tenantSlug = req.headers['x-tenant-slug'] as string ||
                        req.headers['x-tenant-id'] as string ||
                        req.query.tenant as string ||
                        req.body?.tenantSlug;

      if (!tenantSlug) {
        logger.warn('No tenant identifier found in request');
        res.status(400).json({ error: 'Missing tenant identifier' });
        return;
      }

      // 테넌트 조회 (slug 또는 ID로)
      const tenant = await this.prisma.tenant.findFirst({
        where: {
          OR: [
            { slug: tenantSlug },
            { id: tenantSlug }
          ]
        }
      });

      if (!tenant) {
        logger.warn(`Tenant not found: ${tenantSlug}`);
        res.status(404).json({ error: 'Tenant not found' });
        return;
      }

      // Request에 테넌트 정보 설정
      req.tenant = tenant;
      req.tenantId = tenant.id;

      logger.debug(`Tenant set: ${tenant.name} (${tenant.slug})`);
      next();
    } catch (error) {
      logger.error('Error in tenant extraction:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  };

  /**
   * 사용자 인증 및 테넌트 멤버십 확인
   */
  authenticateUser = async (
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> => {
    try {
      const tenantId = req.tenantId;
      if (!tenantId) {
        res.status(400).json({ error: 'Tenant not set' });
        return;
      }

      // 사용자 정보 추출 (JWT, 세션, API 키 등)
      const userEmail = req.headers['x-user-email'] as string ||
                       req.body?.userEmail;

      if (!userEmail) {
        res.status(401).json({ error: 'User authentication required' });
        return;
      }

      // 테넌트 내 사용자 조회
      const user = await this.prisma.user.findUnique({
        where: {
          tenantId_email: {
            tenantId: tenantId,
            email: userEmail
          }
        }
      });

      if (!user) {
        logger.warn(`User not found in tenant: ${userEmail}`);
        res.status(403).json({ error: 'User not authorized for this tenant' });
        return;
      }

      // Request에 사용자 정보 설정
      req.user = user;

      logger.debug(`User authenticated: ${user.email} in tenant ${tenantId}`);
      next();
    } catch (error) {
      logger.error('Error in user authentication:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  };

  /**
   * 테넌트별 데이터 필터링을 위한 간단한 헬퍼 메서드
   */
  getPrismaForTenant(tenantId: string) {
    // 복잡한 확장 기능 대신 일반 prisma 클라이언트 반환
    // 사용할 때 직접 tenantId를 where 조건에 추가
    return this.prisma;
  }

  /**
   * 개발용 테넌트 자동 생성 미들웨어
   */
  createDevTenant = async (
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> => {
    try {
      const devTenantSlug = 'dev-tenant';
      
      let tenant = await this.prisma.tenant.findUnique({
        where: { slug: devTenantSlug }
      });

      if (!tenant) {
        tenant = await this.prisma.tenant.create({
          data: {
            name: 'Development Tenant',
            slug: devTenantSlug
          }
        });
        logger.info(`Created development tenant: ${tenant.id}`);
      }

      req.tenant = tenant;
      req.tenantId = tenant.id;
      next();
    } catch (error) {
      logger.error('Error creating dev tenant:', error);
      res.status(500).json({ error: 'Failed to setup dev tenant' });
    }
  };
}