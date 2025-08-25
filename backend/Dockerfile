# Node.js 백엔드 Dockerfile
FROM node:18-alpine AS base

# 필수 시스템 패키지 설치
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    cairo-dev \
    jpeg-dev \
    pango-dev \
    giflib-dev

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY package*.json ./
COPY prisma ./prisma/

# 의존성 설치
RUN npm ci --only=production && npm cache clean --force

# 소스 코드 복사
COPY . .

# Prisma 클라이언트 생성
RUN npx prisma generate

# TypeScript 빌드
RUN npm run build

# 프로덕션 이미지
FROM node:18-alpine AS production

# 필수 런타임 패키지 설치
RUN apk add --no-cache dumb-init

# 비특권 사용자 생성
RUN addgroup -g 1001 -S nodejs
RUN adduser -S ddalkkak -u 1001

# 작업 디렉토리 설정
WORKDIR /app

# 소유권 변경
USER ddalkkak

# 빌드된 애플리케이션 복사
COPY --from=base --chown=ddalkkak:nodejs /app/dist ./dist
COPY --from=base --chown=ddalkkak:nodejs /app/node_modules ./node_modules
COPY --from=base --chown=ddalkkak:nodejs /app/package.json ./package.json
COPY --from=base --chown=ddalkkak:nodejs /app/prisma ./prisma

# 포트 노출
EXPOSE 3000 8080

# 헬스체크
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node healthcheck.js

# 애플리케이션 실행
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/index.js"]