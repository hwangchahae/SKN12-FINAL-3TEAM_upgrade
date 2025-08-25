# DdalKkak 프로젝트 Makefile

.PHONY: help install dev build test clean docker-build docker-up docker-down db-migrate db-seed

# 기본 타겟
help:
	@echo "DdalKkak 프로젝트 명령어 목록:"
	@echo ""
	@echo "🚀 개발 환경:"
	@echo "  install       - 모든 의존성 설치"
	@echo "  dev           - 개발 서버 시작"
	@echo "  build         - 프로젝트 빌드"
	@echo "  test          - 테스트 실행"
	@echo "  lint          - 코드 린팅"
	@echo "  format        - 코드 포맷팅"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  docker-build  - Docker 이미지 빌드"
	@echo "  docker-up     - Docker 컨테이너 시작"
	@echo "  docker-down   - Docker 컨테이너 종료"
	@echo "  docker-logs   - Docker 로그 확인"
	@echo ""
	@echo "🗄️ 데이터베이스:"
	@echo "  db-migrate    - 데이터베이스 마이그레이션"
	@echo "  db-seed       - 시드 데이터 삽입"
	@echo "  db-reset      - 데이터베이스 리셋"
	@echo ""
	@echo "🧹 유틸리티:"
	@echo "  clean         - 임시 파일 정리"
	@echo "  setup         - 초기 환경 설정"

# 의존성 설치
install:
	@echo "📦 의존성 설치 중..."
	npm install
	cd backend && npm install
	cd frontend/slack-app && npm install
	cd frontend/web-dashboard && npm install
	cd frontend/chrome-extension && npm install
	cd ai-engine && pip install -r requirements.txt

# 개발 서버 시작
dev:
	@echo "🚀 개발 서버 시작..."
	npm run dev

# 프로젝트 빌드
build:
	@echo "🔨 프로젝트 빌드 중..."
	npm run build

# 테스트 실행
test:
	@echo "🧪 테스트 실행 중..."
	npm run test

# 코드 린팅
lint:
	@echo "🔍 코드 린팅 중..."
	npm run lint

# 코드 포맷팅
format:
	@echo "✨ 코드 포맷팅 중..."
	npm run format

# Docker 이미지 빌드
docker-build:
	@echo "🐳 Docker 이미지 빌드 중..."
	docker-compose build

# Docker 컨테이너 시작
docker-up:
	@echo "🚀 Docker 컨테이너 시작..."
	docker-compose up -d

# Docker 컨테이너 종료
docker-down:
	@echo "🛑 Docker 컨테이너 종료..."
	docker-compose down

# Docker 로그 확인
docker-logs:
	@echo "📋 Docker 로그 확인..."
	docker-compose logs -f

# 데이터베이스 마이그레이션
db-migrate:
	@echo "🗄️ 데이터베이스 마이그레이션 중..."
	cd backend && npm run db:migrate

# 시드 데이터 삽입
db-seed:
	@echo "🌱 시드 데이터 삽입 중..."
	cd backend && npm run db:seed

# 데이터베이스 리셋
db-reset:
	@echo "🔄 데이터베이스 리셋 중..."
	cd backend && npm run db:reset

# 임시 파일 정리
clean:
	@echo "🧹 임시 파일 정리 중..."
	npm run clean
	docker system prune -f

# 초기 환경 설정
setup:
	@echo "⚙️ 초기 환경 설정 중..."
	@echo "1. 환경 변수 파일 복사..."
	cp .env.example .env
	@echo "2. 의존성 설치..."
	$(MAKE) install
	@echo "3. Docker 컨테이너 시작..."
	$(MAKE) docker-up
	@echo "4. 데이터베이스 마이그레이션..."
	sleep 10
	$(MAKE) db-migrate
	@echo ""
	@echo "✅ 초기 설정 완료!"
	@echo "📝 .env 파일을 편집하여 필요한 설정을 완료하세요."
	@echo "🚀 'make dev' 명령어로 개발을 시작할 수 있습니다."

# 빠른 재시작
restart:
	@echo "🔄 서비스 재시작 중..."
	$(MAKE) docker-down
	$(MAKE) docker-up

# 프로덕션 배포
deploy-dev:
	@echo "🚀 개발 환경 배포..."
	$(MAKE) build
	$(MAKE) docker-build
	docker-compose -f docker-compose.yml up -d

deploy-prod:
	@echo "🚀 프로덕션 환경 배포..."
	$(MAKE) build
	$(MAKE) docker-build
	docker-compose -f docker-compose.prod.yml up -d