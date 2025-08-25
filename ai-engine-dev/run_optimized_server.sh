#!/bin/bash

# TtalKkak AI 서버 VLLM + 최적화 실행 스크립트

echo "🚀 TtalKkak AI 서버 VLLM + 최적화 버전 시작..."

# 환경변수 설정
export PRELOAD_MODELS=true
export USE_VLLM=true  # VLLM 활성화
export HOST=0.0.0.0
export PORT=8000
export WORKERS=1

# GPU 메모리 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# VLLM 최적화 설정
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # Flash Attention 사용
export VLLM_USE_MODELSCOPE=false

echo "🔧 설정된 환경변수:"
echo "   - PRELOAD_MODELS=$PRELOAD_MODELS"
echo "   - USE_VLLM=$USE_VLLM"
echo "   - HOST=$HOST"
echo "   - PORT=$PORT"
echo "   - WORKERS=$WORKERS"
echo "   - PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "   - VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"

echo ""
echo "🚀 VLLM + 최적화 기능:"
echo "   ⚡ VLLM 초고속 추론 (3-10배 빠름)"
echo "   ✅ 모델 사전 로딩 (병렬 로딩)"
echo "   ✅ BERT 진짜 배치 처리 (32개씩)"
echo "   ✅ GPU 메모리 최적화 (Mixed Precision)"
echo "   ✅ 자동 메모리 정리"
echo "   ✅ PagedAttention (메모리 효율성)"
echo "   ✅ 동적 배치 처리"
echo "   ✅ Flash Attention 2.0"

echo ""
echo "📦 VLLM 설치 확인 중..."
if python -c "import vllm; print(f'VLLM version: {vllm.__version__}')" 2>/dev/null; then
    echo "✅ VLLM 설치 확인됨"
else
    echo "❌ VLLM 미설치 - 설치 중..."
    pip install vllm>=0.3.0
    if [ $? -eq 0 ]; then
        echo "✅ VLLM 설치 완료"
    else
        echo "⚠️ VLLM 설치 실패 - Transformers 모드로 실행"
        export USE_VLLM=false
    fi
fi

echo ""
echo "🎉 서버 시작 중..."

# Python 서버 실행
python ai_server_final_with_triplets.py