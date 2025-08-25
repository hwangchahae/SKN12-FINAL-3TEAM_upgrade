# Similarity 프로젝트 설치 및 실행 가이드

## 📋 사전 요구사항

- Python 3.8 이상
- CUDA 11.8 이상 (GPU 사용 시)
- 최소 16GB RAM (권장 32GB)
- GPU VRAM 8GB 이상 (vLLM 사용 시)

## 🚀 빠른 시작

### 1. 가상환경 생성 및 활성화

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. 의존성 패키지 설치

```bash
# 기본 설치
pip install -r requirements.txt

# GPU 사용 시 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# vLLM 설치 (선택사항, GPU 필수)
pip install vllm
```

### 3. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 추가:

```env
# Hugging Face 토큰 (모델 다운로드용)
HUGGINGFACE_TOKEN=your_huggingface_token_here

# OpenAI API Key (유사도 평가용)
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. 설정 파일 확인

`config.yaml` 파일에서 경로 설정:

```yaml
# 자신의 환경에 맞게 경로 수정
base_model:
  input_dir: "../Raw_Data_val"  # 입력 데이터 경로
  output_dir: "../Pre_Training/4B_base_model_results"  # 출력 경로
```

## 💻 실행 방법

### 회의록 처리

#### Base Model 실행:
```bash
cd Similarity
python base_model_meeting_processor.py
```

#### LoRA Model 실행:
```bash
cd Similarity
python lora_model_meeting_processor.py
```

### 유사도 평가

#### Pre-training 평가:
```bash
cd Similarity/pre_similarity
python pre_SimilarityEvaluator.py --model 4B
```

#### Post-training 평가:
```bash
cd Similarity/post_similarity
python post_SimilarityEvaluator.py --model 8B
```

## 🔧 트러블슈팅

### 1. CUDA/GPU 관련 오류

```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 지원 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 메모리 부족 오류

config.yaml에서 다음 값 조정:
```yaml
base_model:
  gpu_memory_utilization: 0.5  # 0.7 → 0.5로 감소
  chunk_size: 3000  # 5000 → 3000으로 감소
```

### 3. ImportError: prd_generation_prompts

ai-engine-dev 폴더가 없는 경우 자동으로 폴백 프롬프트 사용됨.
경고 메시지는 무시 가능.

### 4. OpenAI API 오류

```bash
# API Key 확인
echo $OPENAI_API_KEY

# .env 파일 확인
cat .env | grep OPENAI
```

### 5. vLLM 설치 실패

```bash
# 의존성 충돌 방지
pip install vllm --no-deps
pip install ray pandas pyarrow
```

## 📦 패키지별 용도

| 패키지 | 용도 | 필수/선택 |
|--------|------|----------|
| torch | 딥러닝 프레임워크 | 필수 |
| transformers | Hugging Face 모델 | 필수 |
| vllm | 고속 추론 엔진 | 권장 |
| peft | LoRA 어댑터 지원 | 필수 |
| scikit-learn | TF-IDF 유사도 | 필수 |
| openai | 임베딩 유사도 | 선택 |
| pyyaml | 설정 파일 파싱 | 필수 |

## 🐍 Python 버전별 호환성

- Python 3.8: ✅ 완전 지원
- Python 3.9: ✅ 완전 지원
- Python 3.10: ✅ 완전 지원
- Python 3.11: ⚠️ vLLM 호환성 확인 필요
- Python 3.12: ❌ 일부 패키지 미지원

## 📊 시스템 요구사항

### 최소 사양
- CPU: 4코어 이상
- RAM: 16GB
- Storage: 50GB 여유 공간
- GPU: GTX 1060 (6GB VRAM)

### 권장 사양
- CPU: 8코어 이상
- RAM: 32GB
- Storage: 100GB SSD
- GPU: RTX 3080 (10GB VRAM) 이상

## 🔗 관련 링크

- [Hugging Face Token 발급](https://huggingface.co/settings/tokens)
- [OpenAI API Key 발급](https://platform.openai.com/api-keys)
- [CUDA 설치 가이드](https://developer.nvidia.com/cuda-downloads)
- [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)

## 📝 추가 참고사항

1. **모델 다운로드**: 첫 실행 시 모델 다운로드로 시간이 걸릴 수 있음 (4B 모델 약 8GB)
2. **캐시 관리**: `~/.cache/huggingface/` 디렉토리에 모델 캐시 저장됨
3. **로그 확인**: 실행 중 문제 발생 시 터미널 로그 확인
4. **설정 백업**: config.yaml 수정 전 백업 권장

## 도움이 필요하신가요?

문제가 지속되면 다음 정보와 함께 이슈를 제기해주세요:
- Python 버전: `python --version`
- 설치된 패키지: `pip list`
- 오류 메시지 전체
- 시스템 사양 (OS, GPU 등)