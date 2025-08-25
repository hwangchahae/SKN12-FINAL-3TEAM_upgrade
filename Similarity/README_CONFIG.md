# 회의록 처리 및 유사도 평가 설정 가이드

## 개요
이 문서는 다음 스크립트들의 설정 방법을 설명합니다:
- 회의록 처리 모델: `base_model_meeting_processor.py`, `lora_model_meeting_processor.py`
- 유사도 평가: `pre_similarity/pre_SimilarityEvaluator.py`, `post_similarity/post_SimilarityEvaluator.py`

팀원들이 코드를 수정하지 않고 설정 파일(`config.yaml`)만 수정하여 다양한 경로와 파라미터로 실행할 수 있습니다.

## 설정 파일 위치
- 설정 파일: `Similarity/config.yaml`
- Base Model 스크립트: `Similarity/base_model_meeting_processor.py`
- LoRA Model 스크립트: `Similarity/lora_model_meeting_processor.py`
- Pre-training 평가 스크립트: `Similarity/pre_similarity/pre_SimilarityEvaluator.py`
- Post-training 평가 스크립트: `Similarity/post_similarity/post_SimilarityEvaluator.py`

## 설정 파일 구조

### 1. Base Model 설정
```yaml
base_model:
  # 사용할 모델 경로
  model_path: "Qwen/Qwen3-4B-AWQ"
  model_name: "Qwen3_4B"  # 출력 폴더명에 사용될 이름
  
  # 입출력 경로 (여기를 수정하세요!)
  input_dir: "../Raw_Data_val"  # 입력 데이터 디렉토리
  output_dir: "../Pre_Training/4B_base_model_results"  # 출력 디렉토리
  
  # 청킹 설정
  chunk_size: 5000  # 텍스트 청킹 크기
  chunk_overlap: 512  # 청크 간 오버랩 크기
  
  # 모델 파라미터
  temperature: 0.2
  max_tokens: 2048
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.7
  max_model_len: 16384
```

### 2. LoRA Model 설정
```yaml
lora_model:
  # 모델 경로
  base_model_path: "Qwen/Qwen3-8B"
  lora_model_path: "qwen3_lora_ttalkkac_8b"
  merged_model_path: "8B_merged_qwen3_lora_model"
  
  # 입출력 경로 (여기를 수정하세요!)
  input_dir: "../Raw_Data_val"  # 입력 데이터 디렉토리
  output_dir: "8B_lora_model_results"  # 출력 디렉토리
  
  # 청킹 설정
  chunk_size: 5000
  chunk_overlap: 512
  
  # 모델 파라미터
  temperature: 0.3
  top_p: 0.9
  repetition_penalty: 1.1
  max_new_tokens: 2048
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
  max_model_len: 8192
  
  # 테스트 설정
  test_file_limit: 0  # 0이면 전체 파일 처리, 양수면 해당 개수만 처리
```

### 3. 유사도 평가 설정
```yaml
similarity_evaluation:
  # Pre-training (학습 전) 평가
  pre_training:
    gold_data_path: "../Gold_Standard_Ttalkkac"  # 정답 데이터 경로
    results:
      "1.7B": "../Pre_Training/1.7B_base_model_results"
      "4B": "../Pre_Training/4B_base_model_results"
      "8B": "../Pre_Training/8B_base_model_results"
    output_dir: "./pre_similarity_results"
    output_prefix: "pretrain"
  
  # Post-training (학습 후) 평가
  post_training:
    gold_data_path: "../Gold_Standard_Ttalkkac"
    results:
      "1.7B": "../Post_Training/1.7B_lora_model_results"
      "4B": "../Post_Training/4B_lora_model_results"
      "8B": "../Post_Training/8B_lora_model_results"
    output_dir: "./post_similarity_results"
    output_prefix: "post"
  
  # 평가 파라미터
  evaluation_params:
    sample_size: 100  # 랜덤 샘플링 크기
    random_seed: 42
    tfidf_max_features: 5000
    tfidf_ngram_range: [1, 2]
    embedding_model: "text-embedding-3-large"
    batch_size: 10
    save_interval: 20
    use_cache: true
```

## 사용 방법

### 1. 경로 설정 변경
팀원이 다른 경로에서 실행하려면 `config.yaml` 파일에서 다음 항목만 수정하면 됩니다:

#### Base Model 경로 변경 예시:
```yaml
base_model:
  input_dir: "C:/MyData/meeting_data"  # 절대 경로도 가능
  output_dir: "C:/MyResults/base_model_output"
```

#### LoRA Model 경로 변경 예시:
```yaml
lora_model:
  input_dir: "../../data/meetings"  # 상대 경로도 가능
  output_dir: "./results/lora_output"
```

#### 유사도 평가 경로 변경 예시:
```yaml
similarity_evaluation:
  pre_training:
    gold_data_path: "C:/MyData/gold_standard"  # 정답 데이터 경로
    results:
      "4B": "C:/MyResults/4B_results"  # 평가할 모델 결과 경로
    output_dir: "./my_evaluation_results"  # 평가 결과 저장 경로
```

### 2. 모델 실행

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

#### Pre-training 유사도 평가 실행:
```bash
cd Similarity/pre_similarity
python pre_SimilarityEvaluator.py --model 4B
# 또는 모든 모델 평가
python pre_SimilarityEvaluator.py --model 1.7B
python pre_SimilarityEvaluator.py --model 4B
python pre_SimilarityEvaluator.py --model 8B
```

#### Post-training 유사도 평가 실행:
```bash
cd Similarity/post_similarity
python post_SimilarityEvaluator.py --model 8B
# 또는 모든 모델 평가
python post_SimilarityEvaluator.py --model 1.7B
python post_SimilarityEvaluator.py --model 4B
python post_SimilarityEvaluator.py --model 8B
```

### 3. 테스트 모드 사용
전체 파일이 아닌 일부만 처리하고 싶을 때:

#### 회의록 처리:
```yaml
lora_model:
  test_file_limit: 5  # 5개 파일만 처리
```

#### 유사도 평가:
```yaml
similarity_evaluation:
  evaluation_params:
    sample_size: 10  # 10개 파일만 샘플링하여 평가
```

## 주의사항

1. **경로 구분자**: Windows에서는 `/` 또는 `\` 모두 사용 가능합니다.
2. **상대 경로**: 스크립트 파일 위치 기준입니다.
3. **절대 경로**: 전체 경로를 지정할 수도 있습니다.
4. **출력 디렉토리**: 자동으로 생성되므로 미리 만들 필요 없습니다.
5. **OpenAI API Key**: 유사도 평가를 위해서는 환경변수 `OPENAI_API_KEY` 설정이 필요합니다.

## 입력 데이터 구조
입력 디렉토리는 다음과 같은 구조여야 합니다:
```
input_dir/
├── folder1/
│   └── 05_final_result.json
├── folder2/
│   └── 05_final_result.json
└── folder3/
    └── 05_final_result.json
```

## 출력 데이터 구조
처리 결과는 다음과 같이 저장됩니다:
```
output_dir/
├── folder1/
│   └── result.json
├── folder2_chunk_1/  # 긴 파일은 청킹됨
│   └── result.json
└── folder2_chunk_2/
    └── result.json
```

## 문제 해결

### 설정 파일을 찾을 수 없을 때
- 스크립트와 같은 디렉토리에 `config.yaml` 파일이 있는지 확인
- 파일명이 정확한지 확인 (config.yml이 아닌 config.yaml)

### 입력 디렉토리를 찾을 수 없을 때
- `config.yaml`의 `input_dir` 경로가 올바른지 확인
- 상대 경로 사용 시 스크립트 위치 기준인지 확인

### GPU 메모리 부족
- `gpu_memory_utilization` 값을 낮춤 (예: 0.7 → 0.5)
- `chunk_size`를 줄임 (예: 5000 → 3000)

## 팀원을 위한 Quick Start

### 회의록 처리:
1. `config.yaml` 파일 열기
2. 자신의 데이터 경로로 `input_dir` 수정
3. 원하는 결과 저장 경로로 `output_dir` 수정
4. 스크립트 실행:
   ```bash
   python base_model_meeting_processor.py  # 또는
   python lora_model_meeting_processor.py
   ```

### 유사도 평가:
1. `config.yaml` 파일 열기
2. `gold_data_path`를 정답 데이터 경로로 수정
3. `results`에서 평가할 모델 결과 경로 수정
4. 스크립트 실행:
   ```bash
   cd pre_similarity
   python pre_SimilarityEvaluator.py --model 4B
   # 또는
   cd post_similarity
   python post_SimilarityEvaluator.py --model 8B
   ```

설정 파일만 수정하면 코드 변경 없이 바로 실행 가능합니다!