import json
import os
import sys
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    __version__ as transformers_version
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging
import gc

# 런팟 환경을 위한 경로 설정
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from meeting_analysis_prompts import generate_meeting_analysis_system_prompt, generate_meeting_analysis_user_prompt

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CUDA 메모리 설정
if torch.cuda.is_available():
    # 메모리 단편화 방지
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
    # GPU 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()

# Flash Attention 가용성 확인
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
    logger.info("Flash Attention 사용 가능")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logger.warning("Flash Attention 사용 불가, 기본 attention 메커니즘 사용")

# packaging 모듈 확인
try:
    from packaging import version
except ImportError:
    logger.error("packaging 모듈이 없습니다. 설치가 필요합니다: pip install packaging")
    sys.exit(1)

def cleanup_gpu_memory():
    """GPU 메모리 정리 및 모니터링"""
    if torch.cuda.is_available():
        # 메모리 사용량 확인
        before_memory = torch.cuda.memory_allocated() / 1024**3
        
        # 캐시 정리
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Python 가비지 컬렉션
        gc.collect()
        
        # 정리 후 메모리 확인
        after_memory = torch.cuda.memory_allocated() / 1024**3
        freed_memory = before_memory - after_memory
        
        logger.info(f"GPU 메모리 정리 완료 (해제: {freed_memory:.2f}GB, 현재: {after_memory:.2f}GB)")

class TtalkkacDatasetConverter:
    """실제 프로젝트에서 사용하는 프롬프트와 데이터 형식으로 골드 스탠다드를 변환"""
    
    def __init__(self):
        # 골드 스탠다드와 동일한 프롬프트 사용
        self.system_prompt = generate_meeting_analysis_system_prompt()
        # user_prompt_template은 동적으로 생성하므로 메서드에서 처리

    def chunk_text(self, text: str, chunk_size: int = 5000, overlap: int = 512) -> List[str]:
        """텍스트를 청킹하여 나누기 (골드 스탠다드와 동일한 방식)"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunk = text[start:]
            else:
                chunk = text[start:end]
                
                # 마지막 완전한 문장에서 끊기 시도
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            
            if end >= len(text):
                break
                
            start = end - overlap
        
        return chunks

    def load_gold_standard_data(self, results_dir: str = None) -> List[Dict[str, Any]]:
        """골드 스탠다드 결과 폴더에서 데이터 로드"""
        if results_dir is None:
            results_dir = os.path.join(PROJECT_ROOT, "ttalkkac_gold_standard_results_output")
        results_path = Path(results_dir)
        data = []
        
        if not results_path.exists():
            logger.error(f"결과 디렉토리를 찾을 수 없습니다: {results_dir}")
            return []
        
        # 성공적으로 생성된 result.json 파일들 스캔
        for folder in results_path.iterdir():
            if folder.is_dir() and folder.name.startswith(('train_', 'val_')):
                result_file = folder / "result.json"
                if result_file.exists():
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            item = json.load(f)
                        data.append(item)
                        logger.info(f"로드: {folder.name}")
                    except Exception as e:
                        logger.error(f"{folder.name} 로드 실패: {e}")
        
        logger.info(f"총 {len(data)}개 골드 스탠다드 데이터 로드 완료")
        return data

    def load_meeting_content(self, source_dir: str, meeting_folder: str) -> str:
        """원본 회의 내용 로드 (batch_triplet_results에서)"""
        # OS 독립적 경로 처리                                                                                │ │
        source_path = Path(source_dir)  

        # result_ 접두사가 이미 있는지 확인
        if meeting_folder.startswith("result_"):
            meeting_path = source_path / meeting_folder / "05_final_result.json"
        else:
            meeting_path = source_path / f"result_{meeting_folder}" / "05_final_result.json"   
        
        if not meeting_path.exists():
            logger.warning(f"회의록 파일 없음: {meeting_path}")
            return ""
        
        try:
            with open(meeting_path, 'r', encoding='utf-8') as f:
                meeting_data = json.load(f)
            
            # 실제 프로젝트에서 사용하는 형식으로 회의 내용 구성
            meeting_text = ""
            for item in meeting_data:
                timestamp = item.get('timestamp', 'Unknown')
                speaker = item.get('speaker', 'Unknown')
                text = item.get('text', '')
                meeting_text += f"[{timestamp}] {speaker}: {text}\n"
            
            return meeting_text.strip()
        except Exception as e:
            logger.error(f"회의록 로드 실패 {meeting_path}: {e}")
            return ""

    def convert_to_training_format(self, gold_data: List[Dict[str, Any]], 
                                 source_dir: str = None) -> List[Dict[str, str]]:
        if source_dir is None:
            source_dir = os.path.join(PROJECT_ROOT, "batch_triplet_results_input")
        """골드 스탠다드를 Qwen3 파인튜닝 형식으로 변환 (청킹 지원)"""
        training_data = []
        
        for item in gold_data:
            try:
                metadata = item.get('metadata', {})
                source_file = metadata.get('source_file', '')
                is_chunk = metadata.get('is_chunk', False)
                
                logger.info(f"🔄 처리 중: {item.get('id', 'Unknown')}")
                logger.info(f"   📂 청킹여부: {is_chunk}")
                logger.info(f"   📄 source_file: {source_file}")
                
                if not source_file:
                    logger.warning(f"❌ source_file 없음: {item.get('id', 'Unknown')}")
                    continue
                
                # source_file에서 실제 폴더명 추출 - OS 독립적 처리
                source_file_path = Path(source_file)
                
                # 경로 분리 후 result_ 폴더명 추출
                parts = source_file_path.parts
                source_folder = None
                
                for part in parts:
                    if part.startswith('result_'):
                        # result_ 접두사 제거하여 실제 폴더명 추출
                        source_folder = part[7:]  # 'result_' 길이가 7
                        break
                    elif part.startswith(('train_', 'val_')):
                        # train_ 또는 val_ 접두사 제거
                        if part.startswith('train_'):
                            source_folder = part[6:]  # 'train_' 길이가 6
                        else:
                            source_folder = part[4:]  # 'val_' 길이가 4
                        break
                
                if not source_folder:
                    # 폴더명을 찾지 못한 경우 파일명에서 추출 시도
                    source_folder = source_file_path.stem
                
                logger.info(f"   🗂️  추출된 폴더명: {source_folder}")
                logger.info(f"   📍 Input 경로: {Path(source_dir) / source_folder / '05_final_result.json'}")
                
                # 원본 회의 내용 로드
                full_meeting_content = self.load_meeting_content(source_dir, source_folder)
                if not full_meeting_content:
                    logger.warning(f"❌ 회의 내용 없음: {source_folder}")
                    continue
                
                logger.info(f"   📊 원본 회의록 길이: {len(full_meeting_content)}자")
                
                # 골드 스탠다드 생성과 동일한 청킹 조건 적용
                CHUNK_THRESHOLD = 5000  # 설정 가능한 상수로 변경
                if len(full_meeting_content) > CHUNK_THRESHOLD:
                    # 5000자 초과: 청킹 처리
                    logger.info(f"   📏 긴 텍스트 감지 ({len(full_meeting_content)}자) - 청킹 처리")
                    chunks = self.chunk_text(full_meeting_content, chunk_size=CHUNK_THRESHOLD, overlap=512)
                    logger.info(f"   ✂️ {len(chunks)}개 청크로 분할")
                    
                    # 골드 스탠다드 ID에서 청크 인덱스 추출
                    item_id = item.get('id', '')
                    if '_chunk_' in item_id:
                        chunk_str = item_id.split('_chunk_')[-1]
                        try:
                            chunk_index = int(chunk_str) - 1  # 1-based → 0-based
                        except ValueError:
                            logger.error(f"   ❌ 청크 인덱스 파싱 실패: {chunk_str}")
                            continue
                        
                        if chunk_index < len(chunks):
                            meeting_content = chunks[chunk_index]
                            logger.info(f"   ✅ 청크 매칭 성공!")
                            logger.info(f"      - 사용할 청크: {chunk_index+1}/{len(chunks)}")
                            logger.info(f"      - 실제 청크 길이: {len(meeting_content)}자")
                        else:
                            logger.error(f"   ❌ 청크 인덱스 초과: {chunk_index+1} > {len(chunks)}")
                            continue
                    else:
                        # 청크 인덱스 없으면 첫 번째 청크 사용
                        meeting_content = chunks[0]
                        logger.info(f"   ⚠️  청크 인덱스 없음, 첫 번째 청크 사용")
                else:
                    # 5000자 이하: 전체 텍스트 사용
                    meeting_content = full_meeting_content
                    logger.info(f"   📖 전체 회의록 사용 (길이: {len(meeting_content)}자)")
                
                # 골드 스탠다드와 동일한 프롬프트 생성
                user_message = generate_meeting_analysis_user_prompt(meeting_content)
                
                # 골드 스탠다드 응답 처리
                notion_output = item.get('notion_output', {})
                if not notion_output:
                    logger.warning(f"notion_output 없음: {item.get('id', 'Unknown')}")
                    continue
                
                # JSON 문자열인 경우 파싱
                if isinstance(notion_output, str):
                    try:
                        # 마크다운 코드블록 제거
                        clean_json = notion_output.strip()
                        if clean_json.startswith('```json\n') and clean_json.endswith('\n```'):
                            clean_json = clean_json[8:-4]  # ```json\n과 \n``` 제거
                        elif clean_json.startswith('```\n') and clean_json.endswith('\n```'):
                            clean_json = clean_json[4:-4]  # ```\n과 \n``` 제거
                        
                        notion_output = json.loads(clean_json)
                    except Exception as e:
                        logger.warning(f"JSON 파싱 실패: {item.get('id', 'Unknown')} - {str(e)}")
                        continue
                    
                assistant_response = json.dumps(notion_output, ensure_ascii=False, indent=2)
                
                # Qwen3 채팅 형식으로 구성
                conversation = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{assistant_response}<|im_end|>"
                
                # 품질 메트릭 추출
                quality_metrics = item.get('quality_metrics', {})
                quality_score = quality_metrics.get('final_score', 7.0)
                is_high_quality = quality_metrics.get('is_high_quality', True)
                
                training_data.append({
                    "text": conversation,
                    "metadata": {
                        "id": item.get('id', 'Unknown'),
                        "source": source_folder,
                        "quality_score": quality_score,
                        "is_high_quality": is_high_quality,
                        "is_chunk": is_chunk,
                        "chunk_info": item.get('chunk_info', {}),
                        "dataset_type": "train" if "train_" in str(item.get('id', '')) else "val"
                    }
                })
                
                logger.info(f"변환 완료: {item.get('id', 'Unknown')} (품질: {quality_score}/10)")
                
            except Exception as e:
                logger.error(f"변환 실패 {item.get('id', 'Unknown')}: {e}")
                continue
        
        return training_data

class QwenFineTuner:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Thinking-2507"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """모델과 토크나이저 설정"""
        logger.info(f"모델 로딩: {self.model_name}")
        
        # GPU 메모리 정리
        cleanup_gpu_memory()
        
        # 메모리 사용량 모니터링
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"초기 GPU 메모리: {initial_memory:.1f}GB")
            
            # GPU 프로세스 확인
            import subprocess
            try:
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], 
                                      capture_output=True, text=True)
                if result.stdout.strip():
                    logger.warning("다른 GPU 프로세스가 실행 중입니다:")
                    logger.warning(result.stdout)
            except:
                pass
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드 전 메모리 정리
        cleanup_gpu_memory()
        
        # 일반 모델 로드 (메모리 최적화)
        try:
            # 4비트 양자화 설정
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # 모델 로드 설정
            model_kwargs = {
                "torch_dtype": torch.float16,
                "quantization_config": bnb_config,
                "device_map": "auto",
                "trust_remote_code": True,
                "use_cache": False,
                "low_cpu_mem_usage": True,
            }
            
            # Flash Attention 설정
            if FLASH_ATTN_AVAILABLE:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 활성화")
            
            logger.info("4비트 양자화로 모델 로딩 중...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # gradient checkpointing 설정 (메모리 효율성)
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                try:
                    self.model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing 활성화")
                except Exception as e:
                    logger.warning(f"Gradient checkpointing 활성화 실패 (4비트 양자화와 충돌 가능): {e}")
            
        except ImportError as e:
            if "bitsandbytes" in str(e):
                logger.error("❌ bitsandbytes 라이브러리가 필요합니다. pip install bitsandbytes")
            raise RuntimeError(f"필수 라이브러리 누락: {e}")
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            logger.error("GPU 메모리 부족이거나 모델 다운로드 문제일 수 있습니다.")
            raise RuntimeError(f"모델 로드 실패: {e}")
        
        # 모델 로딩 후 메모리 확인
        if torch.cuda.is_available():
            model_memory = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"모델 로딩 후 GPU 메모리: {model_memory:.1f}GB")
            
        # 모델 크기 확인
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"실제 모델 파라미터 수: {total_params:,}")
        
        logger.info("모델과 토크나이저 로딩 완료")
    
    def setup_lora_config(self) -> LoraConfig:
        """LoRA 설정"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,   # 원래 설정 유지
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 전체 어텐션 모듈
            bias="none",
        )
    
    def prepare_dataset(self, training_data: List[Dict[str, str]], max_length: int = 12000, val_split: float = 0.2):
        """데이터셋 준비 및 토크나이징"""
        def tokenize_function(examples):
            # 텍스트 토크나이징
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None,
            )
            
            # labels = input_ids (자동회귀 언어 모델링)
            tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
            
            return tokenized
        
        # 모든 데이터 사용
        all_data = training_data
        
        logger.info(f"전체 데이터: {len(all_data)}개")
        
        # train/val 분할
        import random
        random.seed(42)  # 재현성을 위한 시드 고정
        all_data_copy = all_data.copy()
        random.shuffle(all_data_copy)
        all_data = all_data_copy
        split_idx = int(len(all_data) * (1 - val_split))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        logger.info(f"학습 데이터: {len(train_data)}개, 검증 데이터: {len(val_data)}개")
        
        if len(train_data) == 0:
            logger.error("학습 데이터가 없습니다!")
            return None, None
        
        # Dataset 객체 생성
        train_dataset = Dataset.from_list([{"text": item["text"]} for item in train_data])
        val_dataset = Dataset.from_list([{"text": item["text"]} for item in val_data]) if val_data else None
        
        # 토크나이징
        train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=1, remove_columns=["text"])
        if val_dataset is not None:
            val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=1, remove_columns=["text"])
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, output_dir: str = "./qwen3_4B_Thinking_lora_ttalkkac"):
        """LoRA 파인튜닝 실행"""
        
        # LoRA 적용
        lora_config = self.setup_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        
        # 4비트 모델에서 학습 가능하도록 설정
        self.model.enable_input_require_grads()
        
        # LoRA 파라미터 확인
        self.model.print_trainable_parameters()
        
        # 학습 인자 설정
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8 if torch.cuda.is_available() else 4,  # GPU/CPU에 따라 조정
            warmup_steps=50,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,  # 매 스텝마다 로깅
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            push_to_hub=False,
            logging_dir=f"{output_dir}/logs",
            optim="paged_adamw_8bit",  # 8비트 optimizer
            max_grad_norm=0.3,
        )
        
        # 데이터 콜레이터
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # 트레이너 설정
        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": val_dataset,
            "data_collator": data_collator,
        }
        
        # Transformers 버전에 따라 파라미터 선택
        if version.parse(transformers_version) >= version.parse("4.41.0"):
            trainer_kwargs["processing_class"] = self.tokenizer
        else:
            trainer_kwargs["tokenizer"] = self.tokenizer
        
        trainer = Trainer(**trainer_kwargs)
        
        # 학습 실행
        logger.info("파인튜닝 시작...")
        train_result = trainer.train()
        
        # 모델 저장
        trainer.save_model()
        trainer.save_state()
        
        # 학습 결과 저장
        with open(os.path.join(output_dir, "training_results.json"), "w") as f:
            json.dump({
                "train_runtime": train_result.metrics["train_runtime"],
                "train_samples_per_second": train_result.metrics["train_samples_per_second"],
                "train_steps_per_second": train_result.metrics["train_steps_per_second"],
                "total_flos": train_result.metrics.get("total_flos", 0),
                "train_loss": train_result.metrics["train_loss"],
            }, f, indent=2)
        
        logger.info(f"파인튜닝 완료! 모델 저장 경로: {output_dir}")
        return trainer

def main():
    print("=" * 60)
    print("🚀 Ttalkkac Qwen3 LoRA 파인튜닝 시작 (4비트 양자화)")
    print("=" * 60)
    
    # 환경 정보 출력
    print(f"\n📁 프로젝트 루트: {PROJECT_ROOT}")
    print(f"🐍 Python 버전: {sys.version}")
    print(f"🔧 PyTorch 버전: {torch.__version__}")
    print(f"🤗 Transformers 버전: {transformers_version}")
    
    # GPU 메모리 정리
    cleanup_gpu_memory()
    
    # Transformers 버전 확인
    required_version = "4.37.0"
    current_version = version.parse(transformers_version)
    required_version_parsed = version.parse(required_version)
    
    if current_version < required_version_parsed:
        print(f"❌ Transformers 버전이 부족합니다!")
        print(f"   현재 버전: {transformers_version}")
        print(f"   필요 버전: {required_version}+")
        return
    
    print(f"✅ Transformers 버전 확인: {transformers_version}")
    
    # 1. 데이터 변환
    print("\n📊 1. 골드 스탠다드 데이터 로드 및 변환")
    converter = TtalkkacDatasetConverter()
    
    results_dir = os.getenv('RESULTS_DIR', os.path.join(PROJECT_ROOT, "ttalkkac_gold_standard_results_output"))
    gold_data = converter.load_gold_standard_data(results_dir)
    
    if not gold_data:
        print("❌ 골드 스탠다드 데이터를 찾을 수 없습니다.")
        return
    
    print(f"✅ 골드 스탠다드 데이터 로드 완료: {len(gold_data)}개")
    
    # 실제 프로젝트 형식으로 변환
    training_data = converter.convert_to_training_format(gold_data)
    
    if not training_data:
        print("❌ 변환된 학습 데이터가 없습니다.")
        return
    
    print(f"✅ 학습 데이터 변환 완료: {len(training_data)}개")
    
    # 2. 파인튜닝 설정 및 실행
    print("\n🤖 2. Qwen3 모델 설정 및 LoRA 파인튜닝")
    
    # GPU 확인
    if torch.cuda.is_available():
        print(f"✅ GPU 사용 가능: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"   GPU 개수: {torch.cuda.device_count()}")
    else:
        print("⚠️ CPU 모드로 실행됩니다.")
    
    # 파인튜너 초기화
    finetuner = QwenFineTuner("Qwen/Qwen3-4B-Thinking-2507")
    finetuner.data_converter = converter
    
    # 모델과 토크나이저 설정
    try:
        finetuner.setup_model_and_tokenizer()
    except RuntimeError as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 데이터셋 준비
    MAX_SEQ_LENGTH = int(os.getenv('MAX_SEQ_LENGTH', '12000'))
    VAL_SPLIT_RATIO = float(os.getenv('VAL_SPLIT_RATIO', '0.2'))
    train_dataset, val_dataset = finetuner.prepare_dataset(
        training_data, 
        max_length=MAX_SEQ_LENGTH,
        val_split=VAL_SPLIT_RATIO
    )
    
    if train_dataset is None:
        print("❌ 학습 데이터셋 준비 실패")
        return
    
    # 파인튜닝 실행
    output_dir = f"./qwen3_4B_Thinking_lora_ttalkkac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = finetuner.train(train_dataset, val_dataset, output_dir)
    
    print("\n🎉 파인튜닝 완료!")
    print(f"📁 모델 저장 경로: {output_dir}")  

if __name__ == "__main__":
    main()