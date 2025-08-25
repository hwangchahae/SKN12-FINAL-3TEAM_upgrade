import json
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import yaml

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import login

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelConfig:
    """모델 설정 관리 클래스"""
    def __init__(self, config_path="config.yaml"):
        # 기본값 설정
        self.BASE_MODEL_PATH: str = "Qwen/Qwen3-8B"
        self.LORA_MODEL_PATH: str = "qwen3_lora_ttalkkac_8b"
        self.MERGED_MODEL_PATH: str = "8B_merged_qwen3_lora_model"
        self.INPUT_DIR: str = "../Raw_Data_val"
        self.OUTPUT_DIR: str = "8B_lora_model_results"
        self.MAX_NEW_TOKENS: int = 2048
        self.TEMPERATURE: float = 0.3
        self.TOP_P: float = 0.9
        self.REPETITION_PENALTY: float = 1.1
        self.CHUNK_SIZE: int = 5000
        self.CHUNK_OVERLAP: int = 512
        self.TEST_FILE_LIMIT: int = 0
        
        # vLLM 전용 설정
        self.TENSOR_PARALLEL_SIZE: int = 1
        self.GPU_MEMORY_UTILIZATION: float = 0.9
        self.MAX_MODEL_LEN: int = 8192
        self.DTYPE: str = "auto"
        self.TRUST_REMOTE_CODE: bool = True
        
        # 설정 파일 로드
        self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path):
        """YAML 설정 파일에서 설정 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config and 'lora_model' in config:
                lora_config = config['lora_model']
                self.BASE_MODEL_PATH = lora_config.get('base_model_path', self.BASE_MODEL_PATH)
                self.LORA_MODEL_PATH = lora_config.get('lora_model_path', self.LORA_MODEL_PATH)
                self.MERGED_MODEL_PATH = lora_config.get('merged_model_path', self.MERGED_MODEL_PATH)
                self.INPUT_DIR = lora_config.get('input_dir', self.INPUT_DIR)
                self.OUTPUT_DIR = lora_config.get('output_dir', self.OUTPUT_DIR)
                self.MAX_NEW_TOKENS = lora_config.get('max_new_tokens', self.MAX_NEW_TOKENS)
                self.TEMPERATURE = lora_config.get('temperature', self.TEMPERATURE)
                self.TOP_P = lora_config.get('top_p', self.TOP_P)
                self.REPETITION_PENALTY = lora_config.get('repetition_penalty', self.REPETITION_PENALTY)
                self.CHUNK_SIZE = lora_config.get('chunk_size', self.CHUNK_SIZE)
                self.CHUNK_OVERLAP = lora_config.get('chunk_overlap', self.CHUNK_OVERLAP)
                self.TEST_FILE_LIMIT = lora_config.get('test_file_limit', self.TEST_FILE_LIMIT)
                self.TENSOR_PARALLEL_SIZE = lora_config.get('tensor_parallel_size', self.TENSOR_PARALLEL_SIZE)
                self.GPU_MEMORY_UTILIZATION = lora_config.get('gpu_memory_utilization', self.GPU_MEMORY_UTILIZATION)
                self.MAX_MODEL_LEN = lora_config.get('max_model_len', self.MAX_MODEL_LEN)
                
                logger.info(f"설정 파일 로드 완료: {config_path}")
        except FileNotFoundError:
            logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}. 기본 설정을 사용합니다.")
        except Exception as e:
            logger.error(f"설정 파일 로드 오류: {e}. 기본 설정을 사용합니다.")


@dataclass
class MeetingData:
    """회의 데이터 구조체"""
    transcript: Optional[str] = None
    chunks: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_chunked(self) -> bool:
        return self.chunks is not None


@dataclass
class ProcessingStats:
    """처리 통계 관리"""
    total: int = 0
    processed: int = 0
    success: int = 0
    failed: int = 0
    chunked: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.processed == 0:
            return 0.0
        return (self.success / self.processed) * 100


class QwenVLLMMeetingGenerator:
    """vLLM을 사용한 Qwen LoRA 모델 회의록 생성기"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        생성자
        
        Args:
            config: 모델 설정 객체
        """
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.sampling_params = None
        
        self._initialize_model()
    
    def _merge_lora_if_needed(self) -> str:
        """필요시 LoRA 모델 병합"""
        # 현재 스크립트 위치 기준으로 경로 설정
        script_dir = Path(__file__).parent
        lora_path = script_dir / self.config.LORA_MODEL_PATH
        merged_path = script_dir / self.config.MERGED_MODEL_PATH
        
        # 병합된 모델이 이미 있는지 확인
        if merged_path.exists():
            logger.info(f"병합된 모델이 이미 존재: {merged_path}")
            return str(merged_path)
        
        # LoRA 어댑터가 있는지 확인
        if not lora_path.exists():
            logger.info(f"LoRA 어댑터가 없음: {lora_path}. 베이스 모델 사용")
            return self.config.BASE_MODEL_PATH
        
        # LoRA 병합 수행
        logger.info("LoRA 병합 시작...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import gc
            
            # 메모리 정리
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info(f"베이스 모델 로딩: {self.config.BASE_MODEL_PATH}")
            # float16 사용하고 trust_remote_code 추가
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.BASE_MODEL_PATH,
                torch_dtype=torch.float16,  # bfloat16 대신 float16 사용
                device_map="cpu",  # 병합 시에는 CPU 사용
                trust_remote_code=True,
                low_cpu_mem_usage=True  # 메모리 사용량 최소화
            )
            
            logger.info(f"LoRA 어댑터 로딩: {lora_path}")
            
            # safetensors 파일이 있으면 이름 변경
            safetensors_file = lora_path / "adapter_model.safetensors"
            if safetensors_file.exists():
                logger.warning("safetensors 파일 감지 - bin 파일 사용을 위해 이름 변경")
                safetensors_backup = lora_path / "adapter_model.safetensors.disabled"
                if safetensors_backup.exists():
                    safetensors_backup.unlink()
                safetensors_file.rename(safetensors_backup)
                logger.info("safetensors 파일을 .disabled로 이름 변경")
            
            # bin 파일 확인
            bin_file = lora_path / "adapter_model.bin"
            if not bin_file.exists():
                logger.warning("adapter_model.bin이 없습니다. safetensors를 변환합니다...")
                from safetensors.torch import load_file
                safetensors_disabled = lora_path / "adapter_model.safetensors.disabled"
                if safetensors_disabled.exists():
                    state_dict = load_file(str(safetensors_disabled))
                    torch.save(state_dict, str(bin_file))
                    logger.info("✅ bin 파일로 변환 완료")
            
            # LoRA 어댑터 로드 - 로컬 파일 직접 로드
            logger.info("LoRA 어댑터를 로컬 파일에서 직접 로드합니다...")
            model = PeftModel.from_pretrained(
                base_model, 
                lora_path,  # str() 제거 - Path 객체 직접 전달
                is_trainable=False,  # 추론 모드
                local_files_only=True,  # 로컬 파일만 사용
                use_safetensors=False,  # SafeTensors 비활성화
            )
            
            logger.info("모델 병합 중...")
            merged_model = model.merge_and_unload()
            
            logger.info(f"병합된 모델 저장: {merged_path}")
            merged_path.mkdir(parents=True, exist_ok=True)
            
            # safetensors 형식으로 저장
            merged_model.save_pretrained(
                str(merged_path),
                safe_serialization=True,
                max_shard_size="4GB"
            )
            
            # 토크나이저도 저장
            logger.info("토크나이저 저장 중...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.BASE_MODEL_PATH,
                trust_remote_code=True
            )
            tokenizer.save_pretrained(str(merged_path))
            
            logger.info("✅ LoRA 병합 완료!")
            
            # 메모리 정리
            del base_model
            del model
            del merged_model
            gc.collect()
            torch.cuda.empty_cache()
            
            return str(merged_path)
            
        except Exception as e:
            logger.error(f"LoRA 병합 실패: {e}")
            logger.error(f"에러 타입: {type(e).__name__}")
            
            # 더 자세한 에러 정보 출력
            import traceback
            logger.error("상세 에러:")
            logger.error(traceback.format_exc())
            
            logger.warning("베이스 모델을 사용합니다.")
            return self.config.BASE_MODEL_PATH
    
    def _initialize_model(self) -> None:
        """vLLM 모델 초기화"""
        try:
            logger.info("vLLM 모델 초기화 시작...")
            
            # LoRA 병합 확인 및 수행
            model_path = self._merge_lora_if_needed()
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # vLLM 모델 초기화
            logger.info(f"vLLM 모델 로딩: {model_path}")
            self.model = LLM(
                model=model_path,
                tensor_parallel_size=self.config.TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=self.config.GPU_MEMORY_UTILIZATION,
                max_model_len=self.config.MAX_MODEL_LEN,
                dtype=self.config.DTYPE,
                trust_remote_code=self.config.TRUST_REMOTE_CODE,
                enforce_eager=False,  # CUDA graphs 사용 (더 빠름)
                max_num_batched_tokens=self.config.MAX_MODEL_LEN,
                max_num_seqs=256,  # 동시 처리 시퀀스 수
            )
            
            # 샘플링 파라미터 설정
            self.sampling_params = SamplingParams(
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                repetition_penalty=self.config.REPETITION_PENALTY,
                max_tokens=self.config.MAX_NEW_TOKENS,
                stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else None,
            )
            
            logger.info("vLLM 모델 로딩 완료!")
            
        except Exception as e:
            logger.error(f"vLLM 모델 초기화 실패: {e}")
            raise
    
    def find_meeting_files(self, base_dir: str) -> List[Path]:
        """
        회의 파일 검색
        
        Args:
            base_dir: 검색할 기본 디렉토리
            
        Returns:
            발견된 파일 경로 리스트
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            logger.warning(f"디렉토리가 존재하지 않음: {base_dir}")
            return []
        
        target_files = list(base_path.rglob("05_final_result.json"))
        logger.info(f"{len(target_files)}개의 회의 파일 발견")
        return target_files
    
    def chunk_text(self, text: str, chunk_size: int = 5000, overlap: int = 512) -> List[str]:
        """텍스트를 청킹하여 나누기"""
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
    
    def load_meeting_data(self, file_path: Path) -> Optional[MeetingData]:
        """
        회의 데이터 로드
        
        Args:
            file_path: 파일 경로
            
        Returns:
            MeetingData 객체 또는 None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 텍스트 변환
            meeting_lines = []
            speakers = set()
            
            for item in data:
                timestamp = item.get('timestamp', 'Unknown')
                speaker = item.get('speaker', 'Unknown')
                text = item.get('text', '')
                speakers.add(speaker)
                meeting_lines.append(f"[{timestamp}] {speaker}: {text}")
            
            full_text = '\n'.join(meeting_lines)
            
            # 메타데이터 생성
            metadata = {
                "source_file": str(file_path),
                "utterance_count": len(data),
                "speakers": list(speakers),
                "original_length": len(full_text)
            }
            
            # 청킹 여부 결정
            if len(full_text) > self.config.CHUNK_SIZE:
                logger.info(f"긴 텍스트 감지 ({len(full_text)}자) - 청킹 처리")
                chunks = self.chunk_text(full_text, self.config.CHUNK_SIZE, self.config.CHUNK_OVERLAP)
                metadata["chunking_info"] = {
                    "is_chunked": True,
                    "total_chunks": len(chunks)
                }
                return MeetingData(chunks=chunks, metadata=metadata)
            else:
                metadata["chunking_info"] = {
                    "is_chunked": False,
                    "total_chunks": 1
                }
                return MeetingData(transcript=full_text, metadata=metadata)
                
        except Exception as e:
            logger.error(f"파일 로드 오류 ({file_path}): {e}")
            return None
    
    def generate_response(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        vLLM을 사용한 모델 응답 생성
        
        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            
        Returns:
            생성된 응답 또는 None
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 프롬프트 생성
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # vLLM 생성
            outputs = self.model.generate(
                prompts=[prompt],
                sampling_params=self.sampling_params
            )
            

            # 첫 번째 출력 가져오기
            if outputs and len(outputs) > 0:
                response = outputs[0].outputs[0].text
                return response.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"응답 생성 오류: {e}")
            return None
    
    def generate_batch_responses(self, prompts: List[Tuple[str, str]]) -> List[Optional[str]]:
        """
        배치 처리를 위한 vLLM 생성 (더 효율적)
        
        Args:
            prompts: (system_prompt, user_prompt) 튜플 리스트
            
        Returns:
            생성된 응답 리스트
        """
        try:
            # 모든 프롬프트 준비
            formatted_prompts = []
            for system_prompt, user_prompt in prompts:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_prompts.append(prompt)
            
            # vLLM 배치 생성
            outputs = self.model.generate(
                prompts=formatted_prompts,
                sampling_params=self.sampling_params
            )
            
            # 결과 추출
            results = []
            for output in outputs:
                if output.outputs:
                    results.append(output.outputs[0].text.strip())
                else:
                    results.append(None)
            
            return results
            
        except Exception as e:
            logger.error(f"배치 응답 생성 오류: {e}")
            return [None] * len(prompts)
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        JSON 응답 파싱
        
        Args:
            response: 모델 응답 문자열
            
        Returns:
            파싱된 딕셔너리
        """
        try:
            # JSON 블록 추출
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                response = response[start:end]
            
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("JSON 파싱 실패, 원본 텍스트 반환")
            return {"raw_text": response}
    
    def generate_notion_project(self, transcript: str) -> Dict[str, Any]:
        """
        노션 프로젝트 생성
        
        Args:
            transcript: 회의록 텍스트
            
        Returns:
            생성 결과
        """
        try:
            # 프롬프트 임포트 시도
            try:
                import sys
                import os
                ai_engine_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai-engine-dev')
                if ai_engine_path not in sys.path:
                    sys.path.insert(0, ai_engine_path)
                from prd_generation_prompts import generate_notion_project_prompt
            except ImportError:
                # 폴백: 기본 프롬프트 함수 사용
                def generate_notion_project_prompt(meeting_transcript: str) -> str:
                    return f"""
다음 회의 전사본을 바탕으로 노션에 업로드할 프로젝트 기획안을 작성하세요.

**회의 전사본:**
{meeting_transcript}

**작성 지침:**
1. 회의에서 논의된 내용을 바탕으로 체계적인 기획안을 작성
2. 프로젝트명은 회의 내용을 바탕으로 적절히 명명
3. 목적과 목표는 명확하고 구체적으로 작성
4. 실행 계획은 실현 가능한 단계별로 구성
5. 기대 효과는 정량적/정성적 결과를 포함
6. 모든 내용은 한국어로 작성

**응답 형식:**
다음 JSON 형식으로 응답하세요:
{{
    "project_name": "프로젝트명",
    "project_purpose": "프로젝트의 주요 목적",
    "project_period": "예상 수행 기간",
    "project_manager": "담당자명",
    "core_objectives": [
        "목표 1: 구체적인 목표",
        "목표 2: 구체적인 목표",
        "목표 3: 구체적인 목표"
    ],
    "core_idea": "핵심 아이디어 설명",
    "idea_description": "아이디어의 기술적/비즈니스적 설명",
    "execution_plan": "단계별 실행 계획과 일정",
    "expected_effects": [
        "기대효과 1: 자세한 설명",
        "기대효과 2: 자세한 설명",
        "기대효과 3: 자세한 설명"
    ]
}}
"""
            
            user_prompt = generate_notion_project_prompt(transcript)
            system_prompt = """당신은 회의록을 분석하여 체계적인 프로젝트 기획안을 작성하는 전문가입니다.
회의에서 논의된 내용을 바탕으로 명확하고 실행 가능한 기획안을 작성해주세요.
응답은 반드시 요청된 JSON 형식으로만 제공하세요."""
            
            response = self.generate_response(system_prompt, user_prompt)
            
            if not response:
                return {"success": False, "error": "응답 생성 실패"}
            
            result = self.parse_json_response(response)
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"노션 프로젝트 생성 오류: {e}")
            return {"success": False, "error": str(e)}
    
    def process_meeting(self, 
                       meeting_data: MeetingData,
                       output_dir: Path,
                       file_index: int,
                       parent_folder: str) -> Tuple[int, int]:
        """
        회의 데이터 처리
        
        Args:
            meeting_data: 회의 데이터
            output_dir: 출력 디렉토리
            file_index: 파일 인덱스
            parent_folder: 부모 폴더명
            
        Returns:
            (성공 수, 실패 수) 튜플
        """
        success_count = 0
        fail_count = 0
        
        if meeting_data.is_chunked:
            # 청킹된 데이터 배치 처리 준비
            batch_prompts = []
            
            # 프롬프트 함수 임포트 또는 폴백
            try:
                import sys
                import os
                ai_engine_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai-engine-dev')
                if ai_engine_path not in sys.path:
                    sys.path.insert(0, ai_engine_path)
                from prd_generation_prompts import generate_notion_project_prompt
                from meeting_analysis_prompts import generate_meeting_analysis_user_prompt
            except ImportError:
                # 폴백 함수는 이미 generate_notion_project 메서드에서 정의됨
                pass
            
            system_prompt = """당신은 회의록을 분석하여 체계적인 프로젝트 기획안을 작성하는 전문가입니다.
회의에서 논의된 내용을 바탕으로 명확하고 실행 가능한 기획안을 작성해주세요.
응답은 반드시 요청된 JSON 형식으로만 제공하세요."""
            
            # 배치 프롬프트 준비
            for idx, chunk_text in enumerate(meeting_data.chunks):
                if idx == 0:
                    # 첫 번째 청크는 노션 프로젝트 프롬프트 사용
                    user_prompt = generate_notion_project_prompt(chunk_text)
                else:
                    # 후속 청크는 회의 분석 프롬프트 사용 (이전 컨텍스트 포함)
                    if 'generate_meeting_analysis_user_prompt' in locals():
                        additional_context = f"이전 청크 {idx}개 처리 완료"
                        user_prompt = generate_meeting_analysis_user_prompt(chunk_text, additional_context)
                    else:
                        # 폴백: 노션 프로젝트 프롬프트 사용
                        user_prompt = generate_notion_project_prompt(chunk_text)
                batch_prompts.append((system_prompt, user_prompt))
            
            logger.info(f"{len(batch_prompts)}개 청크 배치 처리 시작")
            
            # 배치 생성
            responses = self.generate_batch_responses(batch_prompts)
            
            # 결과 처리
            total_chunks = len(meeting_data.chunks)
            for chunk_idx, (chunk_text, response) in enumerate(zip(meeting_data.chunks, responses)):
                # 청크가 1개면 _chunk_X 붙이지 않음
                if total_chunks == 1:
                    chunk_dir = output_dir / parent_folder
                    chunk_id = parent_folder
                else:
                    chunk_dir = output_dir / f"{parent_folder}_chunk_{chunk_idx+1}"
                    chunk_id = f"{parent_folder}_chunk_{chunk_idx+1}"
                
                chunk_dir.mkdir(exist_ok=True)
                
                if response:
                    result_data = self.parse_json_response(response)
                    
                    chunk_result = {
                        "id": chunk_id,
                        "source_dir": parent_folder,
                        "notion_output": result_data,
                        "metadata": {
                            **meeting_data.metadata,
                            "is_chunk": total_chunks > 1,
                            "chunk_index": chunk_idx + 1 if total_chunks > 1 else None,
                                "processing_date": datetime.now().isoformat()
                        }
                    }
                    
                    with open(chunk_dir / "result.json", 'w', encoding='utf-8') as f:
                        json.dump(chunk_result, f, ensure_ascii=False, indent=2)
                    
                    success_count += 1
                    if total_chunks > 1:
                        logger.info(f"청크 {chunk_idx+1}/{total_chunks} 저장 완료")
                    else:
                        logger.info(f"저장 완료")
                else:
                    fail_count += 1
                    if total_chunks > 1:
                        logger.error(f"청크 {chunk_idx+1}/{total_chunks} 생성 실패")
                    else:
                        logger.error(f"생성 실패")
                    
        else:
            # 단일 텍스트 처리 (청크되지 않은 파일도 저장)
            logger.info("전체 회의록 처리 중")
            result = self.generate_notion_project(meeting_data.transcript)
            
            if result["success"]:
                # 단일 파일도 저장
                single_dir = output_dir / parent_folder
                single_dir.mkdir(exist_ok=True)
                
                single_result = {
                    "id": parent_folder,
                    "source_dir": parent_folder,
                    "notion_output": result["result"],
                    "metadata": {
                        **meeting_data.metadata,
                        "is_chunk": False,
                        "chunk_index": None,
                        "processing_date": datetime.now().isoformat()
                    }
                }
                
                with open(single_dir / "result.json", 'w', encoding='utf-8') as f:
                    json.dump(single_result, f, ensure_ascii=False, indent=2)
                
                success_count += 1
                logger.info("저장 완료")
            else:
                fail_count += 1
                logger.error(f"생성 실패: {result.get('error')}")
        
        return success_count, fail_count


def setup_huggingface_auth() -> bool:
    """
    Hugging Face 인증 설정
    
    Returns:
        인증 성공 여부
    """
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        logger.warning("HUGGINGFACE_TOKEN 환경변수가 설정되지 않았습니다.")
        hf_token = input("Hugging Face 토큰을 입력하세요: ").strip()
        if not hf_token:
            logger.error("토큰이 입력되지 않았습니다.")
            return False
    
    try:
        login(token=hf_token)
        logger.info("Hugging Face 로그인 성공!")
        return True
    except Exception as e:
        logger.error(f"Hugging Face 로그인 실패: {e}")
        return False


def main():
    """메인 실행 함수"""
    logger.info("=" * 60)
    logger.info("vLLM을 사용한 Qwen 모델 회의록 처리 시작!")
    logger.info("=" * 60)
    
    # Hugging Face 인증
    if not setup_huggingface_auth():
        return
    
    # 설정 로드 (config.yaml 파일에서)
    config = ModelConfig("config.yaml")
    
    # 설정에서 경로 가져오기
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    logger.info(f"결과 저장 폴더: {output_dir}")
    
    # 모델 초기화
    try:
        generator = QwenVLLMMeetingGenerator(config)
    except Exception as e:
        logger.error(f"모델 초기화 실패: {e}")
        return
    
    # 회의 파일 검색 (설정에서 경로 가져오기)
    input_dir = config.INPUT_DIR
    meeting_files = generator.find_meeting_files(input_dir)
    
    if not meeting_files:
        logger.error(f"{input_dir} 폴더에서 파일을 찾을 수 없습니다.")
        return
    
    # 테스트 모드 처리
    if config.TEST_FILE_LIMIT > 0:
        meeting_files = meeting_files[:config.TEST_FILE_LIMIT]
        logger.info(f"테스트 모드: {len(meeting_files)}개 파일만 처리")
    else:
        logger.info(f"전체 파일 처리 모드: {len(meeting_files)}개 파일 처리")
    
    # 통계 초기화
    stats = ProcessingStats(total=len(meeting_files))
    dataset = []
    
    # 각 파일 처리
    for i, file_path in enumerate(meeting_files, 1):
        parent_folder = file_path.parent.name
        logger.info(f"\n[{i}/{len(meeting_files)}] {parent_folder}/{file_path.name} 처리 중...")
        
        try:
            # 데이터 로드
            meeting_data = generator.load_meeting_data(file_path)
            if not meeting_data:
                stats.failed += 1
                stats.processed += 1
                continue
            
            if meeting_data.is_chunked:
                stats.chunked += 1
            
            # 처리
            success, fail = generator.process_meeting(
                meeting_data, output_dir, i, parent_folder
            )
            
            stats.success += success
            stats.failed += fail
            stats.processed += success + fail
            
        except Exception as e:
            logger.error(f"처리 중 오류: {e}")
            stats.failed += 1
            stats.processed += 1
    
    # 결과 출력
    logger.info("=" * 60)
    logger.info("처리 완료 통계:")
    logger.info(f"  전체 파일: {stats.total}개")
    logger.info(f"  처리 완료: {stats.processed}개")
    logger.info(f"  성공: {stats.success}개")
    logger.info(f"  실패: {stats.failed}개")
    logger.info(f"  청킹 처리: {stats.chunked}개")
    logger.info(f"  성공률: {stats.success_rate:.1f}%")
    
    logger.info(f"\n✅ 모든 처리 완료! 결과는 {output_dir}에 저장되었습니다.")


if __name__ == "__main__":
    main()