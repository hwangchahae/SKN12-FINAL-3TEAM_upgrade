"""
로컬 회의록 유사도 평가 시스템 - 최적화 버전
- 배치 임베딩 처리
- 진행 상황 저장
- 재시작 가능
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import re
import random
import os
import argparse
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import yaml

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """평가 설정"""
    tfidf_max_features: int = 5000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    sample_size: int = 100  # 랜덤 샘플링 크기
    random_seed: int = 42  # 재현 가능한 랜덤 샘플링을 위한 시드
    embedding_model: str = "text-embedding-3-large"  # 임베딩 모델
    batch_size: int = 10  # 배치 처리 크기
    save_interval: int = 20  # 중간 저장 간격
    use_cache: bool = True  # 캐시 사용 여부


@dataclass
class FileMatch:
    """파일 매칭 정보"""
    gold_folder: Path
    result_file: Path
    suffix: str
    gold_text: str = ""
    result_text: str = ""


@dataclass
class EvaluationScore:
    """개별 평가 점수"""
    file_name: str
    tfidf_cosine: float
    embedding_cosine: float
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return asdict(self)


@dataclass
class EvaluationResult:
    """전체 평가 결과"""
    scores: List[EvaluationScore]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    sample_size: int = 0
    total_available: int = 0
    
    @property
    def mean_tfidf_cosine(self) -> float:
        return np.mean([s.tfidf_cosine for s in self.scores]) if self.scores else 0.0
    
    @property
    def mean_embedding_cosine(self) -> float:
        return np.mean([s.embedding_cosine for s in self.scores]) if self.scores else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "timestamp": self.timestamp,
            "sampling_info": {
                "sample_size": self.sample_size,
                "total_available": self.total_available,
                "sampling_rate": f"{(self.sample_size/self.total_available*100):.1f}%" if self.total_available > 0 else "0%"
            },
            "summary": {
                "total_files": len(self.scores),
                "mean_tfidf_cosine": self.mean_tfidf_cosine,
                "mean_embedding_cosine": self.mean_embedding_cosine,
                
            },
            "details": [
                {
                    "file_name": s.file_name,
                    "tfidf_cosine": s.tfidf_cosine,
                    "embedding_cosine": s.embedding_cosine,
            
                } for s in self.scores
            ]
        }


class FileMatcher:
    """파일 매칭 유틸리티"""
    
    @staticmethod
    def find_matches(gold_base_path: str, result_base_path: str) -> List[FileMatch]:
        """정답과 결과 파일 매칭 - 단일 JSON 파일 버전"""
        matches = []
        gold_path = Path(gold_base_path)
        result_path = Path(result_base_path)
        
        if not gold_path.exists() or not result_path.exists():
            logger.error(f"경로가 존재하지 않음: {gold_base_path} 또는 {result_base_path}")
            return matches
        
        logger.info("파일 매칭 시작...")
        
        # Result 폴더들에서 result.json 파일 찾기
        result_folders = [f for f in result_path.iterdir() if f.is_dir() and f.name.startswith("result_")]
        logger.info(f"Result 폴더 {len(result_folders)}개 발견")
        
        # 각 result 폴더에 대해 매칭되는 gold 폴더 찾기
        for result_folder in result_folders:
            # result.json 파일 확인
            result_json_file = result_folder / "result.json"
            if not result_json_file.exists():
                logger.warning(f"result.json 파일 없음: {result_folder}")
                continue
                
            # 폴더명에서 매칭 정보 추출
            folder_name = result_folder.name
            match = re.search(r'result_([A-Za-z0-9]+)_chunk_(\d+)', folder_name)
            if not match:
                # chunk가 없는 경우 (예: result_Bro015)
                match = re.search(r'result_([A-Za-z0-9]+)$', folder_name)
                if match:
                    base_name = match.group(1)
                    chunk_num = ""
                else:
                    # 한글 폴더명 처리
                    base_name = folder_name.replace("result_", "")
                    chunk_num = ""
            else:
                base_name = match.group(1)
                chunk_num = match.group(2)
            
            # Gold 폴더에서 매칭 찾기
            if chunk_num:
                # 폴더 이름 패턴들
                gold_folder_pattern1 = f"*result_{base_name}_chunk_{chunk_num}"
                gold_folder_pattern2 = f"*result_{base_name}_chunk{chunk_num}"
                
                # 패턴들 시도
                for pattern in [gold_folder_pattern1, gold_folder_pattern2]:
                    matching_folders = list(gold_path.glob(pattern))
                    if matching_folders:
                        gold_folder = matching_folders[0]
                        identifier = f"{base_name}_chunk{chunk_num}"
                        matches.append(FileMatch(gold_folder, result_json_file, identifier))
                        logger.debug(f"매칭 성공: {gold_folder.name} <-> {result_folder.name}")
                        break
            else:
                # chunk가 없는 경우
                gold_folder_pattern = f"*result_{base_name}"
                matching_folders = list(gold_path.glob(gold_folder_pattern))
                if matching_folders:
                    gold_folder = matching_folders[0]
                    matches.append(FileMatch(gold_folder, result_json_file, base_name))
                    logger.debug(f"매칭 성공: {gold_folder.name} <-> {result_folder.name}")
                else:
                    # 정확한 폴더명 매칭 시도 (한글 이름 등)
                    gold_folder = gold_path / folder_name
                    if gold_folder.exists():
                        matches.append(FileMatch(gold_folder, result_json_file, base_name))
                        logger.debug(f"매칭 성공: {gold_folder.name} <-> {result_folder.name}")
        
        logger.info(f"총 {len(matches)}개 파일 쌍 매칭 완료")
        return matches
    
    @staticmethod
    def random_sample_matches(matches: List[FileMatch], sample_size: int, seed: int = 42) -> List[FileMatch]:
        """매칭된 파일 중 랜덤 샘플링"""
        if len(matches) <= sample_size:
            logger.info(f"전체 파일 수({len(matches)})가 샘플 크기({sample_size})보다 작거나 같음. 전체 사용")
            return matches
        
        random.seed(seed)
        sampled = random.sample(matches, sample_size)
        logger.info(f"총 {len(matches)}개 중 {sample_size}개 랜덤 샘플링 완료 (시드: {seed})")
        return sampled


class TextExtractor:
    """텍스트 추출 유틸리티"""
    
    @staticmethod
    def extract_gold_text(folder_path: Path) -> str:
        """정답 데이터에서 텍스트 추출"""
        json_path = folder_path / "result.json"
        
        if not json_path.exists():
            logger.warning(f"result.json 없음: {json_path}")
            return ""
        
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            
            notion = data.get("notion_output", "")
            
            # dict인 경우 JSON 문자열로 변환
            if isinstance(notion, dict):
                return json.dumps(notion, ensure_ascii=False, indent=2)
            
            # 코드 블록 제거
            if isinstance(notion, str) and notion.startswith("```json"):
                notion = notion.replace("```json", "").replace("```", "").strip()
            
            return str(notion)
            
        except Exception as e:
            logger.error(f"Gold 텍스트 추출 실패 ({folder_path}): {e}")
            return ""
    
    @staticmethod
    def extract_result_text(folder_or_file_path: Path) -> str:
        """결과 데이터에서 텍스트 추출"""
        # 폴더인 경우 result.json 파일 찾기
        if folder_or_file_path.is_dir():
            json_path = folder_or_file_path / "result.json"
        else:
            json_path = folder_or_file_path
            
        if not json_path.exists():
            logger.warning(f"결과 파일 없음: {json_path}")
            return ""
        
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            
            # 다양한 형식 처리
            # 1. 단일 JSON 파일의 경우 (result.generation_result.result 경로)
            if "result" in data and isinstance(data["result"], dict):
                if "generation_result" in data["result"] and isinstance(data["result"]["generation_result"], dict):
                    if "result" in data["result"]["generation_result"]:
                        result = data["result"]["generation_result"]["result"]
                        if isinstance(result, dict):
                            return json.dumps(result, ensure_ascii=False, indent=2)
                        return str(result)
            
            # 2. notion_output이 있는 경우 (학습 후 결과)
            if "notion_output" in data:
                notion = data.get("notion_output", "")
                if isinstance(notion, dict):
                    return json.dumps(notion, ensure_ascii=False, indent=2)
                if isinstance(notion, str) and notion.startswith("```json"):
                    notion = notion.replace("```json", "").replace("```", "").strip()
                return str(notion)
            
            # 3. 기타 형식
            result = data.get("result", {}).get("generation_result", {}).get("result", "")
            
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Result 텍스트 추출 실패 ({folder_or_file_path}): {e}")
            return ""


class SimilarityMetrics:
    """유사도 메트릭 계산 - 최적화 버전"""
    
    def __init__(self, config: EvaluationConfig):
        """초기화"""
        self.config = config
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            ngram_range=config.tfidf_ngram_range,
            stop_words=None
        )
        
        # 임베딩 캐시
        self.embedding_cache = {}
        self.cache_file = Path("embedding_cache.json")
        
        # 캐시 로드
        if config.use_cache and self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.embedding_cache = json.load(f)
                logger.info(f"임베딩 캐시 로드: {len(self.embedding_cache)}개 항목")
            except:
                self.embedding_cache = {}
        
        # OpenAI 클라이언트 초기화
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # .env 파일 확인
            env_path = Path(__file__).parent / ".env"
            if env_path.exists():
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.startswith("OPENAI_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            break 
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI 클라이언트 초기화 완료 (모델: {config.embedding_model})")
        else:
            self.openai_client = None
            logger.warning("OPENAI_API_KEY 환경변수가 설정되지 않음. 임베딩 유사도는 계산되지 않습니다.")
    
    def save_cache(self):
        """캐시 저장"""
        if self.config.use_cache:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.embedding_cache, f, ensure_ascii=False)
                logger.info(f"임베딩 캐시 저장: {len(self.embedding_cache)}개 항목")
            except Exception as e:
                logger.error(f"캐시 저장 실패: {e}")
    
    def compute_tfidf_cosine_similarity(self, text1: str, text2: str) -> float:
        """TF-IDF 기반 코사인 유사도 계산"""
        try:
            all_texts = [text1, text2]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"TF-IDF 코사인 유사도 계산 실패: {e}")
            return 0.0
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """OpenAI 임베딩 얻기 (캐시 지원)"""
        if not self.openai_client:
            return None
        
        # 캐시 확인
        text_hash = str(hash(text[:500]))  # 텍스트 앞부분으로 해시 생성
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
            
        try:
            response = self.openai_client.embeddings.create(
                model=self.config.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # 캐시 저장
            self.embedding_cache[text_hash] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return None
    
    def get_batch_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """배치 임베딩 얻기"""
        if not self.openai_client:
            return [None] * len(texts)
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # 캐시 확인
        for i, text in enumerate(texts):
            text_hash = str(hash(text[:500]))
            if text_hash in self.embedding_cache:
                embeddings.append(self.embedding_cache[text_hash])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 캐시되지 않은 텍스트만 API 호출
        if uncached_texts:
            try:
                logger.info(f"API 호출: {len(uncached_texts)}개 텍스트")
                response = self.openai_client.embeddings.create(
                    model=self.config.embedding_model,
                    input=uncached_texts
                )
                
                # 결과 저장
                for idx, embedding_data in zip(uncached_indices, response.data):
                    embedding = embedding_data.embedding
                    embeddings[idx] = embedding
                    
                    # 캐시 저장
                    text_hash = str(hash(texts[idx][:500]))
                    self.embedding_cache[text_hash] = embedding
                    
            except Exception as e:
                logger.error(f"배치 임베딩 생성 실패: {e}")
        
        return embeddings
    
    def compute_embedding_cosine_similarity(self, text1: str, text2: str) -> float:
        """임베딩 기반 코사인 유사도 계산"""
        if not self.openai_client:
            return 0.0
            
        try:
            # 두 텍스트의 임베딩 얻기
            embedding1 = self.get_embedding(text1)
            embedding2 = self.get_embedding(text2)
            
            if embedding1 is None or embedding2 is None:
                return 0.0
            
            # 코사인 유사도 계산
            embedding1 = np.array(embedding1).reshape(1, -1)
            embedding2 = np.array(embedding2).reshape(1, -1)
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"임베딩 코사인 유사도 계산 실패: {e}")
            return 0.0


class LocalSimilarityEvaluator:
    """로컬 유사도 평가기 - 최적화 버전"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """초기화"""
        self.config = config or EvaluationConfig()
        self.model_size = "1.7B"  # 기본값
        self.output_config = None  # 출력 설정
        
        self.file_matcher = FileMatcher()
        self.text_extractor = TextExtractor()
        self.metrics = SimilarityMetrics(self.config)
        
        # 진행 상황 저장 파일
        self.progress_file = Path("evaluation_progress.json")
        
        logger.info("로컬 유사도 평가기 (최적화 버전) 초기화 완료")
        logger.info(f"샘플 크기: {self.config.sample_size}")
        logger.info(f"배치 크기: {self.config.batch_size}")
        logger.info(f"캐시 사용: {self.config.use_cache}")
    
    def save_progress(self, scores: List[EvaluationScore], current_idx: int, total: int):
        """진행 상황 저장"""
        progress_data = {
            "model_size": self.model_size,
            "current_idx": current_idx,
            "total": total,
            "scores": [s.to_dict() for s in scores],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"진행 상황 저장: {current_idx}/{total}")
    
    def load_progress(self) -> Tuple[List[EvaluationScore], int]:
        """진행 상황 로드"""
        if not self.progress_file.exists():
            return [], 0
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get("model_size") != self.model_size:
                logger.info("다른 모델의 진행 상황 파일. 새로 시작.")
                return [], 0
            
            scores = [
                EvaluationScore(
                    file_name=s['file_name'],
                    tfidf_cosine=s['tfidf_cosine'],
                    embedding_cosine=s['embedding_cosine']
                )
                for s in data['scores']
            ]
            
            logger.info(f"진행 상황 로드: {data['current_idx']}/{data['total']} 완료")
            return scores, data['current_idx']
            
        except Exception as e:
            logger.error(f"진행 상황 로드 실패: {e}")
            return [], 0
    
    def evaluate_batch_optimized(self, matches: List[FileMatch]) -> EvaluationResult:
        """배치 평가 - 최적화 버전"""
        logger.info(f"배치 평가 시작: {len(matches)}개 파일")
        
        # 이전 진행 상황 확인
        scores, start_idx = self.load_progress()
        
        if start_idx > 0:
            logger.info(f"이전 진행 상황에서 재개: {start_idx}부터 시작")
            matches = matches[start_idx:]
        
        # 배치 처리
        for batch_start in range(0, len(matches), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(matches))
            batch_matches = matches[batch_start:batch_end]
            
            logger.info(f"배치 처리 [{batch_start+start_idx+1}-{batch_end+start_idx}/{len(matches)+start_idx}]")
            
            # 텍스트 추출
            gold_texts = []
            result_texts = []
            
            for match in batch_matches:
                if not match.gold_text:
                    match.gold_text = self.text_extractor.extract_gold_text(match.gold_folder)
                if not match.result_text:
                    match.result_text = self.text_extractor.extract_result_text(match.result_file)
                
                gold_texts.append(match.gold_text)
                result_texts.append(match.result_text)
            
            # 배치 임베딩 계산
            all_texts = gold_texts + result_texts
            embeddings = self.metrics.get_batch_embeddings(all_texts)
            
            gold_embeddings = embeddings[:len(gold_texts)]
            result_embeddings = embeddings[len(gold_texts):]
            
            # 유사도 계산
            for i, match in enumerate(batch_matches):
                if not match.gold_text.strip() or not match.result_text.strip():
                    logger.warning(f"빈 텍스트: {match.suffix}")
                    continue
                
                # TF-IDF 유사도
                tfidf_sim = self.metrics.compute_tfidf_cosine_similarity(
                    match.gold_text, 
                    match.result_text
                )
                
                # 임베딩 유사도
                if gold_embeddings[i] and result_embeddings[i]:
                    embedding1 = np.array(gold_embeddings[i]).reshape(1, -1)
                    embedding2 = np.array(result_embeddings[i]).reshape(1, -1)
                    embedding_sim = cosine_similarity(embedding1, embedding2)[0][0]
                else:
                    embedding_sim = 0.0
                
                score = EvaluationScore(
                    file_name=match.gold_folder.name,
                    tfidf_cosine=tfidf_sim,
                    embedding_cosine=float(embedding_sim)
                )
                scores.append(score)
                
                logger.info(
                    f"  {match.gold_folder.name}: "
                    f"TF-IDF={score.tfidf_cosine:.4f}, "
                    f"Embedding={score.embedding_cosine:.4f}"
                )
            
            # 중간 저장
            if (batch_end + start_idx) % self.config.save_interval == 0:
                self.save_progress(scores, batch_end + start_idx, len(matches) + start_idx)
                self.metrics.save_cache()
        
        # 최종 캐시 저장
        self.metrics.save_cache()
        
        return EvaluationResult(
            scores=scores,
            sample_size=len(matches),
            total_available=len(matches)
        )
    
    def evaluate_from_paths(self, gold_base_path: str, result_base_path: str) -> Optional[EvaluationResult]:
        """경로 기반 평가 (최적화 버전)"""
        logger.info("=" * 80)
        logger.info("로컬 회의록 유사도 평가 시작 (최적화 버전)")
        logger.info(f"정답 경로: {gold_base_path}")
        logger.info(f"비교 경로: {result_base_path}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # 파일 매칭
        all_matches = self.file_matcher.find_matches(gold_base_path, result_base_path)
        
        if not all_matches:
            logger.error("매칭되는 파일 없음")
            return None
        
        # 랜덤 샘플링
        sampled_matches = self.file_matcher.random_sample_matches(
            all_matches, 
            self.config.sample_size, 
            self.config.random_seed
        )
        
        # 평가 수행
        result = self.evaluate_batch_optimized(sampled_matches)
        result.total_available = len(all_matches)  # 전체 가능한 파일 수 기록
        
        # 소요 시간
        elapsed_time = time.time() - start_time
        logger.info(f"평가 완료: {elapsed_time:.2f}초 소요")
        
        # 결과 출력
        self._print_results(result)
        
        # 결과 저장
        self._save_results(result)
        
        # 진행 상황 파일 삭제
        if self.progress_file.exists():
            self.progress_file.unlink()
            logger.info("진행 상황 파일 삭제")
        
        return result
    
    def _print_results(self, result: EvaluationResult):
        """결과 출력"""
        print("\n" + "=" * 80)
        print("평가 결과 요약 (최적화 버전)")
        print("=" * 80)
        print(f"전체 가능 파일 수: {result.total_available}")
        print(f"샘플링된 파일 수: {result.sample_size}")
        print(f"실제 평가 파일 수: {len(result.scores)}")
        print(f"샘플링 비율: {(result.sample_size/result.total_available*100):.1f}%")
        print("-" * 80)
        print(f"평균 TF-IDF 코사인 유사도: {result.mean_tfidf_cosine:.4f}")
        print(f"평균 Embedding 코사인 유사도: {result.mean_embedding_cosine:.4f}")
        print("=" * 80)
        
        # 상위/하위 10개 결과
        if len(result.scores) >= 20:
            sorted_tfidf = sorted(result.scores, key=lambda x: x.tfidf_cosine, reverse=True)
            sorted_embedding = sorted(result.scores, key=lambda x: x.embedding_cosine, reverse=True)
            
            print("\n[TF-IDF 기준] 상위 5개 파일:")
            for i, score in enumerate(sorted_tfidf[:5], 1):
                print(f"{i}. {score.file_name[:50]:50s}: TF-IDF={score.tfidf_cosine:.4f}, Embedding={score.embedding_cosine:.4f}")
            
            print("\n[Embedding 기준] 상위 5개 파일:")
            for i, score in enumerate(sorted_embedding[:5], 1):
                print(f"{i}. {score.file_name[:50]:50s}: TF-IDF={score.tfidf_cosine:.4f}, Embedding={score.embedding_cosine:.4f}")
    
    def _save_results(self, result: EvaluationResult):
        """결과 저장"""
        # 출력 파일명 설정
        if hasattr(self, 'output_config') and self.output_config:
            json_file = self.output_config['json_file']
            csv_file = self.output_config['csv_file']
            summary_file = self.output_config['summary_file']
        else:
            # 기본값
            model_size = getattr(self, 'model_size', '1.7B')
            json_file = f"{model_size}_pretrain_similarity_results.json"
            csv_file = f"{model_size}_pretrain_similarity_results.csv"
            summary_file = f"{model_size}_pretrain_similarity_summary.txt"
        
        # JSON 저장
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 결과 저장: {json_file}")
        
        # CSV 형식 저장
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("파일명,TF-IDF_Cosine,Embedding_Cosine\n")
            for score in result.scores:
                f.write(f"{score.file_name},{score.tfidf_cosine:.4f},{score.embedding_cosine:.4f}\n")
        logger.info(f"CSV 결과 저장: {csv_file}")
        
        # 요약 저장
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("로컬 회의록 유사도 평가 요약 (최적화 버전)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"평가 시간: {result.timestamp}\n")
            f.write(f"전체 파일 수: {result.total_available}\n")
            f.write(f"샘플링 크기: {result.sample_size}\n")
            f.write(f"실제 평가 파일 수: {len(result.scores)}\n")
            f.write(f"샘플링 비율: {(result.sample_size/result.total_available*100):.1f}%\n")
            f.write(f"랜덤 시드: {self.config.random_seed}\n\n")
            f.write("평균 점수:\n")
            f.write(f"  - TF-IDF 코사인 유사도: {result.mean_tfidf_cosine:.4f}\n")
            f.write(f"  - Embedding 코사인 유사도: {result.mean_embedding_cosine:.4f}\n")
        
        logger.info(f"요약 저장: {summary_file}")


def load_config(config_path: Optional[str] = None) -> Dict:
    """설정 파일 로드 (YAML 형식)"""
    if config_path is None:
        # 기본 config.yaml 파일 경로
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(base_dir))  # Similarity 상위 폴더
        config_path = os.path.join(os.path.dirname(base_dir), "config.yaml")
    
    if not os.path.exists(config_path):
        # 구버전 JSON 설정 파일 확인
        json_config_path = os.path.join(os.path.dirname(base_dir), "config.json")
        if os.path.exists(json_config_path):
            logger.warning("config.yaml을 찾을 수 없어 config.json을 사용합니다.")
            with open(json_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    return config_data


def get_paths(config_data: Dict, model_size: str = "1.7B", training_type: str = "pre_training") -> Tuple[str, str, Dict]:
    """설정에서 경로 추출 (YAML 형식)"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)  # pre_similarity 상위 폴더
    
    # YAML 설정 구조 확인
    if 'similarity_evaluation' in config_data:
        # 새로운 YAML 구조
        sim_config = config_data['similarity_evaluation'][training_type]
        
        # Gold 데이터 경로
        gold_relative = sim_config['gold_data_path']
        gold_path = os.path.join(parent_dir, gold_relative)
        
        # 모델 결과 경로
        result_relative = sim_config['results'][model_size]
        result_path = os.path.join(parent_dir, result_relative)
        
        # 출력 설정
        output_dir = sim_config['output_dir']
        output_prefix = sim_config['output_prefix']
        output_config = {
            'json_file': os.path.join(output_dir, f"{model_size}_{output_prefix}_similarity_results.json"),
            'csv_file': os.path.join(output_dir, f"{model_size}_{output_prefix}_similarity_results.csv"),
            'summary_file': os.path.join(output_dir, f"{model_size}_{output_prefix}_similarity_summary.txt")
        }
    else:
        # 구버전 JSON 구조 (호환성 유지)
        gold_relative = config_data['paths']['gold_standard_data']
        gold_path = os.path.join(parent_dir, gold_relative.replace('../', '').replace('./', ''))
        
        if training_type == "pre_training":
            result_relative = config_data['paths']['pre_training'][model_size]
            result_path = os.path.join(parent_dir, result_relative.replace('./', ''))
            output_config = config_data['output']['pre_training'][model_size]
        else:
            result_relative = config_data['paths']['post_training'][model_size]
            result_path = os.path.join(parent_dir, result_relative.replace('./', ''))
            output_config = config_data['output']['post_training'][model_size]
    
    # 절대 경로로 변환
    gold_path = os.path.abspath(gold_path)
    result_path = os.path.abspath(result_path)
    
    return gold_path, result_path, output_config


def main(model_size: str = "1.7B", config_path: Optional[str] = None):
    """메인 실행 함수"""
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='학습 전 모델 유사도 평가 (최적화 버전)')
    parser.add_argument('--model', default=model_size, choices=['1.7B', '4B', '8B'],
                        help='평가할 모델 크기')
    parser.add_argument('--config', default=config_path, help='설정 파일 경로')
    parser.add_argument('--batch-size', type=int, default=None, help='배치 크기 (기본값: 설정 파일에서 읽기)')
    parser.add_argument('--no-cache', action='store_true', help='캐시 비활성화')
    args = parser.parse_args()
    
    # 설정 로드
    config_data = load_config(args.config)
    
    # 평가 설정 추출 (YAML 구조 확인)
    if 'similarity_evaluation' in config_data:
        # 새로운 YAML 구조
        eval_config = config_data['similarity_evaluation']['evaluation_params']
        batch_size = args.batch_size if args.batch_size else eval_config.get('batch_size', 10)
        config = EvaluationConfig(
            sample_size=eval_config['sample_size'],
            random_seed=eval_config['random_seed'],
            tfidf_max_features=eval_config['tfidf_max_features'],
            tfidf_ngram_range=tuple(eval_config['tfidf_ngram_range']),
            embedding_model=eval_config['embedding_model'],
            batch_size=batch_size,
            save_interval=eval_config.get('save_interval', 20),
            use_cache=not args.no_cache if args.no_cache else eval_config.get('use_cache', True)
        )
    else:
        # 구버전 JSON 구조 (호환성 유지)
        eval_config = config_data['evaluation']
        config = EvaluationConfig(
            sample_size=eval_config['sample_size'],
            random_seed=eval_config['random_seed'],
            tfidf_max_features=eval_config['tfidf_max_features'],
            tfidf_ngram_range=tuple(eval_config['tfidf_ngram_range']),
            embedding_model=eval_config['embedding_model'],
            batch_size=args.batch_size if args.batch_size else 10,
            use_cache=not args.no_cache
        )
    
    # 평가기 초기화
    evaluator = LocalSimilarityEvaluator(config)
    evaluator.model_size = args.model
    
    # 경로 가져오기
    gold_path, result_path, output_config = get_paths(config_data, args.model, "pre_training")
    
    # 출력 디렉토리 생성
    if 'json_file' in output_config:
        output_dir = os.path.dirname(output_config['json_file'])
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"출력 디렉토리 생성: {output_dir}")
    
    evaluator.output_config = output_config
    
    print(f"\n평가 시작: {args.model} 모델 (학습 전)")
    print(f"Gold 데이터: {gold_path}")
    print(f"모델 결과: {result_path}")
    print(f"배치 크기: {args.batch_size}")
    print(f"캐시 사용: {not args.no_cache}")
    
    result = evaluator.evaluate_from_paths(gold_path, result_path)
    
    if result:
        print("\n[OK] 평가 완료!")
        print(f"결과 파일:")
        print(f"  - {output_config['json_file']} (상세 결과)")
        print(f"  - {output_config['csv_file']} (표 형식)")
        print(f"  - {output_config['summary_file']} (요약)")
    else:
        print("\n[FAIL] 평가 실패")


if __name__ == "__main__":
    main()