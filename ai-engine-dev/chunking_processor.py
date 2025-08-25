"""
TtalKkak 청킹 프로세서
Qwen3-32B AWQ 모델 (32,768 토큰 제한) 대응
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import time

logger = logging.getLogger(__name__)

class TtalKkakChunkingProcessor:
    """TtalKkak 전용 토큰 기반 청킹 프로세서"""
    
    def __init__(self, max_context_tokens: int = 32768):
        self.max_context_tokens = max_context_tokens
        # 안전 마진 (시스템 프롬프트, 스키마, 출력 토큰 고려)
        self.safety_margin = 4000
        self.max_input_tokens = max_context_tokens - self.safety_margin
        self.overlap_tokens = 200  # 청크 간 겹침
        
        logger.info(f"🔧 청킹 프로세서 초기화 - 최대 입력 토큰: {self.max_input_tokens}")
    
    def estimate_tokens(self, text: str) -> int:
        """한국어 텍스트의 토큰 수 추정"""
        if not text:
            return 0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other_chars = len(text) - korean_chars - sum(len(word) for word in re.findall(r'[a-zA-Z]+', text))
        
        estimated_tokens = int(
            korean_chars * 1.5 + 
            english_words * 1.3 + 
            other_chars * 1.0
        )
        
        return estimated_tokens
    
    def split_by_sentences(self, text: str) -> List[str]:
        """문장 단위로 텍스트 분할"""
        # 한국어 문장 끝 패턴
        sentence_endings = r'[.!?。！？]\s*'
        sentences = re.split(sentence_endings, text)
        
        # 빈 문장 제거 및 정리
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def create_chunks_with_overlap(self, text: str) -> List[Dict[str, Any]]:
        """겹침을 포함한 청킹"""
        if self.estimate_tokens(text) <= self.max_input_tokens:
            return [{
                "chunk_id": 0,
                "text": text,
                "estimated_tokens": self.estimate_tokens(text),
                "start_sentence": 0,
                "end_sentence": -1,
                "has_overlap": False
            }]
        
        sentences = self.split_by_sentences(text)
        chunks = []
        current_chunk_sentences = []
        current_tokens = 0
        chunk_id = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.estimate_tokens(sentence)
            
            # 단일 문장이 너무 긴 경우 강제 분할
            if sentence_tokens > self.max_input_tokens:
                if current_chunk_sentences:
                    # 현재 청크 저장
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": " ".join(current_chunk_sentences),
                        "estimated_tokens": current_tokens,
                        "start_sentence": i - len(current_chunk_sentences),
                        "end_sentence": i - 1,
                        "has_overlap": chunk_id > 0
                    })
                    chunk_id += 1
                    current_chunk_sentences = []
                    current_tokens = 0
                
                # 긴 문장을 글자 단위로 분할
                long_sentence_chunks = self._split_long_sentence(sentence, chunk_id)
                chunks.extend(long_sentence_chunks)
                chunk_id += len(long_sentence_chunks)
                i += 1
                continue
            
            # 현재 청크에 추가 가능한지 확인
            if current_tokens + sentence_tokens <= self.max_input_tokens:
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens
                i += 1
            else:
                # 현재 청크 저장
                if current_chunk_sentences:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": " ".join(current_chunk_sentences),
                        "estimated_tokens": current_tokens,
                        "start_sentence": i - len(current_chunk_sentences),
                        "end_sentence": i - 1,
                        "has_overlap": chunk_id > 0
                    })
                    chunk_id += 1
                
                # 겹침 처리: 마지막 몇 문장을 다음 청크에 포함
                overlap_sentences = self._get_overlap_sentences(current_chunk_sentences)
                current_chunk_sentences = overlap_sentences + [sentence]
                current_tokens = sum(self.estimate_tokens(s) for s in current_chunk_sentences)
                i += 1
        
        # 마지막 청크 처리
        if current_chunk_sentences:
            chunks.append({
                "chunk_id": chunk_id,
                "text": " ".join(current_chunk_sentences),
                "estimated_tokens": current_tokens,
                "start_sentence": len(sentences) - len(current_chunk_sentences),
                "end_sentence": len(sentences) - 1,
                "has_overlap": chunk_id > 0
            })
        
        logger.info(f"📊 청킹 완료: {len(chunks)}개 청크 생성")
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """겹침용 문장들 선택"""
        if not sentences:
            return []
        
        # 뒤에서부터 overlap_tokens만큼 선택
        overlap_sentences = []
        tokens_count = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = self.estimate_tokens(sentence)
            if tokens_count + sentence_tokens <= self.overlap_tokens:
                overlap_sentences.insert(0, sentence)
                tokens_count += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _split_long_sentence(self, sentence: str, start_chunk_id: int) -> List[Dict[str, Any]]:
        """너무 긴 문장을 강제로 분할"""
        max_chars = int(self.max_input_tokens / 1.5)  # 한글 기준
        chunks = []
        
        for i in range(0, len(sentence), max_chars):
            chunk_text = sentence[i:i + max_chars]
            chunks.append({
                "chunk_id": start_chunk_id + i // max_chars,
                "text": chunk_text,
                "estimated_tokens": self.estimate_tokens(chunk_text),
                "start_sentence": -1,  # 문장 내 분할
                "end_sentence": -1,
                "has_overlap": False,
                "is_sentence_split": True
            })
        
        return chunks
    
    def merge_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """청크별 결과를 통합"""
        if not chunk_results:
            return {"error": "No chunk results to merge"}
        
        # 단일 청크인 경우
        if len(chunk_results) == 1:
            return chunk_results[0]
        
        # 여러 청크 통합
        merged_result = {
            "action_items": [],
            "decisions": [],
            "key_points": [],
            "next_steps": [],
            "participants": set(),
            "summary": "",
            "metadata": {
                "total_chunks": len(chunk_results),
                "processing_method": "chunked"
            }
        }
        
        # 각 청크 결과 통합
        for i, result in enumerate(chunk_results):
            if isinstance(result, dict):
                # 액션 아이템 통합
                if "action_items" in result and isinstance(result["action_items"], list):
                    for item in result["action_items"]:
                        if isinstance(item, dict):
                            item["source_chunk"] = i
                            merged_result["action_items"].append(item)
                
                # 기타 필드 통합
                for field in ["decisions", "key_points", "next_steps"]:
                    if field in result and isinstance(result[field], list):
                        merged_result[field].extend(result[field])
                
                # 참석자 통합
                if "participants" in result and isinstance(result["participants"], list):
                    merged_result["participants"].update(result["participants"])
        
        # 중복 제거
        merged_result["decisions"] = list(set(merged_result["decisions"]))
        merged_result["key_points"] = list(set(merged_result["key_points"]))
        merged_result["next_steps"] = list(set(merged_result["next_steps"]))
        merged_result["participants"] = list(merged_result["participants"])
        
        # 통합 요약 생성
        merged_result["summary"] = self._generate_merged_summary(chunk_results)
        
        # 액션 아이템 중복 제거 및 우선순위 재정렬
        merged_result["action_items"] = self._deduplicate_action_items(
            merged_result["action_items"]
        )
        
        return merged_result
    
    def _generate_merged_summary(self, chunk_results: List[Dict[str, Any]]) -> str:
        """청크별 요약을 통합하여 전체 요약 생성"""
        summaries = []
        for result in chunk_results:
            if isinstance(result, dict) and "summary" in result:
                summary = result["summary"]
                if summary and isinstance(summary, str):
                    summaries.append(summary.strip())
        
        if not summaries:
            return "회의 내용을 분석하여 주요 사항들을 정리했습니다."
        
        # 간단한 통합 (실제로는 더 정교한 LLM 기반 통합이 필요할 수 있음)
        return " ".join(summaries)
    
    def _deduplicate_action_items(self, action_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """액션 아이템 중복 제거"""
        if not action_items:
            return []
        
        seen_tasks = set()
        deduplicated = []
        
        for item in action_items:
            if isinstance(item, dict) and "task" in item:
                task_key = item["task"].lower().strip()
                if task_key not in seen_tasks:
                    seen_tasks.add(task_key)
                    deduplicated.append(item)
        
        # 우선순위별 정렬
        priority_order = {"high": 0, "medium": 1, "low": 2}
        deduplicated.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 1))
        
        return deduplicated

# 전역 인스턴스
_chunking_processor = None

def get_chunking_processor(max_context_tokens: int = 32768) -> TtalKkakChunkingProcessor:
    """전역 청킹 프로세서 인스턴스 반환"""
    global _chunking_processor
    if _chunking_processor is None:
        _chunking_processor = TtalKkakChunkingProcessor(max_context_tokens)
    return _chunking_processor 