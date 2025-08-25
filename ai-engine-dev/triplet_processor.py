"""
TtalKkak Triplet 프로세서
WhisperX 결과를 Triplet 구조로 변환하고 BERT 분류를 수행하는 통합 모듈
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
import tempfile

# 프로젝트 루트의 triplet 모듈들 임포트
sys.path.append('..')

try:
    from whisperX_parser import parse_whisperx_json
    from create_triplets import create_structured_triplets
    from triplet_preprocessor import preprocess_triplets
except ImportError as e:
    logging.warning(f"⚠️ Triplet 모듈 임포트 실패: {e}")
    logging.info("💡 루트 디렉토리의 triplet 파일들을 확인해주세요")

from bert_classifier import get_bert_classifier

logger = logging.getLogger(__name__)

class TripletProcessor:
    """
    WhisperX → Triplet → BERT → 필터링 통합 처리기
    """
    
    def __init__(self):
        self.bert_classifier = None
        logger.info("🔧 Triplet 프로세서 초기화")
    
    def _ensure_bert_classifier(self):
        """BERT 분류기 지연 로딩"""
        if self.bert_classifier is None:
            self.bert_classifier = get_bert_classifier()
    
    def whisperx_to_triplets(self, whisperx_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        WhisperX 결과를 Triplet 구조로 변환
        """
        try:
            logger.info("🔄 WhisperX 결과를 Triplet으로 변환 중...")
            
            # WhisperX 세그먼트를 구조화된 데이터로 변환
            segments = whisperx_result.get("segments", [])
            
            # parse_whisperx_json 대신 직접 변환 (메모리 상의 데이터 처리)
            structured_data = []
            
            for i, segment in enumerate(segments):
                # 화자 정보가 있으면 사용, 없으면 기본값
                speaker = segment.get("speaker", f"SPEAKER_{i%3:02d}")
                text = segment.get("text", "").strip()
                start_time = segment.get("start", 0.0)
                
                # 시간 변환 (초 → HH:MM:SS)
                hours = int(start_time // 3600)
                minutes = int((start_time % 3600) // 60)
                seconds = int(start_time % 60)
                timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                structured_data.append({
                    "timestamp": timestamp,
                    "timestamp_order": f"{i+1}-1",
                    "speaker": speaker,
                    "text": text
                })
            
            # Triplet 구조 생성
            triplets = create_structured_triplets(structured_data)
            
            logger.info(f"✅ Triplet 변환 완료: {len(structured_data)} → {len(triplets)}개 Triplet")
            
            return triplets
            
        except Exception as e:
            logger.error(f"❌ Triplet 변환 실패: {e}")
            # 기본 구조로 대체
            return self._create_fallback_triplets(whisperx_result)
    
    def _create_fallback_triplets(self, whisperx_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Triplet 변환 실패시 기본 구조 생성"""
        logger.warning("⚠️ 기본 Triplet 구조로 대체")
        
        segments = whisperx_result.get("segments", [])
        triplets = []
        
        for i, segment in enumerate(segments):
            triplet = {
                "timestamp": f"00:00:{i:02d}",
                "timestamp_order": f"{i+1}-1",
                "speaker": segment.get("speaker", "UNKNOWN"),
                "prev": "",
                "target": f"[TGT] {segment.get('text', '')} [/TGT]",
                "next": "",
                "label": None
            }
            triplets.append(triplet)
        
        return triplets
    
    def classify_triplets(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        BERT 모델을 사용한 Triplet 분류
        """
        if not triplets:
            return []
        
        try:
            logger.info(f"🧠 BERT 분류 시작: {len(triplets)}개 Triplet")
            
            # BERT 분류기 로딩
            self._ensure_bert_classifier()
            
            # 배치 분류 수행
            classified_triplets = self.bert_classifier.classify_triplets_batch(triplets)
            
            return classified_triplets
            
        except Exception as e:
            logger.error(f"❌ BERT 분류 실패: {e}")
            # 실패시 모든 발화를 중요한 것으로 분류
            for triplet in triplets:
                triplet["label"] = 0  # 중요한 발화
                triplet["confidence"] = 0.5
            return triplets
    
    def filter_important_triplets(
        self, 
        classified_triplets: List[Dict[str, Any]], 
        save_noise_log: bool = True
    ) -> List[Dict[str, Any]]:
        """
        분류된 Triplet에서 중요한 발화만 필터링
        """
        try:
            logger.info("🧹 중요한 발화 필터링 중...")
            
            # 로그 파일 경로 설정
            log_file_path = None
            if save_noise_log:
                log_dir = "logs"
                os.makedirs(log_dir, exist_ok=True)
                log_file_path = os.path.join(log_dir, "noise_triplets.jsonl")
            
            # triplet_preprocessor 사용 (사용 가능한 경우)
            try:
                filtered_triplets = preprocess_triplets(classified_triplets, log_file_path)
                logger.info(f"✅ 필터링 완료: {len(classified_triplets)} → {len(filtered_triplets)}개 유지")
                return filtered_triplets
                
            except Exception as e:
                logger.warning(f"⚠️ triplet_preprocessor 사용 실패: {e}")
                return self._manual_filter(classified_triplets, log_file_path)
                
        except Exception as e:
            logger.error(f"❌ 필터링 실패: {e}")
            return classified_triplets
    
    def _manual_filter(
        self, 
        classified_triplets: List[Dict[str, Any]], 
        log_file_path: Optional[str]
    ) -> List[Dict[str, Any]]:
        """수동 필터링 (백업용)"""
        important_triplets = []
        noise_triplets = []
        
        for triplet in classified_triplets:
            if triplet.get("label") == 0:  # 중요한 발화
                # label 필드 제거하여 저장
                filtered_triplet = {
                    "timestamp": triplet.get("timestamp", ""),
                    "timestamp_order": triplet.get("timestamp_order", ""),
                    "speaker": triplet.get("speaker", ""),
                    "text": triplet.get("target", "")
                }
                important_triplets.append(filtered_triplet)
            else:  # 노이즈 발화
                noise_triplets.append(triplet)
        
        # 노이즈 로그 저장
        if log_file_path and noise_triplets:
            try:
                with open(log_file_path, 'w', encoding='utf-8') as f:
                    for item in noise_triplets:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                logger.info(f"📝 노이즈 로그 저장: {log_file_path}")
            except Exception as e:
                logger.warning(f"⚠️ 노이즈 로그 저장 실패: {e}")
        
        return important_triplets
    
    def process_whisperx_result(
        self, 
        whisperx_result: Dict[str, Any],
        enable_bert_filtering: bool = True,
        save_noise_log: bool = True
    ) -> Dict[str, Any]:
        """
        WhisperX 결과를 전체 Triplet 파이프라인으로 처리
        """
        try:
            logger.info("🚀 Triplet 파이프라인 처리 시작")
            
            # 1. WhisperX → Triplet 변환
            triplets = self.whisperx_to_triplets(whisperx_result)
            
            # 원본 텍스트 추출
            original_text = whisperx_result.get("full_text", "")
            if not original_text:
                segments = whisperx_result.get("segments", [])
                original_text = " ".join([seg.get("text", "") for seg in segments])
            
            # BERT 필터링이 비활성화된 경우
            if not enable_bert_filtering:
                logger.info("⚠️ BERT 필터링 비활성화, 원본 텍스트 반환")
                return {
                    "success": True,
                    "original_transcript": original_text,
                    "filtered_transcript": original_text,
                    "triplet_data": {
                        "triplets": triplets,
                        "conversation_segments": triplets,
                        "statistics": {
                            "total_triplets": len(triplets),
                            "filtered_triplets": len(triplets),
                            "conversation_segments": len(triplets),
                            "speakers": list(set([t.get("speaker", "") for t in triplets])),
                            "total_duration": 0,
                            "average_context_quality": 1.0
                        }
                    },
                    "processing_stats": {
                        "processing_time": 0,
                        "total_segments": len(triplets),
                        "total_triplets": len(triplets),
                        "conversation_segments": len(triplets)
                    }
                }
            
            # 2. BERT 분류
            classified_triplets = self.classify_triplets(triplets)
            
            # 3. 중요 발화 필터링
            filtered_triplets = self.filter_important_triplets(
                classified_triplets, 
                save_noise_log=save_noise_log
            )
            
            # 4. 필터링된 텍스트 재구성
            filtered_text = " ".join([
                triplet["text"].replace("[TGT]", "").replace("[/TGT]", "").strip()
                for triplet in filtered_triplets
            ])
            
            # 5. 통계 정보 생성
            classification_stats = self.bert_classifier.get_classification_stats(classified_triplets)
            
            # 6. 결과 반환
            result = {
                "success": True,
                "original_transcript": original_text,
                "filtered_transcript": filtered_text,
                "triplet_data": {
                    "triplets": classified_triplets,
                    "conversation_segments": filtered_triplets,
                    "statistics": {
                        "total_triplets": len(triplets),
                        "filtered_triplets": len(filtered_triplets),
                        "conversation_segments": len(filtered_triplets),
                        "speakers": list(set([t.get("speaker", "") for t in triplets])),
                        "total_duration": 0,  # TODO: 실제 계산
                        "average_context_quality": classification_stats.get("avg_confidence", 0.5)
                    }
                },
                "classification_stats": classification_stats,
                "processing_stats": {
                    "processing_time": 0,  # TODO: 실제 측정
                    "total_segments": len(whisperx_result.get("segments", [])),
                    "total_triplets": len(triplets),
                    "conversation_segments": len(filtered_triplets)
                }
            }
            
            logger.info("✅ Triplet 파이프라인 처리 완료")
            logger.info(f"📊 결과: {len(triplets)} → {len(filtered_triplets)}개 발화 (노이즈 {classification_stats.get('noise_reduction_ratio', 0)*100:.1f}% 제거)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Triplet 파이프라인 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_transcript": whisperx_result.get("full_text", ""),
                "filtered_transcript": whisperx_result.get("full_text", "")
            }

# 전역 인스턴스
triplet_processor = None

def get_triplet_processor() -> TripletProcessor:
    """Triplet 프로세서 싱글톤 인스턴스 반환"""
    global triplet_processor
    
    if triplet_processor is None:
        triplet_processor = TripletProcessor()
    
    return triplet_processor