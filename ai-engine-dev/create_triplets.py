from typing import List, Dict
from natsort import natsorted
from whisperX_parser import parse_whisperx_json

# WhisperX 결과 데이터를 triplet 구조로 변환하는 함수
def create_structured_triplets(data: List[Dict]) -> List[Dict]:
    speech_data = [d for d in data if "text" in d] # 실제 발화 데이터만 필터링 (타이틀, 참석자 정보 등 제외)
    sorted_data = natsorted(speech_data, key=lambda x: x["timestamp_order"]) # timestamp_order 기준으로 자연스러운 순서로 정렬   
    result = []
    # 각 발화에 대해 triplet 구조 생성
    for i in range(len(sorted_data)):
        item = sorted_data[i]
        
        # 이전 맥락: 현재 발화 기준 이전 2개 발화 텍스트 결합
        prev_1 = sorted_data[i - 2]["text"] if i - 2 >= 0 else ""
        prev_2 = sorted_data[i - 1]["text"] if i - 1 >= 0 else ""
        prev = f"{prev_1} {prev_2}".strip()
        
        # 다음 맥락: 현재 발화 기준 다음 2개 발화 텍스트 결합
        next_1 = sorted_data[i + 1]["text"] if i + 1 < len(sorted_data) else ""
        next_2 = sorted_data[i + 2]["text"] if i + 2 < len(sorted_data) else ""
        next_ = f"{next_1} {next_2}".strip()
        
        # triplet 구조 데이터 생성
        result.append({
            "timestamp": item.get("timestamp", ""),           # 발화 시간
            "timestamp_order": item["timestamp_order"],       # 시간 순서
            "speaker": item["speaker"],                       # 화자
            "prev": prev,                                     # 이전 맥락 (2개 발화)
            "target": f"[TGT] {item['text']} [/TGT]",        # 현재 발화 (타겟 마킹)
            "next": next_,                                    # 다음 맥락 (2개 발화)
            "label": None                                     # 라벨 (추후 사용)
        })
    
    return result