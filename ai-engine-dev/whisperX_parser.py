
import json
from typing import List, Dict
from datetime import timedelta
import re

#시간 변환 함수
def seconds_to_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)  # 초를 timedelta 객체로 변환
    # str(td) → "1:01:01.500000" 형태
    # .split('.')[0] → 소수점 앞 부분만 추출하여 밀리초 제거
    # .zfill(8) → 8자리로 맞춤 (예: "1:01:01" → "01:01:01")
    return str(td).split('.')[0].zfill(8)

#문장 부호
def split_sentences(text: str) -> List[str]: # 문장부호 뒤에 공백 정규화: 문장부호 뒤에 무조건 1칸 띄우기
    text = re.sub(r'([.?!])(?!\s)', r'\1 ', text)  # 문장부호 뒤에 공백 없으면 추가
    text = re.sub(r'([.?!])\s{2,}', r'\1 ', text)  # 문장부호 뒤 2칸 이상 공백을 1칸으로
    
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text.strip()) if s.strip()]

#추출한 음성 정렬 후 반환
def parse_whisperx_json(json_path: str) -> List[Dict]:
    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # segments 배열 추출 (WhisperX 출력의 핵심 데이터)
    segments = data.get("segments", [])
    entries = []
    global_index = 1  # 전체 세그먼트 순번
    
    # 각 세그먼트 처리
    for seg_index, seg in enumerate(segments, 1):
        speaker = seg.get("speaker", "UNKNOWN")  # 화자 정보
        text = seg.get("text", "").strip()       # 발화 텍스트
        start_time = seg.get("start", 0.0)       # 시작 시간(초)
        timestamp = seconds_to_timestamp(start_time)  # HH:MM:SS 형태로 변환
        
        # 긴 발화를 문장 단위로 분리
        sentences = split_sentences(text)
        for i, sent in enumerate(sentences, 1):
            entries.append({
                "timestamp": timestamp,                    # 시작 시간
                "timestamp_order": f"{global_index}-{i}",  # 세그먼트-문장 순번
                "speaker": speaker,                        # 화자
                "text": sent                              # 문장 텍스트
            })
        global_index += 1  # 다음 세그먼트로
        
    return entries

'''
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python whisperx_parser.py <json_file>")
        sys.exit(1)
    
    entries = parse_whisperx_json(sys.argv[1])
    print(f"Parsed {len(entries)} entries")
    for entry in entries[:3]:  # Show first 3 entries
        print(f"[{entry['timestamp']}] ({entry['timestamp_order']}) {entry['speaker']}: {entry['text']}")
'''