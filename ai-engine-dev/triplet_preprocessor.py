import json
from typing import List, Dict


def preprocess_triplets(triplets_with_labels: List[Dict], log_file_path: str = None) -> List[Dict]:
    """
    라벨 기반 triplet 분기 처리
    - label 0: 메모리에 저장하여 반환 (label 필드 제거)
    - label 1: 로그파일에 저장
    
    Args:
        triplets_with_labels: label이 0 또는 1로 설정된 triplet 딕셔너리 리스트
        log_file_path: label 1 항목들의 로그 파일 경로 (선택)
        
    Returns:
        List[Dict]: label 0 항목들 (timestamp, timestamp_order, speaker, text만 포함)
    """
    label_0_items = []
    label_1_items = []
    
    # 라벨별로 분류
    for triplet in triplets_with_labels:
        if triplet["label"] == 0:
            # label 0: label 필드 제거하여 저장
            item = {
                "timestamp": triplet.get("timestamp", ""),
                "timestamp_order": triplet["timestamp_order"],
                "speaker": triplet["speaker"],
                "text": triplet["target"]  # [TGT] [/TGT] 태그 그대로 유지
            }
            label_0_items.append(item)
            
        elif triplet["label"] == 1:
            # label 1: 로그용 형태로 저장
            item = {
                "timestamp": triplet.get("timestamp", ""),
                "timestamp_order": triplet["timestamp_order"],
                "speaker": triplet["speaker"],
                "text": triplet["target"],
                "label": triplet["label"]
            }
            label_1_items.append(item)
    
    # label 1 항목들을 로그 파일에 저장
    if log_file_path and label_1_items:
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            for item in label_1_items:
                log_file.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return label_0_items

'''
# 사용 예시
if __name__ == "__main__":
    # 테스트 데이터
    test_triplets = [
        {
            "timestamp": "2024-01-01 10:00:00",
            "timestamp_order": 1,
            "speaker": "A",
            "prev": "이전 맥락",
            "target": "[TGT] 현재 발화 내용 [/TGT]",
            "next": "다음 맥락",
            "label": 0
        },
        {
            "timestamp": "2024-01-01 10:01:00",
            "timestamp_order": 2,
            "speaker": "B",
            "prev": "이전 맥락2",
            "target": "[TGT] 다른 발화 내용 [/TGT]",
            "next": "다음 맥락2",
            "label": 1
        }
    ]
    
    # 함수 사용
    label_0_items = preprocess_triplets(test_triplets, log_file_path="label_1_log.jsonl")
    
    print("Label 0 항목 수:", len(label_0_items))
    print("Label 0 첫 번째 항목:", label_0_items[0] if label_0_items else "없음")

'''