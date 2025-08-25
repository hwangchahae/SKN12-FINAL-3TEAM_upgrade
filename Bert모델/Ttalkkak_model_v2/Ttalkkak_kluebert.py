from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import torch

# config 로드
config    = AutoConfig.from_pretrained(".", num_labels=2)

# 모델 아키텍처 생성 후 .pt 로드
model = AutoModelForSequenceClassification.from_config(config)
state_dict = torch.load("Ttalkkak_model_v2.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(".")

# 예측
# inputs = tokenizer("테스트 문장", return_tensors="pt", padding=True, truncation=True)
# with torch.no_grad():
#     logits = model(**inputs).logits
# print("예측 라벨:", torch.argmax(logits, dim=-1).item())