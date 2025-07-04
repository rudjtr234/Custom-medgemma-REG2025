from datasets import load_dataset
from typing import Any
import json

# Step 1: 라벨 정의
TISSUE_CLASSES = [
    "A: adipose",
    "B: background",
    "C: debris",
    "D: lymphocytes",
    "E: mucus",
    "F: smooth muscle",
    "G: normal colon mucosa",
    "H: cancer-associated stroma",
    "I: colorectal adenocarcinoma epithelium"
]
LABEL_INDEX = 8  # colorectal adenocarcinoma epithelium
options = "\n".join(TISSUE_CLASSES)
PROMPT = f"What is the most likely tissue type shown in the histopathology image?\n{options}"

# Step 2: 포맷 함수 정의
def format_data(example: dict[str, Any]) -> dict[str, Any]:
    example["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": example["image"],  # 파일 경로로 처리됨
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": TISSUE_CLASSES[example["label"]],
                },
            ],
        },
    ]
    return example

# Step 3: 데이터 로딩 및 라벨 부여
data_dir = "/home/mts/jks/medgemma/medgemma_reg2025/notebooks/data/thumbnail_pngs_1000"
dataset = load_dataset("imagefolder", data_dir=data_dir)

# 단일 클래스이므로 정수 라벨 8번으로 통일
dataset = dataset["train"].map(lambda x: {**x, "label": LABEL_INDEX})

# Step 4: 포맷 적용
dataset = dataset.map(format_data)

# Step 5: 전체를 JSON 형식으로 저장 (리스트 형태)
output_path = "breast_medgemma_messages.json"
json_data = [{"messages": example["messages"]} for example in dataset]

with open(output_path, "w") as f:
    json.dump(json_data, f, indent=2)

print(f"✅ JSON 저장 완료 (list 형식): {output_path}")
