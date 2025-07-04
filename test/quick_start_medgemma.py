from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

# 1. 모델 경로 (로컬 경로)
model_id = "/home/mts/jks/medgemma/medgemma-4b-it"

# 2. 모델 로드
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# 3. 이미지 불러오기 (로컬 이미지 경로 사용)
image_path = "/home/mts/jks/medgemma/medgemma_reg2025/notebooks/data/thumbnail_pngs_1000/PIT_01_00003_01.png"
image_path2 = "/home/mts/jks/medgemma/medgemma_reg2025/notebooks/data/thumbnail_pngs_1000/PIT_01_00004_01.png"
image = Image.open(image_path2)

# 4. 메시지 구성
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are a pathologist AI assistant trained to analyze whole slide images (WSI). "
                    "You interpret the histopathological content and generate concise pathology reports.\n\n"
                    "Please follow this structure:\n"
                    "- Line 1: [Organ/tissue], [procedure];\\n\n"
                    "- Line 2~: Findings (numbered if multiple)\n"
                    "- Notes (optional): add after double newline (\\n\\n), starting with 'Note)'\n"
                    "- For benign findings, use: 'No tumor present'\n\n"
                    "Example:\n"
                    "Urinary bladder, transurethral resection;\n"
                    "1. Non-invasive papillary urothelial carcinoma, high grade\n"
                    "2. Urothelial carcinoma in situ\n\n"
                    "Note) The specimen includes muscle proper."
                )
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please provide a pathology report for this H&E stained slide."},
            {"type": "image", "image": image}
        ]
    }
]

# 5. 입력 처리
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

# 6. 텍스트 생성
input_len = inputs["input_ids"].shape[-1]
with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
    generation = generation[0][input_len:]

# 7. 디코딩 및 출력
decoded = processor.decode(generation, skip_special_tokens=True)
print("🔍 Generated Report:\n", decoded)
