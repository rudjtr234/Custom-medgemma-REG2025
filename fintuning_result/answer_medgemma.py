from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from peft import PeftModel
import torch, os

# ── 0. (RTX 40 시리즈 드라이버 경고 방지, 필요 없으면 생략) ──────────
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# ── 경로 지정 ─────────────────────────────────
base_id    = "/home/mts/jks/medgemma/medgemma-4b-it"
adapter_id = "/home/mts/jks/medgemma/medgemma_reg2025/notebooks/fintuning_model/medgemma-REG_2025_06_16"
image_path = "/home/mts/jks/medgemma/medgemma_reg2025/notebooks/data/thumbnail_pngs_1000/PIT_01_02994_01.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 모델 + 어댑터 로드 ────────────────────────
model = AutoModelForImageTextToText.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_id, torch_dtype=torch.bfloat16)
#model = model.to(device)

# ── 프로세서 로드 (어댑터 경로에 토크나이저 변경사항이 있다면 adapter_id 사용) ──
processor = AutoProcessor.from_pretrained(base_id)

# ── 2. 테스트 이미지 경로 ───────────────────────────────────────────
image_path = "/home/mts/jks/medgemma/medgemma_reg2025/notebooks/data/thumbnail_pngs_1000/PIT_01_00004_01.png"

# ── 3. 메시지(ChatML) 구성 ──────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a pathologist AI assistant trained to analyze whole slide images (WSI). "
    "You interpret the histopathological content and generate concise pathology reports.\n\n"
    "Please follow this structure:\n"
    "- Line 1: [Organ/tissue], [procedure];\n"
    "- Line 2: Findings (numbered if multiple)\n"
    "- Notes (optional): add after double newline (\\n\\n), starting with 'Note)'\n"
    "- For benign findings, use: 'No tumor present'\n\n"
)

messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user",   "content": [
        {"type": "text",  "text": "Please provide a pathology report for this H&E-stained slide."},
        {"type": "image", "image": image_path}            # ← PIL 대신 경로 문자열
    ]},
]

prompt_str = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False).strip()

# (1) 이미지 리스트, 두 겹 -> 한 겹
inputs = processor(
    text   = [prompt_str],
    images = [Image.open(image_path).convert("RGB")],   # 한 겹!
    return_tensors = "pt",
    padding = True,
).to("cuda:0", dtype=torch.bfloat16)                   # 단일 GPU 기준

# (2) 어댑터가 실제로 활성화돼 있는지 체크
print("active adapters:", model.active_adapters)       # ['default'] 가 아니면
model.set_adapter("default")

# (3) 디버그용: prompt·pixel_values 확인
print("prompt 앞부분:", prompt_str[:200])
print("pixel_values.shape:", inputs["pixel_values"].shape)  # (1, 3, 224, 224) ?

model.generation_config.do_sample = False
model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
processor.tokenizer.padding_side = "left"

with torch.inference_mode():
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,        # 샘플링 켜기
        temperature=0.7,
        top_p=0.9,
    )

generated = gen_ids[0, inputs["input_ids"].shape[-1]:]
report = processor.decode(generated, skip_special_tokens=True)
print("\n🔍 Generated Report:\n", report)
