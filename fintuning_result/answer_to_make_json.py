
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, json, gc
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import os
import glob
import json
import gc
import re


def clean_report(text):
    # 마크업 제거
    text = re.sub(r'<[^>]+>', '', text)

    # \\n → 실제 줄바꿈
    text = text.replace('\\n\\n', '\n\n').replace('\\n', '\n')

    # n/a 제거
    text = re.sub(r'\b[nN]/?[aA]\b', '', text)

    # 반복 문장 제거
    lines = text.split('\n')
    seen = set()
    new_lines = []
    for line in lines:
        line = line.strip()
        line = re.sub(r'^#*\s*(Human|Assistant|Response|Report|Example)\s*[:：]?\s*', '', line, flags=re.IGNORECASE)
        if line and line not in seen:
            seen.add(line)
            new_lines.append(line)

    # No tumor present. 여러 번 나오면 하나만
    filtered = []
    for l in new_lines:
        if l == "No tumor present." and "No tumor present." in filtered:
            continue
        filtered.append(l)

    return '\n'.join(filtered).strip()

# ──────────────────────────────────────────────────────────────
# 0. 경로 및 환경 설정

BASE_ID     = "/home/mts/ssd_16tb/member/jks/medgemma/medgemma-4b-it"
ADAPTER_ID  = "/home/mts/ssd_16tb/member/jks/medgemma/medgemma_reg2025/notebooks/fintuning_model/"
MERGE_DIR   = "/home/mts/ssd_16tb/member/jks/medgemma/models/medgemma-merged_07_03"
IMG_DIR     = "/home/mts/ssd_16tb/member/jks/medgemma/medgemma_reg2025/notebooks/data/preprocess_tile/tiles_test_415"
OUT_JSON    = "/home/mts/ssd_16tb/member/jks/medgemma/medgemma_reg2025/notebooks/data/medgemma_report_tile_final_07_04.json"
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"]  = "1"

# ──────────────────────────────────────────────────────────────
def merge_adapter_once():
    if Path(MERGE_DIR).exists():
        print(f"▶ 이미 병합된 모델 폴더가 있습니다 → {MERGE_DIR}")
        return

    print("▶ 어댑터 병합 시작…")
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(
        model, ADAPTER_ID, torch_dtype=torch.bfloat16
    )
    model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(BASE_ID)

    Path(MERGE_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MERGE_DIR)
    processor.save_pretrained(MERGE_DIR)
    print(f"✅ 병합 완료 → {MERGE_DIR}")

# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a pathologist AI assistant trained to analyze whole slide images (WSI).\n"
    "You are able to understand the visual content of histopathological slides and generate structured pathology reports.\n"
    "Your response must follow this exact format:\n"
    "[Organ/tissue], [procedure];\n"
    "[Findings]\n"
    "- If there are multiple findings, number them (1., 2., etc.), each on a new line.\n"
    "- Add notes only after a double newline (\\n\\n), and only if directly relevant to the specimen (e.g., \"Note) The specimen includes muscle proper.\")\n"
    "- For non-malignant cases, state: \"No tumor present.\"\n"
    "Strictly output only the report. Do not provide any explanation, commentary, or analysis beyond the formatted report itself. "
    "Do not refer to the image, model, reasoning, or any additional information.\n"
    "Examples (do not describe them, just use structure):\n"
    "Prostate, biopsy;\nAcinar adenocarcinoma, Gleason score 7 (4+3), grade group 3\n"
    "Breast, biopsy;\nMucinous carcinoma\n"
    "Urinary bladder, transurethral resection;\n1. Non-invasive papillary urothelial carcinoma, high grade\n2. Urothelial carcinoma in situ\n\nNote) The specimen includes muscle proper.\n"
    "Do not copy the above examples. Generate a report only based on the current image."
    "Output must follow the exact format above without any additional text or remarks."
)

def build_prompt() -> str:
    return (
        "### System:\n"
        f"{SYSTEM_PROMPT}\n"
        "### User:\n"
        "<start_of_image>\n"
        "### Assistant:\n"
    )

GEN_KWARGS = dict(
    max_new_tokens      = 128,
    num_beams           = 7,
    repetition_penalty  = 1.0,
    temperature = 0.01,
    top_p = 0.9,
    do_sample           = False,
)

# ──────────────────────────────────────────────────────────────
def generate_reports():
    print("▶ 모델 로드…")
    model = AutoModelForImageTextToText.from_pretrained(
        MERGE_DIR, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    processor = AutoProcessor.from_pretrained(MERGE_DIR)

    GEN_KWARGS["eos_token_id"] = processor.tokenizer.convert_tokens_to_ids("###")

    # 1. 하위 디렉토리까지 .jpg 이미지 탐색
    jpg_paths = sorted(glob.glob(os.path.join(IMG_DIR, "**/*.jpg"), recursive=True))
    print(f"▶ {len(jpg_paths)} 장 추론 시작…")

    results = []
    for p in tqdm(jpg_paths, desc="Generating"):
        prompt = build_prompt()
        inputs = processor(
            text   = [prompt],
            images = [Image.open(p).convert("RGB")],
            return_tensors="pt",
            padding=True
        ).to(DEVICE, dtype=torch.bfloat16)

        with torch.inference_mode():
            out_ids = model.generate(**inputs, **GEN_KWARGS)

        report = processor.decode(
            out_ids[0, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        report = clean_report(report)

        results.append({
            "id": os.path.splitext(os.path.basename(p))[0] + ".tiff",  # 안전하게 확장자 변환
            "report": report
        })

        del inputs, out_ids
        gc.collect()

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 완료: {len(results)} 개 리포트가 {OUT_JSON} 에 저장되었습니다.")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    merge_adapter_once()
    generate_reports()
