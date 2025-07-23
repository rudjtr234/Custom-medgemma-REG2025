
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
import multiprocessing as mp
from collections import defaultdict, Counter
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import time
from peft import PeftModel
from transformers import AutoTokenizer


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

BASE_ID     = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/medgemma-4b-it"
ADAPTER_ID  = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/fintuning_model/medgemma_2400_VIT_v0.1.1"
MERGE_DIR   = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/models/medgemma-merged_2400_VIT_v0.1.1"
#IMG_DIR     = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/data/tile_testphase1_preprocess_data"

IMG_DIR     = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/data/tile_test_2400"

OUT_JSON    = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/data/preprocess_tile/fintuning_result/REG2025_Medgemma_report_tile_VIT_v0.1.1.json"
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"]  = "1"

# ──────────────────────────────────────────────────────────────
def merge_adapter_once():
    # 병합되었는지 확인할 수 있는 파일 패턴들
    model_patterns = ["pytorch_model.bin", "model.safetensors", "pytorch_model-*.bin"]

    def model_exists():
        for pattern in model_patterns:
            if "*" in pattern:
                if any(Path(MERGE_DIR).glob(pattern)):
                    return True
            else:
                if (Path(MERGE_DIR) / pattern).exists():
                    return True
        return False

    if Path(MERGE_DIR).exists():
        if model_exists():
            print(f"▶ 이미 병합된 모델이 존재합니다 → {MERGE_DIR}")
            return
        else:
            print(f"⚠️ 병합 폴더는 있으나 모델 파일이 없습니다. 병합을 다시 시도합니다.")

    print("▶ 어댑터 병합 시작…")

    try:
        # ① untied 상태로 로딩
        model = AutoModelForImageTextToText.from_pretrained(
            BASE_ID,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            tie_word_embeddings=False,
            local_files_only=True,
            trust_remote_code=True,
        )

        # ② PEFT 어댑터 병합
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_ID,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        model = model.merge_and_unload()

        # ③ lm_head를 embed_tokens와 분리된 객체로 생성
        import torch.nn as nn
        new_lm_head = nn.Linear(
            model.language_model.embed_tokens.weight.shape[0],
            model.language_model.embed_tokens.weight.shape[1],
            bias=False
        ).to(dtype=torch.bfloat16)
        new_lm_head.weight.data = model.language_model.embed_tokens.weight.data.clone().detach()
        model.lm_head = new_lm_head

        # ④ 저장
        Path(MERGE_DIR).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(MERGE_DIR, safe_serialization=True)

        processor = AutoProcessor.from_pretrained(BASE_ID, local_files_only=True)
        processor.save_pretrained(MERGE_DIR)

    except Exception as e:
        print(f"❌ 병합 또는 저장 중 오류 발생: {e}")
        return

    # ⑤ 저장 확인 (sharded 모델도 포함해서 확인)
    if model_exists():
        print(f"✅ 병합 완료 및 저장 성공 → {MERGE_DIR}")
    else:
        print(f"⚠️ 병합은 되었으나 모델 파일이 저장되지 않았습니다. 경로 확인 필요 → {MERGE_DIR}")

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

tokenizer = AutoTokenizer.from_pretrained(BASE_ID)

# 그 다음에 GEN_KWARGS 정의
GEN_KWARGS = dict(
    max_new_tokens     = 128,
    num_beams          = 10,
    do_sample          = False,
    repetition_penalty = 1.2,
    pad_token_id       = tokenizer.eos_token_id,  # ✅ 콤마 제거!
    eos_token_id       = tokenizer.eos_token_id
)
# ──────────────────────────────────────────────────────────────


def generate_partial_reports(gpu_id, image_paths):
    print(f"[GPU {gpu_id}] {len(image_paths)}개 이미지 추론 시작")
    device = f"cuda:{gpu_id}"

    model = AutoModelForImageTextToText.from_pretrained(
        MERGE_DIR, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    processor = AutoProcessor.from_pretrained(MERGE_DIR)
    GEN_KWARGS["eos_token_id"] = processor.tokenizer.convert_tokens_to_ids("###")
    GEN_KWARGS["return_dict_in_generate"] = True
    GEN_KWARGS["output_scores"] = False

    results = []
    total = len(image_paths)
    prev_time = time.time()

    for i, path in enumerate(image_paths, start=1):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[GPU {gpu_id}] 이미지 로딩 실패: {path} - {e}", flush=True)
            continue

        try:
            prompt = str(build_prompt())
            inputs = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                padding=True
            ).to(device)

            print(f"[GPU {gpu_id}] 입력 생성 완료 - {os.path.basename(path)}")
        except Exception as e:
            print(f"[GPU {gpu_id}] processor 입력 생성 실패: {e}", flush=True)
            continue

        try:
            with torch.no_grad():
                gen_out = model.generate(**inputs, **GEN_KWARGS)
                output_ids = gen_out.sequences[0]
            input_len = len(inputs["input_ids"][0])
            gen_output = output_ids[input_len:]  # 입력 부분 제외
            response = processor.decode(gen_output, skip_special_tokens=True)
            cleaned = clean_report(response)

            results.append({
                "id": os.path.splitext(os.path.basename(path))[0] + ".tiff",
                "report": cleaned
            })
        except Exception as e:
            print(f"[GPU {gpu_id}] 결과 처리 실패 - {path}: {e}", flush=True)
            continue

        # 진행률 출력
        if i % 10 == 0 or i == total:
            now = time.time()
            elapsed = now - prev_time
            print(f"[GPU {gpu_id}] {i}/{total}장 완료 ({(i/total)*100:.1f}%) - 10장 처리 시간: {elapsed:.1f}초", flush=True)
            prev_time = now

    # 결과 저장
    tmp_file = f"{OUT_JSON}.gpu{gpu_id}.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[GPU {gpu_id}] 완료 → {tmp_file}")

# ──────────────────────────────────────────────────────────────

def generate_reports_multi_gpu(num_gpus=5):
    all_slide_dirs = sorted(glob.glob(os.path.join(IMG_DIR, "*")))[:100]  # 슬라이드 디렉토리 상한선
    jpg_paths = []
    for slide_dir in all_slide_dirs:
        jpg_paths.extend(sorted(glob.glob(os.path.join(slide_dir, "*.jpg"))))

    print(f"\n📊 총 {len(jpg_paths):,}장의 이미지를 {num_gpus}개의 GPU로 나누어 추론합니다.\n")

    chunk_size = len(jpg_paths) // num_gpus
    chunks = [jpg_paths[i * chunk_size: (i + 1) * chunk_size] for i in range(num_gpus - 1)]
    chunks.append(jpg_paths[(num_gpus - 1) * chunk_size:])  # 마지막에 나머지 포함

    processes = []
    start_time = time.time()

    for i in range(num_gpus):
        print(f"🚀 GPU {i} → {len(chunks[i]):,}장 할당됨")
        p = mp.Process(target=generate_partial_reports, args=(i, chunks[i]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 결과 병합
    final_results = []
    for i in range(num_gpus):
        tmp_file = f"{OUT_JSON}.gpu{i}.tmp"
        print(f"📥 GPU {i} 결과 병합 중: {tmp_file}")
        with open(tmp_file, "r", encoding="utf-8") as f:
            partial = json.load(f)
            final_results.extend(partial)
        os.remove(tmp_file)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 타일 단위 리포트 {len(final_results):,}개 저장 완료 → {OUT_JSON}")

    # 대표 리포트 생성: 슬라이드 ID 추출 시 .tiff 포함
    slide_report_map = defaultdict(list)
    for r in final_results:
        tile_id = os.path.basename(r["id"])
        slide_id = "_".join(tile_id.split("_")[:-2]) + ".tiff"
        slide_report_map[slide_id].append(r["report"])

    representative_results = []
    for slide_id, reports in slide_report_map.items():
        most_common_report = Counter(reports).most_common(1)[0][0]
        representative_results.append({
            "id": slide_id,
            "report": most_common_report
        })

    rep_out_json = OUT_JSON.replace(".json", "_representative.json")
    with open(rep_out_json, "w", encoding="utf-8") as f:
        json.dump(representative_results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print(f"📁 슬라이드 대표 리포트 {len(representative_results):,}개 저장 완료 → {rep_out_json}")
    print(f"⏱ 전체 소요 시간: {elapsed/60:.2f}분 ({elapsed:.1f}초)\n")
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    merge_adapter_once()
    generate_reports_multi_gpu(num_gpus=4)
