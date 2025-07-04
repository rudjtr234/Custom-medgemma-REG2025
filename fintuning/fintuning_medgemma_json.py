import os, json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ------------------------------ 설정 ------------------------------
wsi_dir    = Path("/home/mts/jks/medgemma/medgemma_reg2025/notebooks/data/thumbnail_pngs_6800")   # 이미지 폴더
json_file  = Path("/home/mts/jks/medgemma/medgemma_reg2025/notebooks/data/reg_2025_json/train_medgemma.json")  # id-report JSON
out_path   = Path("wsi_reports_train.json")  # 출력 JSON 파일
ID_KEY, REPORT_KEY = "id", "report"
# ------------------------------------------------------------------

# 1. JSON 로드 ------------------------------------------------------
with json_file.open() as f:
    pairs = [(item[ID_KEY], item[REPORT_KEY]) for item in json.load(f)]
print(f"✔ {len(pairs)} pairs loaded from {json_file}")

# 2. messages 레코드 생성 -------------------------------------------
records = []
for iid, rpt in tqdm(pairs, desc="build"):
    png_name = iid.replace(".tiff", ".png")
    path = wsi_dir / png_name
    if not path.exists():
        raise FileNotFoundError(path)

    # 이미지 손상 검증 (선택적)
    Image.open(path).verify()

    messages = [
        {
            "role": "user",
            "content": [ { "type": "image", "image": str(path) } ]
        },
        {
            "role": "assistant",
            "content": [ { "type": "text", "text": rpt } ]
        },
    ]
    records.append({ "image": str(path), "messages": messages })

# 3. 단일 학습 JSON 저장 ---------------------------------------------
out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
print(f"✅ saved {out_path} ({len(records):,} samples)")
