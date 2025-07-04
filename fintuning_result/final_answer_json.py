import json
from collections import defaultdict, Counter

# 입력 / 출력 경로 설정
input_path  = "/home/mts/ssd_16tb/member/jks/medgemma/medgemma_reg2025/notebooks/data/medgemma_report_tile_final_07_02.json"
output_path_detailed = "slide_final_report_07_02.json"
output_path_simple   = "slide_final_report_simple_07_02.json"

# JSON 불러오기
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 슬라이드 ID 기준으로 리포트 모으기
slide_dict = defaultdict(list)
for item in data:
    tile_id = item["id"]  # 예: PIT_01_00002_01_1792_11648.tiff
    slide_id = "_".join(tile_id.split("_")[:4])  # 예: PIT_01_00002_01
    report = item["report"]
    slide_dict[slide_id].append(report)

# 상세 결과 및 단순 결과 생성
results = []
simplified_results = []

for slide_id, reports in slide_dict.items():
    report_counts = Counter(reports)
    most_common_report, freq = report_counts.most_common(1)[0]
    top_alt_reports = report_counts.most_common(3)

    # 상세 정보
    results.append({
        "slide_id": slide_id,
        "final_report": most_common_report,
        "frequency": freq,
        "tile_count": len(reports),
        "alternative_reports": [
            {"report": rep, "count": cnt}
            for rep, cnt in top_alt_reports
        ]
    })

    # 단순 정보
    simplified_results.append({
        "id": f"{slide_id}.tiff",
        "report": most_common_report
    })

# JSON 저장
with open(output_path_detailed, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

with open(output_path_simple, "w", encoding="utf-8") as f:
    json.dump(simplified_results, f, ensure_ascii=False, indent=2)

print(f"✅ 상세 JSON 저장 완료 → {output_path_detailed}")
print(f"✅ 단순 JSON 저장 완료 → {output_path_simple}")
