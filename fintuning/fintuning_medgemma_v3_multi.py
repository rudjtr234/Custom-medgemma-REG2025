#!/usr/bin/env python
# coding: utf-8
"""
Fine‑tuning script for MedGemma (4‑bit QLoRA, multi‑image per sample)
====================================================================

변경 및 개선 사항 (v0.2)
-----------------------
1. 전처리 단계에서의 제대로 이미지 input 또는 json input이 제대로 들어가는지 현황파악 추가
2. 단일 GPU에서만 쓰던 코드를 accelerate config를 통해서 멀티 gpu로 변환하여 학습속도 및 메모리를 증강
3. 변경된 실행코드 

CUDA_VISIBLE_DEVICES=0,1,3,4 accelerate launch   
--mixed_precision fp16   
fintuning_medgemma_v3_multi.py     
--model_path /home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/medgemma-4b-it     
--train_json /home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/data/preprocess_tile/make_json/train_json/train_json_clean.json     
--epochs 3     
--lr 3e-5     
--rank 8     
--no_vis

4. json에서의 불필요한 공백 및 데이터 정제

5. eval , train set 분리 추가

6. GPU 4개 사용 NVIDIA RTX 6000 Ada 

"""

# ────────────────────────────────────────────────
# 0. Imports & CLI
# ────────────────────────────────────────────────
import os, mimetypes, datetime, argparse
import torch
from PIL import Image
import random                         # ←★ 추가
from evaluate import load as evload   # ←★ 추가import matplotlib
import matplotlib.pyplot as plt

from datasets import load_dataset
from accelerate import Accelerator
from transformers import (
    AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# ────────────────────────────────────────────────
# 1. CLI
# ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fine‑tune MedGemma with QLoRA")
parser.add_argument("--model_path", default="/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/medgemma-4b-it")
parser.add_argument("--train_json", default="/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/data/preprocess_tile/make_json/train_json/train_json_clean.json")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--rank", type=int, default=8)
parser.add_argument("--no_vis", action="store_true", help="Disable sample visualization for headless run")
args = parser.parse_args()

# ────────────────────────────────────────────────
# 2. Model & LoRA 준비
# ────────────────────────────────────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.float32,
)

accel = Accelerator(mixed_precision="fp16")

with accel.main_process_first():
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)


# ─── Gradient checkpointing 설정 ───────────────────────────
# ① 먼저 전체 모듈의 체크포인트링을 **완전히 끄고**
model.gradient_checkpointing_disable()

# ② **Vision‑tower** 레이어에만 다시 켭니다.
vt = model.vision_tower
vision_layers = vt.encoder.layers if hasattr(vt, "encoder") else vt.vision_model.encoder.layers
for blk in vision_layers:
    blk.gradient_checkpointing = True

# LoRA 설정
targets = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
lora_cfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 4, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

processor = AutoProcessor.from_pretrained(args.model_path)

# ────────────────────────────────────────────────
# 3. Dataset 로드 & Sanity‑check
# ────────────────────────────────────────────────

def is_valid(ex):
    return ex.get("image") and os.path.exists(ex["image"]) and ex.get("messages")

with accel.main_process_first():
    ds_all = load_dataset("json", data_files={"train": args.train_json})["train"].filter(is_valid)

print(f"✅ 전체 샘플: {len(ds_all):,}")

# 9:1 shuffle split ---------------------------------------------------
indices      = list(range(len(ds_all)))
random.seed(42); random.shuffle(indices)
cut          = int(len(indices) * 0.9)
train_idx    = indices[:cut]
val_idx      = indices[cut:]

ds_train     = ds_all.select(train_idx)
# fallback: HuggingFace Dataset select returns empty if indices=[]
ds_val       = ds_all.select(val_idx) if val_idx else ds_all.select(indices[-1:])

if accel.is_main_process:
    print(f"✅ Train : {len(ds_train):,} | Val : {len(ds_val):,}")


# (1) 샘플 프롬프트·이미지 시각 확인
if not args.no_vis and accel.is_main_process:
    print("\n[🔎 샘플 시각화 & Prompt 확인]")
    for i in range(min(3, len(ds))):
        ex = ds_all[i]
        prompt = processor.apply_chat_template(ex["messages"], add_generation_prompt=False, tokenize=False).strip()
        print(f"\n─ Sample {i} ─\nPrompt:\n" + prompt)
        img_path = ex["image"]
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.title(f"Sample {i}")
            plt.axis("off")
            plt.close()
        else:
            print("⚠ 이미지 파일 없음:", img_path)

# (2) 유니크 라벨 다양성 체크
print("\n[🧪 라벨 다양성 검사]")
label_set = set()
for i in range(min(1000, len(ds_all))):
    for m in ds_all[i]["messages"]:
        if m["role"] == "assistant":
            for c in m["content"]:
                if c.get("text"):
                    label_set.add(c["text"].strip())
label_cnt = len(label_set)
print(f"유니크 라벨 수: {label_cnt}")
if label_cnt < 2:
    print("⚠ 경고: 모든 샘플이 동일한 라벨을 갖고 있을 가능성이 높습니다.")

# (3) Processor 테스트
print("\n[⚙ Processor 마스킹 테스트]")
_sample = [ds_all[i] for i in range(min(4, len(ds_all)))]
texts = [processor.apply_chat_template(s["messages"], add_generation_prompt=False, tokenize=False).strip() for s in _sample]
imgs = [[Image.open(s["image"]).convert("RGB")] for s in _sample]
encoded = processor(text=texts, images=imgs, padding=True, return_tensors="pt")
labels = encoded["input_ids"].clone()
labels[labels == processor.tokenizer.pad_token_id] = -100
label_tokens = (labels != -100).sum().item()
print(f"유효 라벨 토큰 수: {label_tokens}")
if label_tokens == 0:
    raise ValueError("❌ 모든 라벨이 -100으로 마스킹되었습니다. 학습 불가!")
print("✅ Processor 테스트 통과")

# ────────────────────────────────────────────────
# 4. Collate 함수
# ────────────────────────────────────────────────

def load_image(path: str):
    mime, _ = mimetypes.guess_type(path)
    return Image.open(path).convert("RGB") if mime and mime.startswith("image") else torch.load(path, map_location="cpu")


def collate(batch):
    batch = [ex for ex in batch if is_valid(ex)] or [ds_all[0]]  # 빈 배치 방지
    texts, images = [], []
    for ex in batch:
        texts.append(processor.apply_chat_template(ex["messages"], add_generation_prompt=False, tokenize=False).strip())
        images.append([load_image(ex["image"])])

    out = processor(text=texts, images=images, padding=True, return_tensors="pt")
    labels = out["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    start_id = processor.tokenizer.convert_tokens_to_ids("<start_of_image>")
    soft_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    for i, ids in enumerate(out["input_ids"]):
        for p in (ids == start_id).nonzero(as_tuple=True)[0]:
            q = p + 1
            while q < ids.size(0) and soft_id <= ids[q] < soft_id + 256:
                q += 1
            labels[i, p:q] = -100

    out["labels"] = labels
    return out



# ─── Metrics ───────────────────────────────────
bleu = evload("bleu")
try:
    import nltk; nltk.download("wordnet", quiet=True)
    meteor = evload("meteor")
except Exception:
    meteor = None
try:
    clip_s = evload("microsoft/clip_score")   # hf hub id
except Exception:
    clip_s = None

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds_txt  = processor.batch_decode(preds, skip_special_tokens=True)
    labels_txt = processor.batch_decode(labels, skip_special_tokens=True)
    res = {"bleu": bleu.compute(predictions=preds_txt, references=[[t] for t in labels_txt])["bleu"]}
    if meteor:
        res["meteor"] = meteor.compute(predictions=preds_txt, references=[[t] for t in labels_txt])["meteor"]
    if clip_s:
        try:
            imgs = [load_image(ds_val[i]["image"]) for i in range(len(preds_txt))]
            res["clip"] = clip_s.compute(predictions=preds_txt, images=imgs)["clip_score"]
        except Exception:
            pass
    return res


# ─── Trainer ───────────────────────────────────
stamp = datetime.datetime.now().strftime("%m%d_%H%M")
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=f"qlora_ckpt_{stamp}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        logging_steps=50,
        save_strategy="steps",
        eval_steps=500,         # 500 스텝마다 Val
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,
        label_names=["labels"],
        report_to=["tensorboard"],
    ),
    train_dataset=ds_train,
    eval_dataset=ds_val,
    compute_metrics=compute_metrics,
    data_collator=collate,
)

if accel.is_main_process:
    print("[🚀 학습 시작]")
trainer.train()

# ─── Save ──────────────────────────────────────
accel.wait_for_everyone()
if accel.is_main_process:
    out_dir = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/fintuning_model/medgemma_2400_VIT_v0.1.1"
    trainer.save_model(out_dir)
    print("모델 저장 완료:", out_dir)

