import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch, json
from pathlib import Path
from transformers import (
    AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from PIL import Image
from typing import Any
from peft.utils import prepare_model_for_kbit_training
from datetime import datetime

# 0. 기본 환경 확인: GPU가 bfloat16을 지원하는지 확인
if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("bfloat16을 지원하는 GPU가 필요합니다 (Ada/Hopper 계열 이상)")

# 1. 모델 및 Processor 로드
model_path = "/home/mts/ssd_16tb/member/jks/medgemma/medgemma-4b-it"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.float32,
)

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    quantization_config=bnb_cfg,
)
model = prepare_model_for_kbit_training(model)

processor = AutoProcessor.from_pretrained(model_path)
processor.tokenizer.padding_side = "right"

# 2. PEFT 설정 (LoRA)
peft_cfg = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_tokens"]
)

def collate_fn(examples: list[dict[str, Any]]):
    texts, images = [], []

    for idx, ex in enumerate(examples):
        try:
            if "image" not in ex or ex["image"] is None:
                print(f"Warning: Sample {idx} has no image field, skipping")
                continue
            if not os.path.exists(ex["image"]):
                print(f"Warning: Image file not found for sample {idx}: {ex['image']}")
                continue
            img = Image.open(ex["image"]).convert("RGB")
            if img.size[0] == 0 or img.size[1] == 0:
                print(f"Warning: Invalid image size for sample {idx}, skipping")
                continue
            if "messages" not in ex or not ex["messages"]:
                print(f"Warning: Sample {idx} has no messages, skipping")
                continue

            text = processor.apply_chat_template(
                ex["messages"], add_generation_prompt=False, tokenize=False
            ).strip()

            if not text:
                print(f"Warning: Sample {idx} produced empty text, skipping")
                continue

            if "<start_of_image>" not in text and "<image>" not in text:
                print(f"Warning: Sample {idx} has no image tokens in text")

            images.append([img])
            texts.append(text)

        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue

    if not texts:
        raise ValueError("No valid samples found in batch!")

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id
    labels[labels == pad_id] = -100

    start_image_token_id = processor.tokenizer.convert_tokens_to_ids("<start_of_image>")
    img_soft_start = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")

    masked_samples = 0
    total_masked_tokens = 0

    for i in range(labels.size(0)):
        input_ids = batch["input_ids"][i].tolist()
        image_start_positions = [j for j, token_id in enumerate(input_ids) if token_id == start_image_token_id]

        if not image_start_positions:
            print(f"Warning: No <start_of_image> token found in sample index {i}")
            continue

        sample_masked_tokens = 0
        for start_idx in image_start_positions:
            end_idx = start_idx + 1
            while end_idx < len(input_ids) and (
                img_soft_start <= input_ids[end_idx] < img_soft_start + 256
            ):
                end_idx += 1
            labels[i, start_idx:end_idx] = -100
            sample_masked_tokens += (end_idx - start_idx)

        if sample_masked_tokens > 0:
            masked_samples += 1
            total_masked_tokens += sample_masked_tokens

    batch["labels"] = labels

    if not getattr(collate_fn, "_printed", False):
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Processed samples: {len(texts)}")
        print(f"Masked image token samples: {masked_samples}")
        print(f"Total masked tokens: {total_masked_tokens}")

        print(f"\nBatch shapes:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        if 'pixel_values' in batch:
            print(f"  pixel_values: {batch['pixel_values'].shape}")

        print(f"\nFirst sample preview:")
        print(f"  Text length: {len(texts[0])} chars")
        print(f"  Token length: {len(batch['input_ids'][0])}")
        masked_in_first_50 = (batch["labels"][0][:50] == -100).sum().item()
        print(f"  Masked tokens in first 50: {masked_in_first_50}")
        print(f"  Sample text preview: {texts[0][:200]}...")
        image_positions = (batch["input_ids"][0] == start_image_token_id).nonzero().flatten()
        print(f"  Image token positions: {image_positions.tolist() if len(image_positions) > 0 else 'None'}")
        print("="*60)

        collate_fn._printed = True

    return batch

    exit()

# 4. 학습 설정 구성
now = datetime.now().strftime("%m%d_%H%M")
sft_cfg = SFTConfig(
    output_dir=f"/home/mts/ssd_16tb/member/jks/medgemma/fintuning_ckpts/medgemma_crc5000_{now}",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=5e-5,
    optim="adamw_torch_fused",
    logging_steps=5,
    logging_first_step=True,
    save_strategy="epoch",
    eval_strategy="no",
    load_best_model_at_end=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    push_to_hub=False,
    report_to=["tensorboard"],
    remove_unused_columns=False,
    label_names=["labels"],
    dataset_kwargs={"skip_prepare_dataset": True}
)

# 5. 데이터 로드
data_paths = {
    "train": "/home/mts/ssd_16tb/member/jks/medgemma/medgemma_reg2025/notebooks/data/preprocess_tile/make_json/medgemma_tile_train_415_07_03.json"
}
ds = load_dataset("json", data_files=data_paths)

# 6. Trainer 정의 및 학습 수행
trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=ds["train"],
    peft_config=peft_cfg,
    data_collator=collate_fn,
)

trainer.train()

trainer.save_model("/home/mts/ssd_16tb/member/jks/medgemma/medgemma_reg2025/notebooks/fintuning_model/medgemma-REG_2025_07_32_tile_415_test_model")
print("🎉 파인튜닝 완료 및 LoRA 어댑터 저장")
