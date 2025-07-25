
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script is a provided example of how to generate pathology reports
# for whole-slide images using the MedGemma model. It demonstrates how
# to load the merged model and run inference across multiple GPUs. The
# script includes functions for merging model adapters, generating
# partial reports in parallel, and aggregating results into final
# representative reports per slide. Note that file paths and model
# identifiers are environment-specific and may need to be adjusted for
# different setups.

import os
import glob
import json
import gc
import re
import multiprocessing as mp
from collections import defaultdict, Counter
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel


def clean_report(text: str) -> str:
    """ Clean the generated report by removing HTML tags, normalizing newlines,
    removing 'n/a', filtering duplicate lines, and keeping a single
    instance of 'No tumor present.' when repeated.

    Args:
        text (str): Raw report text from the model.

    Returns:
        str: Cleaned report text.
    """
    # Remove any HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize newline characters
    text = text.replace('\\n\\n', '\n\n').replace('\\n', '\n')
    # Remove 'n/a' variations
    text = re.sub(r'\b[nN]/?[aA]\b', '', text)
    # Process lines and filter duplicates
    lines = text.split('\n')
    seen = set()
    new_lines = []
    for line in lines:
        line = line.strip()
        line = re.sub(r'^#*\s*(Human|Assistant|Response|Report|Example)\s*[:竊??\s*', '', line, flags=re.IGNORECASE)
        if line and line not in seen:
            seen.add(line)
            new_lines.append(line)
    # Ensure 'No tumor present.' appears only once
    filtered = []
    for l in new_lines:
        if l == "No tumor present." and "No tumor present." in filtered:
            continue
        filtered.append(l)
    return '\n'.join(filtered).strip()


# Base paths for model and data (example values; adjust to your environment)
BASE_ID = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/medgemma-4b-it"
ADAPTER_ID = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/fintuning_model/medgemma_2400_VIT_v0.1.0"
MERGE_DIR = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/models/medgemma-merged_2400_VIT_v0.1.0"
IMG_DIR = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/data/tile_test_2400"
OUT_JSON = "/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/data/preprocess_tile/fintuning_result/REG2025_Medgemma_report_tile_VIT_v0.1.0.json"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Environment variables to avoid certain NCCL issues
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# System prompt for the model, instructing it to generate structured pathology reports
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
    "Urinary bladder, transurethral resection;\n1. Non-invasive papillary urothelial carcinoma, high grade\n2. Urothelial carcinoma in situ\n\n"
    "Note) The specimen includes muscle proper.\n"
    "Do not copy the above examples. Generate a report only based on the current image."
    "Output must follow the exact format above without any additional text or remarks."
)

# Generation parameters for the model
GEN_KWARGS = dict(
    max_new_tokens=128,
    num_beams=10,
    do_sample=False,
    repetition_penalty=1.2
)


def merge_adapter_once():
    """
    Merge the adapter weights into the base model and save the merged model
    into MERGE_DIR. This function checks whether the merged model already
    exists before attempting to merge. It includes creation of a new
    language modeling head to ensure proper alignment of embedding weights.
    """
    model_patterns = ["pytorch_model.bin", "model.safetensors", "pytorch_model-*.bin"]

    def model_exists() -> bool:
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
            print(f"Merged model already exists in {MERGE_DIR}")
            return
        else:
            print("Directory exists but merged model is not found. Proceeding to merge.")
    else:
        print("Starting adapter merge process...")

    try:
        # Load base model
        model = AutoModelForImageTextToText.from_pretrained(
            BASE_ID,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            tie_word_embeddings=False,
            local_files_only=True,
            trust_remote_code=True,
        )

        # Load and merge PEFT adapter
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_ID,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        model = model.merge_and_unload()

        # Create a new lm_head using the embedding weights
        import torch.nn as nn
        new_lm_head = nn.Linear(
            model.language_model.embed_tokens.weight.shape[0],
            model.language_model.embed_tokens.weight.shape[1],
            bias=False
        ).to(dtype=torch.bfloat16)
        new_lm_head.weight.data = model.language_model.embed_tokens.weight.data.clone().detach()
        model.lm_head = new_lm_head

        # Save the merged model
        Path(MERGE_DIR).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(MERGE_DIR, safe_serialization=True)
        processor = AutoProcessor.from_pretrained(BASE_ID, local_files_only=True)
        processor.save_pretrained(MERGE_DIR)

    except Exception as e:
        print(f"Error occurred during adapter merging: {e}")
        return

    # Verify the merged model
    if model_exists():
        print(f"Adapter merge completed successfully: {MERGE_DIR}")
    else:
        print("Merge appears incomplete. Please verify the contents of the merge directory.")


def generate_partial_reports(gpu_id: int, image_paths: list) -> None:
    """
    Generate pathology reports for a subset of images on a specific GPU.
    This function loads the model on the given GPU, processes the images,
    generates reports, and stores them in a temporary JSON file.

    Args:
        gpu_id (int): Identifier for the GPU to use.
        image_paths (list): List of image file paths to process.
    """
    print(f"[GPU {gpu_id}] Starting inference on {len(image_paths)} images")
    device = f"cuda:{gpu_id}"

    model = AutoModelForImageTextToText.from_pretrained(
        MERGE_DIR, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    processor = AutoProcessor.from_pretrained(MERGE_DIR)

    results = []
    total = len(image_paths)
    prev_time = time.time()

    for i, path in enumerate(image_paths, start=1):
        try:
            image = Image.open(path).convert("RGB")
            print(f"[GPU {gpu_id}] Loaded image: {path}", flush=True)
        except Exception as e:
            print(f"[GPU {gpu_id}] Failed to load image: {path} - {e}", flush=True)
            continue

        try:
            prompt = SYSTEM_PROMPT + "\n<start_of_image>"
            image_inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            if "input_ids" not in image_inputs or "pixel_values" not in image_inputs:
                print(f"[GPU {gpu_id}] Missing input_ids or pixel_values: {path}", flush=True)
                continue
            input_ids = image_inputs["input_ids"].to(device)
            pixel_values = image_inputs["pixel_values"].to(device)
            print(f"[GPU {gpu_id}] input_ids shape: {input_ids.shape}, pixel_values shape: {pixel_values.shape}", flush=True)
        except Exception as e:
            print(f"[GPU {gpu_id}] Failed to prepare inputs: {path} - {e}", flush=True)
            continue

        try:
            with torch.no_grad():
                gen_out = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False,
                    **GEN_KWARGS
                )
                output_ids = gen_out.sequences[0]
            input_len = input_ids.shape[1]
            gen_output = output_ids[input_len:]
            response = processor.tokenizer.decode(gen_output, skip_special_tokens=True)
            cleaned = clean_report(response)
            results.append({
                "id": os.path.splitext(os.path.basename(path))[0] + ".tiff",
                "report": cleaned
            })
        except Exception as e:
            print(f"[GPU {gpu_id}] Error during generation or decoding: {path}: {e}", flush=True)
            continue

        if i % 10 == 0 or i == total:
            now = time.time()
            print(f"[GPU {gpu_id}] Completed {i}/{total} images ({(i/total)*100:.1f}%) - Last 10 images took {now - prev_time:.1f}s")
            prev_time = now

    tmp_file = f"{OUT_JSON}.gpu{gpu_id}.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[GPU {gpu_id}] Finished. Temporary results saved to {tmp_file}")


def generate_reports_multi_gpu(num_gpus: int = 4) -> None:
    """
    Orchestrate multi-GPU report generation by splitting image paths across GPUs,
    launching processes, merging results, and producing representative reports
    per whole-slide image.

    Args:
        num_gpus (int): Number of GPUs to utilize.
    """
    all_slide_dirs = sorted(glob.glob(os.path.join(IMG_DIR, "*")))[:100]
    jpg_paths = []
    for slide_dir in all_slide_dirs:
        jpg_paths.extend(sorted(glob.glob(os.path.join(slide_dir, "*.jpg"))))
    print(f"\nTotal of {len(jpg_paths):,} images will be processed across {num_gpus} GPUs.\n")
    chunk_size = len(jpg_paths) // num_gpus
    chunks = [jpg_paths[i * chunk_size: (i + 1) * chunk_size] for i in range(num_gpus - 1)]
    chunks.append(jpg_paths[(num_gpus - 1) * chunk_size:])
    processes = []
    start_time = time.time()
    for i in range(num_gpus):
        print(f"🚀 Assigning {len(chunks[i]):,} images to GPU {i}")
        p = mp.Process(target=generate_partial_reports, args=(i, chunks[i]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    final_results = []
    for i in range(num_gpus):
        tmp_file = f"{OUT_JSON}.gpu{i}.tmp"
        print(f"📥 Merging results from GPU {i}: {tmp_file}")
        with open(tmp_file, "r", encoding="utf-8") as f:
            partial = json.load(f)
            final_results.extend(partial)
        os.remove(tmp_file)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved {len(final_results):,} tile-level reports to {OUT_JSON}")
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
    print(f"📁 Saved {len(representative_results):,} representative slide reports to {rep_out_json}")
    print(f"⏱ Total elapsed time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    generate_reports_multi_gpu(num_gpus=4)


















