import os
import json
import csv
import random
import logging
from collections import defaultdict

import torch
import faiss
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
import clip
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from expert_generator import expert_generator  # 确保 expert_generator.py 中没有 "from expert_generator import ..."

# ==========================
# Configurations
# ==========================
CONFIG = {
    "seed": 123,
    "device": "cuda",
    "clip_model": "ViT-B/16",
    "qwen_path": "Qwen/Qwen2-VL-7B-Instruct",
    "reference_index": "../Memory/memory.index",
    "reference_images": "../Memory/reference_image_locations.txt",
    "annotation_file": "../Annotation/DS-MVTec.json",
    "domain_knowledge": "../Knowledge Guide/domain_knowledge_detection.json",
    "results_csv": "Results_Qwen2VL.csv"
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def set_seed(seed=123):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def smart_resize(image_path, max_edge=448):
    """Resize image with max edge <= max_edge, preserving aspect ratio."""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    scale = min(max_edge / w, max_edge / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image

def get_image_feature(image_path, clip_model, preprocess, device="cuda"):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().squeeze()

def find_all_descriptions(json_file_path, img_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        object_name = img_path.split('/')[1]
        for sub_dict in data.values():
            if isinstance(sub_dict, dict) and object_name in sub_dict:
                return {"object_name": object_name, "descriptions": sub_dict[object_name]}
        return {"object_name": object_name, "descriptions": "No descriptions found."}
    except Exception as e:
        return {"error": str(e)}

def load_reference_images(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def evaluate_model():
    set_seed(CONFIG["seed"])

    # Enable TF32 for Ampere GPUs (RTX 3090)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load CLIP
    logging.info("Loading CLIP model...")
    clip_model, preprocess = clip.load(CONFIG["clip_model"], device=CONFIG["device"])

    # Load FAISS index
    logging.info("Loading FAISS index and reference images...")
    index_img = faiss.read_index(CONFIG["reference_index"])
    image_paths = load_reference_images(CONFIG["reference_images"])

    # Load Qwen2-VL
    logging.info("Loading Qwen2-VL-7B model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        CONFIG["qwen_path"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(CONFIG["device"])
    processor = AutoProcessor.from_pretrained(CONFIG["qwen_path"], trust_remote_code=True, use_fast=False)
    logging.info("Model loaded successfully.")

    # Load and filter dataset: keep only existing images
    logging.info("Loading annotation file and filtering missing images...")
    with open(CONFIG["annotation_file"], 'r') as f:
        raw_data = json.load(f)

    base_dir = "../Dataset/MMAD"  #在test中是test
    filtered_data = {}
    for img_path, value in raw_data.items():
        full_path = os.path.join(base_dir, img_path)
        if os.path.exists(full_path):
            filtered_data[img_path] = value
        else:
            logging.warning(f"Skipped (not found): {full_path}")

    logging.info(f"✅ Found {len(filtered_data)} valid images out of {len(raw_data)} total entries.")

    if len(filtered_data) == 0:
        logging.error("No valid images found. Please check your dataset path.")
        return

    metrics = defaultdict(lambda: {'y_true': [], 'y_pred': []})

    # Open CSV for writing results
    with open(CONFIG["results_csv"], 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Image Path', 'Question', 'Predicted Answer', 'Correct Answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (img_path, item_value) in enumerate(filtered_data.items()):
            logging.info(f"\n[{idx+1}/{len(filtered_data)}] Processing: {img_path}")

            query_img_path = os.path.join(base_dir, img_path)

            # Step 1: Extract CLIP feature
            try:
                feat = get_image_feature(query_img_path, clip_model, preprocess)
            except Exception as e:
                logging.error(f"Failed to extract CLIP feature: {e}")
                continue

            # Step 2: Retrieve reference image
            I = index_img.search(feat.reshape(1, -1), k=1)
            ref_img_path = image_paths[I[0][0]]
            if not os.path.exists(ref_img_path):
                logging.warning(f"Reference image missing: {ref_img_path}")
                continue

            # Step 3: Load and resize images
            try:
                ref_img = smart_resize(ref_img_path, max_edge=448)
                query_img = smart_resize(query_img_path, max_edge=448)
            except Exception as e:
                logging.error(f"Image loading/resizing failed: {e}")
                continue

            # Step 4: Get domain knowledge
            domain_knowledge = find_all_descriptions(CONFIG["domain_knowledge"], img_path)

            # Step 5: Process each QA pair
            for conv in item_value['conversation']:
                question = conv['Question']
                correct_answer = conv['Answer']
                options = conv['Options']
                q_type = conv['type']

                options_text = "\n".join(f"{k}: {v}" for k, v in options.items())
                messages = expert_generator(ref_img, query_img, q_type, question, options_text, domain_knowledge)

                # Tokenize
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(CONFIG["device"])

                # Generate
                with torch.no_grad():
                    response_ids = model.generate(**inputs, max_new_tokens=128)

                # Decode
                response_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, response_ids)
                ]
                response = processor.batch_decode(response_ids_trimmed, skip_special_tokens=True)
                predicted = response[0].strip() if response else "N/A"

                # Record
                metrics[q_type]['y_true'].append(correct_answer)
                metrics[q_type]['y_pred'].append(predicted)

                writer.writerow({
                    'Image Path': img_path,
                    'Question': question,
                    'Predicted Answer': predicted,
                    'Correct Answer': correct_answer
                })

                # Clean up GPU memory
                del inputs, response_ids, response_ids_trimmed, response
                torch.cuda.empty_cache()

    # Compute accuracy per question type
    for q_type, vals in metrics.items():
        valid_pairs = [(t, p) for t, p in zip(vals['y_true'], vals['y_pred']) if p != "N/A"]
        if not valid_pairs:
            acc = "N/A"
        else:
            y_true, y_pred = zip(*valid_pairs)
            # Extract first alphabetic character as answer (e.g., "A" from "A. Normal")
            y_pred_clean = [next((c for c in p if c.isalpha()), p) for p in y_pred]
            acc = accuracy_score(y_true, y_pred_clean)
        logging.info(f"\n📊 Question Type: {q_type} | Accuracy: {acc}")

    logging.info("✅ Evaluation completed.")


if __name__ == "__main__":
    evaluate_model()