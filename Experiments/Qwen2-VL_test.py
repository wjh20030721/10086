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
from expert_generator_test import expert_generator_test  # 确保 expert_generator_test.py 中没有 "from expert_generator_test import ..."

# ==========================
# Configurations
# ==========================
CONFIG = {
    "seed": 123,
    "device": "cuda",
    "clip_model": "ViT-B/16",
    "qwen_path": "Qwen/Qwen2-VL-7B-Instruct",
    # "reference_index": "../Memory/memory.index",
    # "reference_images": "../Memory/reference_image_locations.txt",
    "annotation_file": "../Annotation/test.json",
    "domain_knowledge": "../Knowledge Guide/domain_knowledge_test.json",
    "results_csv": "Results_Qwen2VL_test.csv"
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
        # 添加默认描述
        return {"object_name": object_name, "descriptions": "This image may involve gray industry activities. Please analyze carefully."}
    except Exception as e:
        return {"error": str(e)}

def load_reference_images(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def extract_answer(predicted):
    if not predicted or predicted == "N/A":
        return "N/A"
    # 提取第一个字母作为答案
    for char in predicted:
        if char.isalpha():
            return char.upper()
    return predicted

def evaluate_model():
    set_seed(CONFIG["seed"])

    # Enable TF32 for Ampere GPUs (RTX 3090)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load CLIP
    logging.info("Loading CLIP model...")
    clip_model, preprocess = clip.load(CONFIG["clip_model"], device=CONFIG["device"])

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

    base_dir = "../Dataset/test/loan"  #在test中是test
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
        # Add per-option scores for AUROC/AUPRC
        fieldnames = [
            'Image Path', 'Question', 'Predicted Answer', 'Correct Answer',
            'Score_A', 'Score_B', 'Score_C', 'Score_D'
        ]
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

            # Step 3: Load and resize query image
            try:
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
                messages = expert_generator_test(query_img, q_type, question, options_text, domain_knowledge)

                # Tokenize
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(CONFIG["device"])

                # Compute next-token logits for option-level scores (A/B/C/D)
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Next-token distribution at last position of the prompt
                    logits = outputs.logits[:, -1, :]  # [1, vocab]
                    probs = torch.softmax(logits, dim=-1).squeeze(0)

                # Map option letters to token ids robustly
                tok = processor.tokenizer

                def letter_token_id(letter: str):
                    variants = [letter, ' ' + letter, letter + '.', ' ' + letter + '.', letter + ':', ' ' + letter + ':']
                    for v in variants:
                        ids = tok.encode(v, add_special_tokens=False)
                        if len(ids) == 1:
                            return ids[0]
                    ids = tok.encode(letter, add_special_tokens=False)
                    if ids:
                        return ids[-1]
                    return None

                # Only score letters present in options
                available_letters = [k for k in options.keys() if isinstance(k, str) and len(k) == 1 and k.isalpha()]
                scores_map = {'A': '', 'B': '', 'C': '', 'D': ''}
                for ch in ['A', 'B', 'C', 'D']:
                    if ch in available_letters:
                        tid = letter_token_id(ch)
                        if tid is not None and 0 <= tid < probs.numel():
                            scores_map[ch] = float(probs[tid].item())
                        else:
                            scores_map[ch] = ''

                # Also run generation to keep consistency with previous evaluation outputs
                with torch.no_grad():
                    response_ids = model.generate(**inputs, max_new_tokens=128)
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
                    'Correct Answer': correct_answer,
                    'Score_A': scores_map['A'],
                    'Score_B': scores_map['B'],
                    'Score_C': scores_map['C'],
                    'Score_D': scores_map['D'],
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
            # 使用新的答案提取逻辑
            y_pred_clean = [extract_answer(p) for p in y_pred]
            acc = accuracy_score(y_true, y_pred_clean)
        logging.info(f"\n📊 Question Type: {q_type} | Accuracy: {acc}")

    logging.info("✅ Evaluation completed.")


if __name__ == "__main__":
    evaluate_model()