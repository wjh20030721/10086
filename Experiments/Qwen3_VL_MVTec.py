import os
import json
import csv
import random
import logging
from collections import defaultdict
from tqdm import tqdm
import time

import torch
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
import clip

# ==========================
# 关键修改 1: 导入 Qwen3-VL 专用类 (或使用 AutoModel)
# ==========================
# 注意: Qwen3-VL 需要 transformers >= 4.51.0
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info


from expert_generator_MVTec import expert_generator_MVTec 

# ==========================
# Configurations
# ==========================
CONFIG = {
    "seed": 123,
    "device": "cuda",
    "clip_model": "ViT-B/16",
    "qwen_path": "Qwen/Qwen3-VL-8B-Instruct", 
    "annotation_file": "../Annotation/DS-MVTec.json",
    "domain_knowledge": "../Knowledge Guide/domain_knowledge_MVTec.json",
    "results_csv": "Results_Qwen3VL.csv",
    "max_image_edge": 1280  
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def set_seed(seed=123):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def smart_resize(image_path, max_edge=1280):
    """
    修改说明: Qwen3-VL 拥有极强的原生分辨率处理能力 (DeepStack ViT)。
    原代码的 448px 会严重破坏模型性能。
    这里将限制放宽到 1280px，仅对超大图进行缩放，尽量保留细节。
    """
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
        return {"object_name": object_name, "descriptions": "This image may involve gray industry activities. Please analyze carefully."}
    except Exception as e:
        return {"error": str(e)}

def extract_answer(predicted):
    if not predicted or predicted == "N/A":
        return "N/A"
    for char in predicted:
        if char.isalpha():
            return char.upper()
    return predicted

def evaluate_model():
    set_seed(CONFIG["seed"])

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load CLIP
    logging.info("Loading CLIP model...")
    clip_model, preprocess = clip.load(CONFIG["clip_model"], device=CONFIG["device"])

    # Load Qwen3-VL
    logging.info(f"Loading {CONFIG['qwen_path']} model...")
    
   
    try:
        # 尝试加载 Qwen3 专用类
        model_class = Qwen3VLForConditionalGeneration
    except NameError:
        # 如果当前环境 transformers 版本未找到该类，回退到 AutoModel
        logging.warning("Qwen3VLForConditionalGeneration not found, using AutoModelForVision2Seq.")
        model_class = AutoModelForVision2Seq

    try:
        model = model_class.from_pretrained(
            CONFIG["qwen_path"],
            torch_dtype=torch.bfloat16, # Qwen3 推荐使用 bfloat16
            attn_implementation="flash_attention_2", 
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        logging.warning(f"Flash Attention 2 loading failed, falling back to auto: {e}")
        model = model_class.from_pretrained(
            CONFIG["qwen_path"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    # Processor 加载 (设置 min/max pixels 控制显存)
    # Qwen3-VL 支持动态分辨率，范围可以很大
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28 # 根据显存调整：8B模型在24G显存下建议设为 1280*28*28 左右
    
    processor = AutoProcessor.from_pretrained(
        CONFIG["qwen_path"], 
        trust_remote_code=True, 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    logging.info("Model loaded successfully.")

    # Load and filter dataset
    logging.info("Loading annotation file and filtering missing images...")
    with open(CONFIG["annotation_file"], 'r') as f:
        raw_data = json.load(f)

    base_dir = "../Dataset/MMAD" 
    filtered_data = {}
    for img_path, value in raw_data.items():
        full_path = os.path.join(base_dir, img_path)
        if os.path.exists(full_path):
            filtered_data[img_path] = value
        else:
            logging.warning(f"Skipped (not found): {full_path}")

    logging.info(f"✅ Found {len(filtered_data)} valid images out of {len(raw_data)} total entries.")

    metrics = defaultdict(lambda: {'y_true': [], 'y_pred': []})

    with open(CONFIG["results_csv"], 'w', newline='', encoding='utf-8') as csvfile:
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

            
            try:
                # 使用更高的分辨率配置
                query_img = smart_resize(query_img_path, max_edge=CONFIG["max_image_edge"])
            except Exception as e:
                logging.error(f"Image loading/resizing failed: {e}")
                continue

            
            domain_knowledge = find_all_descriptions(CONFIG["domain_knowledge"], img_path)

            
            for conv in item_value['conversation']:
                question = conv['Question']
                correct_answer = conv['Answer']
                options = conv['Options']
                q_type = conv['type']

                options_text = "\n".join(f"{k}: {v}" for k, v in options.items())
                messages = expert_generator_MVTec(query_img, q_type, question, options_text, domain_knowledge)

                # Qwen3-VL 预处理
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # process_vision_info 在 Qwen3 中通常返回 image_inputs 和 video_inputs
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                # Compute logits
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits[:, -1, :]  # [1, vocab]
                    probs = torch.softmax(logits, dim=-1).squeeze(0)

                # Token ID 映射
                tok = processor.tokenizer
                
                def letter_token_id(letter: str):
                    # 针对 Qwen Tokenizer 的鲁棒性检查
                    candidates = [letter, ' ' + letter]
                    for cand in candidates:
                        ids = tok.encode(cand, add_special_tokens=False)
                        if len(ids) == 1:
                            return ids[0]
                    # Fallback
                    ids = tok.encode(letter, add_special_tokens=False)
                    return ids[-1] if ids else None

                available_letters = [k for k in options.keys() if isinstance(k, str) and len(k) == 1 and k.isalpha()]
                scores_map = {'A': '', 'B': '', 'C': '', 'D': ''}
                for ch in ['A', 'B', 'C', 'D']:
                    if ch in available_letters:
                        tid = letter_token_id(ch)
                        if tid is not None and 0 <= tid < probs.numel():
                            scores_map[ch] = float(probs[tid].item())
                        else:
                            scores_map[ch] = ''

                # Generate
                with torch.no_grad():
                    response_ids = model.generate(**inputs, max_new_tokens=128)
                
                response_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, response_ids)
                ]
                response = processor.batch_decode(response_ids_trimmed, skip_special_tokens=True)
                predicted = response[0].strip() if response else "N/A"

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

                del inputs, response_ids, response_ids_trimmed, response
                torch.cuda.empty_cache()

    for q_type, vals in metrics.items():
        valid_pairs = [(t, p) for t, p in zip(vals['y_true'], vals['y_pred']) if p != "N/A"]
        if not valid_pairs:
            acc = "N/A"
        else:
            y_true, y_pred = zip(*valid_pairs)
            y_pred_clean = [extract_answer(p) for p in y_pred]
            acc = accuracy_score(y_true, y_pred_clean)
        logging.info(f"\n📊 Question Type: {q_type} | Accuracy: {acc}")

    logging.info("✅ Evaluation completed.")

if __name__ == "__main__":
    evaluate_model()