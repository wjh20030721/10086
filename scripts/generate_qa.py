import os
import json
import time
import argparse
import random
import base64
from pathlib import Path
import requests

# ========== 配置区域 ==========
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
# 注意：原代码 URL 后有空格，这里已去除，防止请求错误
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen-vl-plus" 
REQUEST_INTERVAL = 1.0

# 定义允许的类别白名单
ALLOWED_CATEGORIES = [
    "pedestrian", "people", "bicycle", "car", "van", "truck", 
    "tricycle", "awning-tricycle", "bus", "motor"
]

def encode_image_to_base64(image_path: str) -> str:
    """将本地图片转为 base64 字符串（data URL 格式）"""
    with open(image_path, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode("utf-8")
        mime_type = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        return f"data:{mime_type};base64,{encoded}"

def build_multimodal_messages(image_abs_path: str) -> list:
    """构造包含图像和文本的 messages"""
    image_url = encode_image_to_base64(image_abs_path)
    
    # 将列表转换为字符串形式，方便嵌入 prompt
    categories_str = ", ".join(ALLOWED_CATEGORIES)
    
    # 修改了 Prompt，增加了对第一个任务的严格限制
    prompt_text = (
        "Analyze this drone-captured traffic image. "
        "Generate 3 multiple-choice QA pairs in the following order:\n\n"
        
        "1. Object Frequency Detection\n"
        "   - Goal: Identify the object category that appears most frequently.\n"
        f"   - CONSTRAINT: The Options (A, B, C, D) and the Answer MUST be strictly selected from this list only: [{categories_str}].\n"
        "   - Do NOT use adjectives (e.g., use 'car', NOT 'red car') or categories outside this list.\n\n"
        
        "2. Traffic Scene Description\n"
        "   - Goal: Describe the overall traffic density or environment.\n\n"
        
        "3. Traffic Analysis\n"
        "   - Goal: Analyze potential risks or traffic flow behavior.\n\n"
        
        "Output ONLY a JSON array of 3 objects with fields: type, Question, Options (A-D), Answer, annotation. No other text."
    )

    return [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt_text}
        ]}
    ]

def call_qwen_vl(json_payload: dict) -> str:
    if not QWEN_API_KEY:
        raise RuntimeError("QWEN_API_KEY is not set in environment variables.")
    
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    
    resp = requests.post(QWEN_API_URL, headers=headers, json=json_payload, timeout=120)
    
    if resp.status_code != 200:
        raise RuntimeError(f"Qwen-VL API error [{resp.status_code}]: {resp.text}")
    
    data = resp.json()
    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"Unexpected Qwen-VL API response: {data}")

def parse_json_array(text: str):
    text = text.strip()
    # 简单的 Markdown 代码块清洗，防止模型输出 ```json ... ```
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("\n", 1)[0]
    
    if text.startswith("["):
        json_str = text
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return None
        json_str = text[start:end+1]
    
    try:
        data = json.loads(json_str)
    except Exception as e:
        print(f"[JSON PARSE ERROR] {str(e)}")
        return None
    
    if not isinstance(data, list) or len(data) != 3:
        return None
    
    for item in data:
        if not isinstance(item, dict):
            return None
        required_keys = ["Question", "Answer", "Options", "type", "annotation"]
        if not all(k in item for k in required_keys):
            return None
    return data

# ========== 后处理函数（保持不变）==========
def _shuffle_options_and_adjust_answer(item: dict) -> dict:
    try:
        opts = item.get("Options")
        ans = item.get("Answer")
        if not isinstance(opts, dict) or ans not in opts:
            return item
        correct_text = opts[ans]
        letters = sorted(opts.keys())
        texts = list(opts.values())
        # 即使只有1个唯一选项也要处理，防止报错
        if len(set(texts)) >= 1: 
            random.shuffle(texts)
        new_opts = {letter: txt for letter, txt in zip(letters, texts)}
        for letter, txt in new_opts.items():
            if txt == correct_text:
                item["Answer"] = letter
                break
        item["Options"] = new_opts
    except Exception as e:
        print(f"[SHUFFLE ERROR] {str(e)}")
    return item

def _diversify_answer_letters(qa_list: list) -> list:
    try:
        answers = [q.get("Answer") for q in qa_list]
        if len(answers) == 3 and answers.count(answers[0]) == 3:
            for i in (1, 2):
                before = qa_list[i]["Answer"]
                qa_list[i] = _shuffle_options_and_adjust_answer(qa_list[i])
                after = qa_list[i]["Answer"]
                if after != before:
                    break
    except Exception as e:
        print(f"[DIVERSIFY ERROR] {str(e)}")
    return qa_list

def post_process_randomize(qa_list: list) -> list:
    if not qa_list or len(qa_list) != 3:
        return qa_list
    qa_list = [_shuffle_options_and_adjust_answer(item) for item in qa_list]
    qa_list = _diversify_answer_letters(qa_list)
    return qa_list

# ========== 主生成函数 ==========
def generate_qa_for_image(image_abs_path: str) -> list | None:
    messages = build_multimodal_messages(image_abs_path)
    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": 0.1,      # 降低 temperature 以确保严格遵守类别限制
        "max_tokens": 2000
    }
    
    raw = call_qwen_vl(payload)
    qa_list = parse_json_array(raw)
    
    if qa_list is None:
        print(f"[WARN] Failed to parse JSON. Raw output (truncated): {raw[:200]!r}")
    return qa_list

# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="Generate vision-grounded QA pairs using Qwen-VL")
    parser.add_argument("--images_dir", type=str, default="Dataset/VisDrone/test")
    parser.add_argument("--annotation_file", type=str, default="Annotation/test.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_randomize", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    images_dir = (root / args.images_dir).resolve()
    ann_path = (root / args.annotation_file).resolve()

    if not images_dir.is_dir():
        raise SystemExit(f"[ERROR] Images dir not found: {images_dir}")
    if not ann_path.is_file():
        print(f"[INFO] Annotation file not found, creating new one at {ann_path}")
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    print(f"[INFO] Loaded {len(annotations)} existing entries")

    img_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if args.limit:
        img_files = img_files[:args.limit]
    print(f"[INFO] Found {len(img_files)} image files")

    for idx, img_path in enumerate(img_files, start=1):
        rel_key = f"test/{img_path.name}"
        print(f"\n[{idx}/{len(img_files)}] Processing {rel_key} ...")

        if rel_key in annotations:
            print(f"[INFO] Already exists, skip.")
            continue

        try:
            qa_list = generate_qa_for_image(str(img_path))
            if not qa_list or len(qa_list) != 3:
                print(f"[WARN] Invalid QA, skip.")
                continue

            if not args.no_randomize:
                qa_list = post_process_randomize(qa_list)

            annotations[rel_key] = {
                "image_path": rel_key,
                "conversation": qa_list,
            }
            print(f"[INFO] Added QA for {rel_key}")

            time.sleep(REQUEST_INTERVAL)
        except Exception as e:
            print(f"[ERROR] Failed on {rel_key}: {e}")
            continue

    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)

    print(f"\n[INFO] Done. Total entries: {len(annotations)}")

if __name__ == "__main__":
    main()