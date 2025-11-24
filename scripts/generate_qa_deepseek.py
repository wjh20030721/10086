import os
import json
import time
import argparse
from pathlib import Path
import requests

# ========== 配置区域 ==========
# 请在环境变量中设置你的 DeepSeek API Key，例如：
# export DEEPSEEK_API_KEY="sk-xxxxxx"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# DeepSeek API 兼容 OpenAI 风格的 chat completions
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"  # 替换为你实际使用的模型名

# 每次请求之间的最小间隔（秒），防止 QPS 过高
REQUEST_INTERVAL = 0.5


SYSTEM_PROMPT = """You are an expert in traffic analysis using drone imagery.

You will be given the file name of a traffic surveillance image captured by a drone.
You DO NOT see the real image, so you should imagine a plausible real-world traffic scene.

Your task is to generate **3 multiple-choice QA pairs** for this single image,
corresponding to the following tasks in order:
1. Traffic Congestion Detection  (binary classification: congested vs not congested)
2. Traffic Scene Description
3. Traffic Analysis

Requirements for EACH QA item:

[1] For the FIRST QA item (Traffic Congestion Detection):
- "type": must be exactly "Traffic Congestion Detection".
- "Options" must contain **ONLY two options**:
  - key "A": one possible answer.
  - key "B": the other possible answer.
- Do NOT include keys "C" or "D" in this first item's Options.
- "Answer": must be either "A" or "B" only.

[2] For the SECOND QA item (Traffic Scene Description):
- "type": must be exactly "Traffic Scene Description".
- "Options": a dictionary with keys "A", "B", "C", "D".
- "Answer": one of "A", "B", "C", or "D".

[3] For the THIRD QA item (Traffic Analysis):
- "type": must be exactly "Traffic Analysis".
- "Options": a dictionary with keys "A", "B", "C", "D".
- "Answer": one of "A", "B", "C", or "D".

General requirements:
- Each QA item must have:
  - "Question": a clear question in English.
  - "Answer": the correct option letter ("A", "B", "C", or "D" as required above).
  - "Options": a dictionary as specified above.
  - "type": one of the three task types listed.
  - "annotation": true

Answer diversity requirements:
- The correct option letter ("Answer") must not always be the same across all 3 QA items.
- Try to vary the correct letters, e.g., one of them is "A", another is "B" or "C", etc.
- Do NOT force all answers to be "A"; choose the correct option naturally according to the options you design.

Mimic the style and difficulty of the following example (DO NOT copy content, only follow format):

Example conversation for test/0000.jpg:
{
  "image_path": "test/0000.jpg",
  "conversation": [
    {
      "Question": "Is the traffic in this image congested?",
      "Answer": "B",
      "Options": {
        "A": "Yes.",
        "B": "No."
      },
      "type": "Traffic Congestion Detection",
      "annotation": true
    },
    {
      "Question": "Which description best matches the traffic scene in this image?",
      "Answer": "C",
      "Options": {
        "A": "A gridlocked multi-lane road with bumper-to-bumper vehicles and no movement.",
        "B": "Moderate congestion at a busy intersection with buses and long queues in every lane.",
        "C": "Light traffic on a multi-lane urban road; a full parking lot adjacent to a shopping mall with many pedestrians nearby.",
        "D": "An empty rural highway with no buildings, no pedestrians, and only a few scattered trees."
      },
      "type": "Traffic Scene Description",
      "annotation": true
    },
    {
      "Question": "What is the most likely implication for drivers on the main road?",
      "Answer": "A",
      "Options": {
        "A": "Traffic should remain smooth; brief slowdowns may occur near the parking-lot entrance as vehicles enter or exit.",
        "B": "Severe delays are expected due to multiple collisions blocking all lanes.",
        "C": "The road is closed for construction, requiring a long detour.",
        "D": "High-speed freeway conditions prevail with no cross-traffic or pedestrian activity."
      },
      "type": "Traffic Analysis",
      "annotation": true
    }
  ]
}

Now, ONLY output a JSON array of 3 objects, in the exact order:
1) Traffic Congestion Detection (binary A/B options),
2) Traffic Scene Description,
3) Traffic Analysis.

Do NOT add any explanations outside the JSON array.
Each object MUST have: "Question", "Answer", "Options", "type", "annotation".
"""


def build_user_prompt(image_rel_path: str) -> str:
    """
    image_rel_path: e.g., "test/0001.jpg"
    """
    return f"""The image file name is: {image_rel_path}.

Please imagine a realistic traffic scene that could correspond to this image
(e.g., an urban road, intersection, parking lot, or highway).

Based on that imagined scene, generate 3 QA items as described.
Remember: output ONLY a JSON array of 3 objects, no additional text.
"""


def call_deepseek(json_payload: dict) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in environment variables.")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=json_payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek API error: {resp.status_code} {resp.text}")

    data = resp.json()
    # OpenAI 风格：choices[0].message.content
    return data["choices"][0]["message"]["content"]


def parse_json_array(text: str):
    """
    从模型返回的字符串中提取 JSON 数组 [ {...}, {...}, {...} ]
    """
    text = text.strip()
    # 如果已经是数组开头
    if text.startswith("["):
        json_str = text
    else:
        # 尝试截取第一个 '[' 到最后一个 ']'
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return None
        json_str = text[start : end + 1]

    try:
        data = json.loads(json_str)
    except Exception:
        return None

    if not isinstance(data, list) or len(data) != 3:
        return None

    # 简单校验字段
    ok = True
    for item in data:
        if not isinstance(item, dict):
            ok = False
            break
        for key in ["Question", "Answer", "Options", "type", "annotation"]:
            if key not in item:
                ok = False
                break
        if not ok:
            break

    return data if ok else None


def generate_qa_for_image(image_rel_path: str) -> list | None:
    """
    调用 DeepSeek，为单张图片（用其相对路径标识）生成 3 条 QA。
    image_rel_path 例如 "test/0001.jpg"
    """
    user_prompt = build_user_prompt(image_rel_path)
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    raw = call_deepseek(payload)
    qa_list = parse_json_array(raw)
    if qa_list is None:
        print(f"[WARN] Failed to parse JSON for {image_rel_path}. Raw output (truncated): {raw[:200]!r}")
    return qa_list


def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs for VisDrone test images via DeepSeek API")
    parser.add_argument(
        "--images_dir",
        type=str,
        default="Dataset/VisDrone/test",
        help="Relative path to images directory (from project root)",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="Annotation/test.json",
        help="Relative path to test.json (from project root)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit number of images to process (for debugging)",
    )
    args = parser.parse_args()

    # 项目根目录：假设脚本在 scripts/ 下
    root = Path(__file__).resolve().parents[1]

    images_dir = (root / args.images_dir).resolve()
    ann_path = (root / args.annotation_file).resolve()

    if not images_dir.is_dir():
        raise SystemExit(f"[ERROR] Images dir not found: {images_dir}")
    if not ann_path.is_file():
        raise SystemExit(f"[ERROR] Annotation file not found: {ann_path}")

    print(f"[INFO] Using images_dir = {images_dir}")
    print(f"[INFO] Using annotation_file = {ann_path}")

    # 读取现有 test.json（保留其中条目，比如 test/0000.jpg）
    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    print(f"[INFO] Loaded {len(annotations)} existing entries from test.json")

    # 列出所有图片文件
    img_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    )
    if args.limit:
        img_files = img_files[: args.limit]

    print(f"[INFO] Found {len(img_files)} image files under {images_dir}")

    # 主循环
    for idx, img_path in enumerate(img_files, start=1):
        rel_key = f"test/{img_path.name}"  # JSON 中的 key 和 image_path 字段

        print(f"\n[{idx}/{len(img_files)}] Processing {rel_key} ...")

        # 已经有人类标注的（比如 test/0000.jpg）直接跳过
        if rel_key in annotations:
            print(f"[INFO] {rel_key} already exists in test.json, skip.")
            continue

        try:
            qa_list = generate_qa_for_image(rel_key)
            if not qa_list:
                print(f"[WARN] Skip {rel_key} due to parse failure.")
                continue

            annotations[rel_key] = {
                "image_path": rel_key,
                "conversation": qa_list,
            }
            print(f"[INFO] Added QA for {rel_key}")

            # 控制请求间隔，避免触发限流
            time.sleep(REQUEST_INTERVAL)

        except Exception as e:
            print(f"[ERROR] Exception when processing {rel_key}: {e}")
            continue

    # 写回 test.json（覆盖原文件，但保留原有条目）
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)

    print(f"\n[INFO] All done. Final entries in test.json: {len(annotations)}")
    print(f"[INFO] Updated file: {ann_path}")


if __name__ == "__main__":
    main()