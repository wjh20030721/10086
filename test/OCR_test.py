# test_qwen2vl_ocr.py
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ===== 配置 =====
MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda" 


IMAGE_PATHS = [
    "../Dataset/test/loan/repayment/4.webp",   
    "../Dataset/test/loan/bank statement/0.webp",
    "../Dataset/test/loan/refund/2.webp",   
]


GROUND_TRUTH = []
# 如果没有，设为：GROUND_TRUTH = []

# ===== 加载模型 =====
print("Loading Qwen2-VL-7B-Instruct...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

# ===== OCR 推理函数 =====
def extract_text_with_qwen2vl(image_path):
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Please extract all visible text from the image exactly as it appears, "
                        "including numbers, symbols, and line breaks. "
                        "Do not add any explanations, prefixes, or suffixes. Output only the raw text."
                    )
                }
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

# ===== 主测试循环 =====
print("\n" + "="*50)
print("Testing Qwen2-VL OCR on 2 images")
print("="*50)

for i, img_path in enumerate(IMAGE_PATHS):
    print(f"\nImage {i+1}: {img_path}")
    try:
        pred = extract_text_with_qwen2vl(img_path)
        print("Predicted Text:")
        print(pred)
        
        if GROUND_TRUTH and i < len(GROUND_TRUTH):
            gt = GROUND_TRUTH[i]
            print("\nGround Truth:")
            print(gt)
            
            match = pred == gt
            print(f"\nExact Match: {'Yes' if match else 'No'}")
            
            # 计算字符级相似度（可选）
            try:
                from rapidfuzz import fuzz
                sim = fuzz.ratio(pred, gt)
                print(f"Similarity: {sim:.1f}%")
            except ImportError:
                pass  # 如果没装 rapidfuzz，跳过相似度
                
    except Exception as e:
        print(f"Error: {e}")

print("\nOCR test completed.")